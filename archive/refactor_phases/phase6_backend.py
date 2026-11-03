"""PHASE 6 -- honor build_dataset(backend=), and survive crunch_tools.test().

Three changes:

  1. `backend=` was accepted and then ignored (the call hard-coded backend="loky").
     It is now honored, with a module-level EXTRACT_BACKEND default.

  2. A fallback chain. loky re-imports __main__ in every worker; under
     crunch_tools.test() that import points at a path which does not exist here
     (the observed failure was FileNotFoundError: /kaggle/working/main.py), so the
     workers die on start-up. "auto" now probes for that, falls back to a fork Pool,
     and finally to serial.

  3. The _USE_GLOBAL_ACTIVE sentinel is resolved in the parent before dispatch.
     It is a bare object(); pickling it to a worker produces a DIFFERENT object there,
     so the `is` check would fail and the sentinel would be read as a set of family
     names. Resolving first means only None or a real set ever crosses a process
     boundary.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell, new_markdown_cell

BUILD_CELL = r'''KEEP_EVERY = ((24, 1), (128, 2), (512, 4), (10 ** 9, 8))   # (up_to_t, stride)

import functools
import multiprocessing
import sys

_USE_GLOBAL_ACTIVE = object()   # sentinel: "use whatever ACTIVE_FAMILIES is right now"

# "auto" | "loky" | "fork" | "threading" | "serial"
# "auto" is the offline default: try loky, fall back to a fork Pool, then to serial.
# Naming one explicitly honors it with NO fallback, which is what you want when you
# are diagnosing a backend rather than getting work done.
EXTRACT_BACKEND = "auto"
FORK_CHUNKSIZE = 8


def _stride_at(t: int) -> int:
    for up_to, stride in KEEP_EVERY:
        if t < up_to:
            return stride
    return KEEP_EVERY[-1][1]


def extract_series(x_hist, x_online, tau=None, subsample=True, active_families=_USE_GLOBAL_ACTIVE):
    """Run the extractor over one series. Returns (X, y, tvec, w).

    active_families: _USE_GLOBAL_ACTIVE (default) resolves to the current
    ACTIVE_FAMILIES global at call time (this is what train()/infer() want --
    fast, submission-path extraction). Pass None explicitly to force the FULL
    registry regardless of the global (this is what the offline cache/diagnostic/
    ablate loop wants -- always extract everything, select columns after)."""
    fams = ACTIVE_FAMILIES if active_families is _USE_GLOBAL_ACTIVE else active_families
    fx = StreamFeatures(x_hist, active_families=fams)
    rows, ys, ts, ws = [], [], [], []
    for t, point in enumerate(x_online):
        f = fx.update(point)
        if subsample:
            stride = _stride_at(t)
            if t % stride != 0:
                continue
            ws.append(float(stride))
        else:
            ws.append(1.0)
        rows.append(list(f))
        ys.append(0 if (tau is None or t < tau) else 1)
        ts.append(t)
    X = np.asarray(rows, dtype=np.float32)
    return X, np.asarray(ys, np.int8), np.asarray(ts, np.int32), np.asarray(ws, np.float32)


def _extract_one(rec, subsample=True, active_families=None):
    """Module-level on purpose: multiprocessing.Pool pickles the callable BY REFERENCE,
    and a closure defined inside build_dataset cannot be pickled at all."""
    did, xh, xo, tau = rec
    X, y, t, w = extract_series(xh, xo, tau, subsample=subsample,
                                 active_families=active_families)
    g = np.full(t.shape, did, dtype=np.int64)
    return X, y, t, w, g


def _loky_can_start() -> bool:
    """loky spawns fresh interpreters that re-import __main__. crunch_tools.test()
    installs a process initializer pointing __main__ at a module path that does not
    exist in this environment, and every worker then dies with
    FileNotFoundError: /kaggle/working/main.py before running any of our code.

    If __main__ claims a __file__ that isn't on disk, spawning is going to fail, so
    don't pay for the attempt."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return not (path is not None and not os.path.exists(path))


def _backend_chain(backend):
    if backend != "auto":
        return [backend]                       # explicit request: no silent fallback
    chain = []
    if _loky_can_start():
        chain.append("loky")
    if "fork" in multiprocessing.get_all_start_methods():
        chain.append("fork")
    chain.append("serial")
    return chain


def _run_fork(fn, items, n_jobs):
    ctx = multiprocessing.get_context("fork")
    procs = None if n_jobs in (-1, 0) else n_jobs
    with ctx.Pool(processes=procs) as pool:
        return pool.map(fn, items, chunksize=FORK_CHUNKSIZE)


def _stack(res):
    return (np.concatenate([r[0] for r in res]),
            np.concatenate([r[1] for r in res]),
            np.concatenate([r[2] for r in res]),
            np.concatenate([r[3] for r in res]),
            np.concatenate([r[4] for r in res]))


def build_dataset(datasets, subsample=True, max_series=None, n_jobs=-1, verbose=1,
                   active_families=_USE_GLOBAL_ACTIVE, backend=None):
    """Extract many series in parallel. `backend` overrides EXTRACT_BACKEND for this
    call; None means "use the global".

    Each family is .clone()-d off the REGISTRY prototype (see FeatureFamily.clone()),
    so concurrent extraction never shares mutable per-series state.
    """
    items = list(datasets)[: max_series] if max_series else list(datasets)
    if not items:
        raise ValueError("build_dataset got no series")

    # Resolve the sentinel HERE, in the parent. It is a bare object(); pickled into a
    # worker it becomes a different object, the `is` test in extract_series fails, and
    # the sentinel gets read as a set of family names.
    fams = ACTIVE_FAMILIES if active_families is _USE_GLOBAL_ACTIVE else active_families
    worker = functools.partial(_extract_one, subsample=subsample, active_families=fams)

    if n_jobs == 1 or len(items) == 1:
        return _stack([worker(r) for r in items])

    chain = _backend_chain(backend or EXTRACT_BACKEND)
    errors = []
    for b in chain:
        try:
            if b == "serial":
                if verbose:
                    print("build_dataset: serial over %d series" % len(items))
                return _stack([worker(r) for r in items])
            if b == "fork":
                if verbose:
                    print("build_dataset: fork Pool over %d series (chunksize %d)"
                          % (len(items), FORK_CHUNKSIZE))
                return _stack(_run_fork(worker, items, n_jobs))
            res = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=8, backend=b)(
                joblib.delayed(worker)(r) for r in items)
            return _stack(res)
        except Exception as exc:
            errors.append((b, type(exc).__name__, str(exc)[:200]))
            if b == chain[-1]:
                raise RuntimeError(
                    "every extraction backend failed: %s"
                    % "; ".join("%s -> %s: %s" % e for e in errors)) from exc
            print("build_dataset: backend %r failed (%s: %s) -- falling back to %r"
                  % (b, type(exc).__name__, str(exc)[:160], chain[chain.index(b) + 1]))
'''

BACKEND_MD = '''### Extraction backend

`build_dataset(backend=)` used to be accepted and then ignored -- the call hard-coded
`backend="loky"`. It is honored now, defaulting to the module-level `EXTRACT_BACKEND`.

**Why there is a fallback chain.** loky spawns fresh interpreters, and every worker
re-imports `__main__`. Under `crunch_tools.test()` that import resolves to a path which
does not exist in this environment, and the workers die before running any of our code:

```
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/main.py'
```

That is crunch's process-initializer monkey-patch, not a bug in the extractor, and it
is not something this notebook can fix from the inside. So `EXTRACT_BACKEND = "auto"`:

1. **loky** -- the offline default. Real process parallelism, and it survives the
   notebook namespace because joblib pickles with cloudpickle. Skipped up front when
   `__main__.__file__` points at a file that is not on disk, which is exactly the
   crunch-runner situation, so we don't pay for a spawn that is going to fail.
2. **fork `Pool`** (`multiprocessing.get_context("fork")`, `chunksize=8`) -- Linux and
   macOS. Workers inherit the interpreter image instead of re-importing `__main__`, so
   crunch's patched initializer never runs. Offered only when `fork` is in
   `multiprocessing.get_all_start_methods()`.
3. **serial** -- always works.

Naming a backend explicitly (`build_dataset(..., backend="fork")`) honors it with **no**
fallback: when you are diagnosing a backend you want the failure, not a silent downgrade.

Two things this needed. `_extract_one` is module-level rather than a closure inside
`build_dataset`, because `multiprocessing.Pool` pickles the callable by reference and a
closure cannot be pickled at all. And the `_USE_GLOBAL_ACTIVE` sentinel is resolved in
the parent before dispatch: it is a bare `object()`, so a pickled copy is a *different*
object in the worker, the `is` check in `extract_series` fails, and the sentinel gets
read as a set of family names. Only `None` or a real set crosses a process boundary now.
'''


def main():
    nb = nbedit.load()
    i = find_cell(nb, "def build_dataset(", "code")
    set_src(nb["cells"][i], BUILD_CELL)
    nb["cells"].insert(i, new_markdown_cell(BACKEND_MD))
    nbedit.save(nb)
    print("phase 6 applied")


if __name__ == "__main__":
    main()
