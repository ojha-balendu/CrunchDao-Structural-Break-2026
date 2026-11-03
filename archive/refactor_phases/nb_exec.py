"""Pull the code cells out of the notebook and run them in a plain namespace.

The competition data is not available offline, so every test in tools/ drives the
notebook's own code with synthetic series instead. Cells that only exist to talk
to the crunch runtime (`import crunch`, `crunch_tools.load_data()`,
`crunch_tools.test()`) are skipped, as are the plotting / post-hoc-analysis cells
at the bottom that need a trained model artifact or the `merged` DataFrame.

Usage:
    from nb_exec import run_notebook, make_synthetic
    ns = run_notebook()                      # namespace with REGISTRY, StreamFeatures, ...
    data = make_synthetic(20, seed=0)
"""

import json
import os
import re
import sys

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "new-structural (5).ipynb")

# A cell is skipped when any of these appears in its source. These are exactly the
# cells that need the crunch runtime, a trained artifact, or notebook-only globals.
SKIP_PATTERNS = (
    "import crunch",
    "crunch_tools.load_data",
    "import matplotlib",
    "glob.glob(\"**/model.joblib\"",
)

# Skip a cell that CALLS the crunch runner at top level, but not one that merely
# defines a wrapper around it -- section 3d's local_test() is such a wrapper.
_CRUNCH_CALL_RE = re.compile(r"^\s{0,3}(?:\w+\s*=\s*)?crunch_tools\.test\s*\(", re.M)

# Long-running or plotting DRIVER cells: the definitions above them are what the
# tests need, not the call itself. Tests drive those functions directly instead.
DRIVER_SKIP = (
    "bench_extractor(train_data)",
    "= run_feature_diagnostic(",    # the CALL cell, not the `def` cell
    "family_rollup[",
    "plot_feature_auc_by_d(",
    "prediction = pd.read_parquet",
    "merged",              # the TEST 1/2/3 plot cells, which need infer() output
)


def load_code_cells(nb_path=NB_PATH):
    """[(cell_index, source)] for every code cell, in notebook order."""
    with open(nb_path) as fh:
        nb = json.load(fh)
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        out.append((i, "".join(c["source"])))
    return out


_MAGIC_RE = re.compile(r"^(%{1,2}[A-Za-z]\w*|![^=])")


def _strip_magics(src):
    """Drop IPython magics / shell escapes, which plain exec() cannot parse.

    Only column-0 lines count: an indented line starting with `%` is a string
    continuation of a `... % (args)` format expression, not a magic.
    """
    keep = []
    for line in src.split("\n"):
        if _MAGIC_RE.match(line):
            keep.append("pass  # (magic stripped) " + line)
        else:
            keep.append(line)
    return "\n".join(keep)


def should_skip(src, extra_skip=()):
    if _CRUNCH_CALL_RE.search(src):
        return True
    for pat in SKIP_PATTERNS:
        if pat in src:
            return True
    for pat in extra_skip:
        if pat in src:
            return True
    return False


def run_notebook(nb_path=NB_PATH, upto=None, extra_skip=DRIVER_SKIP, ns=None, verbose=False,
                 synthetic_train_data=None, as_main=False):
    """Exec the notebook's code cells into `ns` (a fresh dict by default).

    upto: stop after this cell index (inclusive). None = run every eligible cell.
    synthetic_train_data: bound to the name `train_data` before execution, so
        driver cells like `bench_extractor(train_data)` work on synthetic series.
    as_main: exec into the real `__main__` module instead of a private dict.

    A note on `as_main`, because it decides whether the extraction backends work.
    In Jupyter the notebook's globals ARE the `__main__` module, and the two process
    backends need opposite things from that:

      * loky spawns a fresh interpreter, so the worker function has to be pickled BY
        VALUE. cloudpickle does that only for functions whose `__module__` is
        `__main__` (or is otherwise not importable).
      * multiprocessing's fork Pool uses the stdlib pickler, which serializes a
        function BY REFERENCE as "<module>.<qualname>" -- so that module must be
        resolvable in the child. After a fork it is, because the child inherited
        `__main__`.

    The default private-dict namespace satisfies the first and not the second; a named
    module in sys.modules satisfies the second and not the first. Only `__main__`
    satisfies both, which is why it works in the real notebook and why the Phase 6 test
    passes `as_main=True`.
    """
    if ns is None:
        ns = sys.modules["__main__"].__dict__ if as_main else {"__name__": "__nb__"}
    if synthetic_train_data is not None:
        ns["train_data"] = synthetic_train_data

    for i, src in load_code_cells(nb_path):
        if upto is not None and i > upto:
            break
        if should_skip(src, extra_skip):
            if verbose:
                print("  [skip] cell %d" % i)
            continue
        code = _strip_magics(src)
        if verbose:
            print("  [run ] cell %d" % i)
        try:
            exec(compile(code, "<nbcell %d>" % i, "exec"), ns)
        except Exception as exc:
            raise RuntimeError("notebook cell %d failed: %s: %s" % (i, type(exc).__name__, exc)) from exc
    return ns


# ---------------------------------------------------------------------------
#  Synthetic series
# ---------------------------------------------------------------------------
def make_synthetic(n_series=20, seed=0, tie_every=3):
    """Student-t series that look like the competition's shape.

    - historical segment: 300-2000 points
    - online segment: 600-1300 points
    - half the series get a scale change at a random tau (a real structural break)
    - every `tie_every`-th series is rounded to 2 decimals, which forces heavy
      ties -- the path that separates tie-averaged ranks from plain argsort.

    Returns [(series_id, x_hist, x_online, tau)], tau=None for a non-breaking series.
    Ids are ints because build_dataset packs them into an int64 `g` vector.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_series):
        n_h = int(rng.integers(300, 2001))
        n_o = int(rng.integers(600, 1301))
        df = float(rng.uniform(3.0, 12.0))
        scale = float(rng.uniform(0.5, 2.0))
        mu = float(rng.normal(0.0, 0.5))

        xh = mu + scale * rng.standard_t(df, size=n_h)
        xo = mu + scale * rng.standard_t(df, size=n_o)

        if i % 2 == 0:                       # half of them break
            tau = int(rng.integers(n_o // 5, 4 * n_o // 5))
            bump = float(rng.choice([1.6, 2.2, 0.55]))
            xo[tau:] = mu + scale * bump * rng.standard_t(df, size=n_o - tau)
        else:
            tau = None

        if i % tie_every == 0:               # a third of them are heavily tied
            xh = np.round(xh, 2)
            xo = np.round(xo, 2)

        out.append((1000 + i, xh.astype(np.float64), xo.astype(np.float64), tau))
    return out
