"""PHASE 6 test -- build_dataset honors backend= and falls back correctly."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nb_exec import run_notebook, make_synthetic


def main():
    print("=" * 74)
    print("PHASE 6 TEST -- extraction backends")
    print("=" * 74)

    data = make_synthetic(12, seed=44)
    # as_main=True reproduces Jupyter's pickling semantics -- see run_notebook's
    # docstring; loky needs by-value pickling and the fork Pool needs by-reference,
    # and only a real __main__ gives you both.
    ns = run_notebook(synthetic_train_data=data, as_main=True)
    bd = ns["build_dataset"]

    print("\nEXTRACT_BACKEND default: %r" % ns["EXTRACT_BACKEND"])
    print("fork available: %r" % ("fork" in __import__("multiprocessing").get_all_start_methods()))
    print("_loky_can_start(): %r" % ns["_loky_can_start"]())

    ref = None
    results = {}
    for backend in ("serial", "loky", "fork", "threading", "auto"):
        t0 = time.perf_counter()
        try:
            out = bd(data, backend=backend, verbose=0)
            dt = time.perf_counter() - t0
            results[backend] = out
            same = "-" if ref is None else str(all(
                np.array_equal(a, b) for a, b in zip(ref, out)))
            if ref is None:
                ref = out
            print("  %-10s ok  %s  %6.2fs  identical-to-serial=%s"
                  % (backend, out[0].shape, dt, same))
        except Exception as exc:
            print("  %-10s FAILED %s: %s" % (backend, type(exc).__name__, str(exc)[:120]))
            raise

    for b, out in results.items():
        for a, c in zip(ref, out):
            assert np.array_equal(a, c), "%s produced different numbers" % b
    print("\n  [ok] every backend produces bit-identical output")

    # ------------------------------------------------ explicit backend does NOT fall back
    print("\n-- an explicit bad backend must raise, not silently downgrade --")
    try:
        bd(data[:4], backend="nonsense-backend", verbose=0)
        raise SystemExit("FAIL: bad backend was silently accepted")
    except RuntimeError as exc:
        print("  raised RuntimeError as expected:")
        print("    %s" % str(exc)[:170])

    # ------------------------------------------------ simulated crunch: loky is skipped
    print("\n-- simulated crunch_tools.test() environment --")
    main_mod = sys.modules["__main__"]
    old = getattr(main_mod, "__file__", None)
    main_mod.__file__ = "/kaggle/working/main.py"          # the path crunch patches in
    try:
        print("  __main__.__file__ = %r (does not exist)" % main_mod.__file__)
        print("  _loky_can_start(): %r" % ns["_loky_can_start"]())
        assert ns["_loky_can_start"]() is False
        chain = ns["_backend_chain"]("auto")
        print("  auto chain is now: %s" % chain)
        assert "loky" not in chain, "loky should have been skipped"
        t0 = time.perf_counter()
        out = bd(data, backend="auto", verbose=1)
        print("  extraction still succeeded: %s in %.2fs" % (out[0].shape,
                                                             time.perf_counter() - t0))
        for a, c in zip(ref, out):
            assert np.array_equal(a, c)
        print("  [ok] identical output via the fallback path")
    finally:
        if old is None:
            del main_mod.__file__
        else:
            main_mod.__file__ = old

    # ------------------------------------------------ the sentinel never crosses a process
    print("\n-- _USE_GLOBAL_ACTIVE is resolved in the parent --")
    ns["ACTIVE_FAMILIES"] = {"e2", "z2", "time"}
    for backend in ("loky", "fork", "serial"):
        out = bd(data[:4], backend=backend, verbose=0)     # sentinel default
        print("  %-8s ACTIVE_FAMILIES={'e2','z2','time'} -> %d columns"
              % (backend, out[0].shape[1]))
        assert out[0].shape[1] == 22, "sentinel not resolved on backend %r" % backend
    ns["ACTIVE_FAMILIES"] = None
    print("  [ok] all backends honored the global, none saw the sentinel object")

    print("\n" + "=" * 74)
    print("PHASE 6: PASS")
    print("=" * 74)


if __name__ == "__main__":
    main()
