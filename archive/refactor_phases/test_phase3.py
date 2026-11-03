"""PHASE 3 test -- fast TS-AUC / permutation null vs. the frozen sklearn originals.

Synthetic matrix with >= 300 series, ties in some columns, and one constant column.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nb_exec import run_notebook, make_synthetic
from reference_impl import build_reference


def make_matrix(n_series=350, n_steps=26, seed=3):
    """Rows shaped like the real cache: many series x a shared t grid, half breaking."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_series), n_steps)
    t = np.tile(np.arange(n_steps), n_series)
    breaks = rng.random(n_series) < 0.5
    tau = rng.integers(n_steps // 4, 3 * n_steps // 4, size=n_series)
    y = ((t >= np.repeat(tau, n_steps)) & np.repeat(breaks, n_steps)).astype(np.int8)

    n = g.size
    cols, names = [], []

    cols.append(rng.normal(size=n) + 1.2 * y);              names.append("strong")
    cols.append(rng.normal(size=n) + 0.15 * y);             names.append("weak")
    cols.append(rng.normal(size=n));                        names.append("null")
    # heavy ties: only 5 distinct values, so tie-averaged ranks are load-bearing
    cols.append(np.round(rng.normal(size=n) + 0.6 * y));    names.append("tied_coarse")
    cols.append((rng.random(n) < 0.3).astype(float));       names.append("tied_binary")
    cols.append(np.full(n, 7.0));                           names.append("constant")
    cols.append(t.astype(float));                           names.append("pure_t")

    X = np.column_stack(cols).astype(np.float32)
    return X, y, t, g, names


def main():
    print("=" * 74)
    print("PHASE 3 TEST -- fast metrics vs. frozen sklearn reference")
    print("=" * 74)

    ns = run_notebook(synthetic_train_data=make_synthetic(1, seed=0))
    ref = build_reference(ns)

    X, y, t, g, names = make_matrix()
    print("\nmatrix: %d rows, %d series, %d t-values, %d columns"
          % (X.shape[0], len(np.unique(g)), len(np.unique(t)), X.shape[1]))
    print("columns:", ", ".join(names))
    print("  ('tied_coarse'/'tied_binary' are heavily tied, 'constant' has std 0)\n")

    # ------------------------------------------------------------ folded table
    t0 = time.perf_counter(); tab_ref = ref["_folded_ts_auc_table"](X, y, t, names)
    dt_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); tab_new = ns["_folded_ts_auc_table"](X, y, t, names)
    dt_new = time.perf_counter() - t0

    d_abs = np.abs(tab_ref["ts_auc_abs"].to_numpy() - tab_new["ts_auc_abs"].to_numpy()).max()
    d_raw = np.abs(tab_ref["ts_auc_raw"].to_numpy() - tab_new["ts_auc_raw"].to_numpy()).max()
    print("-- folded TS-AUC table --")
    print("  %-14s %12s %12s %10s" % ("feature", "reference", "fast", "abs diff"))
    for k, nm in enumerate(names):
        a, b = tab_ref["ts_auc_raw"][k], tab_new["ts_auc_raw"][k]
        print("  %-14s %12.9f %12.9f %10.2e" % (nm, a, b, abs(a - b)))
    print("\n  max abs diff: ts_auc_abs %.3e | ts_auc_raw %.3e" % (d_abs, d_raw))
    print("  time: reference %.3fs -> fast %.3fs (%.1fx)" % (dt_ref, dt_new, dt_ref / dt_new))
    assert d_abs < 1e-12 and d_raw < 1e-12, "table mismatch"
    assert list(tab_ref["feature"]) == list(tab_new["feature"])
    print("  [ok] < 1e-12")

    # ------------------------------------------------------------ scalar ts_auc
    worst = 0.0
    for k in range(X.shape[1]):
        s = np.nan_to_num(X[:, k].astype(np.float64))
        worst = max(worst, abs(ref["ts_auc"](y, t, s) - ns["ts_auc"](y, t, s)))
    print("\n-- scalar ts_auc (the _fit_and_score / train() entry point) --")
    print("  max abs diff over %d columns: %.3e" % (X.shape[1], worst))
    assert worst < 1e-12
    print("  [ok] < 1e-12")

    # ------------------------------------------------------- permutation null
    idx = list(range(len(names)))
    for n_perm in (30, 100):
        t0 = time.perf_counter()
        nl_ref = ref["permutation_null_ts_auc"](X, y, t, g, idx, names, n_perm=n_perm, seed=0)
        dt_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        nl_new = ns["permutation_null_ts_auc"](X, y, t, g, idx, names, n_perm=n_perm, seed=0)
        dt_new = time.perf_counter() - t0

        dm = np.abs(nl_ref["null_mean"].to_numpy() - nl_new["null_mean"].to_numpy()).max()
        dp = np.abs(nl_ref["null_p95"].to_numpy() - nl_new["null_p95"].to_numpy()).max()
        print("\n-- permutation null, n_perm=%d, seed=0 --" % n_perm)
        print("  %-14s %12s %12s %12s %12s"
              % ("feature", "ref mean", "fast mean", "ref p95", "fast p95"))
        for k, nm in enumerate(names):
            print("  %-14s %12.9f %12.9f %12.9f %12.9f"
                  % (nm, nl_ref["null_mean"][k], nl_new["null_mean"][k],
                     nl_ref["null_p95"][k], nl_new["null_p95"][k]))
        print("\n  max abs diff: null_mean %.3e | null_p95 %.3e" % (dm, dp))
        print("  CUTOFF: reference %.9f | fast %.9f"
              % (nl_ref["null_p95"].max(), nl_new["null_p95"].max()))
        print("  time: reference %.2fs -> fast %.2fs (%.0fx)" % (dt_ref, dt_new, dt_ref / dt_new))
        assert dm < 1e-12 and dp < 1e-12, "null mismatch at n_perm=%d" % n_perm
        print("  [ok] < 1e-12")

    # ------------------------------------------------- one shared ranker path
    print("\n-- run_feature_diagnostic (one shared TSRanker) --")
    w = np.ones(X.shape[0], dtype=np.float32)
    ns["FEATURE_TO_FAMILY"] = {nm: nm for nm in names}
    ns["get_cached_matrix"] = lambda datasets, **kw: (X, y, t, w, g, names)
    t0 = time.perf_counter()
    diag, rollup, cutoff, _ = ns["run_feature_diagnostic"](
        list(range(len(np.unique(g)))), seed=0, n_series=len(np.unique(g)), n_perm=50)
    print("  wall clock: %.2fs" % (time.perf_counter() - t0))
    print()
    print(diag[["feature", "ts_auc_abs", "null_p95", "above_null"]].to_string(index=False))
    assert float(diag.loc[diag["feature"] == "constant", "ts_auc_abs"].iloc[0]) == 0.5

    print("\n" + "=" * 74)
    print("PHASE 3: PASS")
    print("=" * 74)


if __name__ == "__main__":
    main()
