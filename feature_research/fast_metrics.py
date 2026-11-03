"""
Vectorized TS-AUC + permutation null. Drop-in replacements for ts_auc /
_folded_ts_auc_table / permutation_null_ts_auc (same outputs, verified vs sklearn).

Key idea: within a t-slice, AUC = (sum of positive ranks - P(P+1)/2) / (P*N).
The within-t ranks of a feature do NOT depend on the labels, so for the
permutation null we rank each feature ONCE and every permutation is just two
np.bincount calls.
"""
import numpy as np
import pandas as pd


class TSRanker:
    """Precomputes the t-grouping once for a fixed row set (y may change later)."""

    def __init__(self, t):
        t = np.asarray(t)
        self.order_t = np.argsort(t, kind="mergesort")
        ts = t[self.order_t]
        _, self.gid_sorted = np.unique(ts, return_inverse=True)
        self.gid = np.empty_like(self.gid_sorted)
        self.gid[self.order_t] = self.gid_sorted          # group id per original row
        self.n_groups = int(self.gid_sorted.max()) + 1 if ts.size else 0

    def ranks(self, score):
        """Average within-t ranks (1-based, ties averaged) for every row, original order."""
        s = np.asarray(score, dtype=np.float64)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        o = np.lexsort((s, self.gid))                     # sort by group, then score
        g_o, s_o = self.gid[o], s[o]
        n = s.size
        new_run = np.empty(n, dtype=bool)
        new_run[0] = True
        new_run[1:] = (g_o[1:] != g_o[:-1]) | (s_o[1:] != s_o[:-1])
        new_grp = np.empty(n, dtype=bool)
        new_grp[0] = True
        new_grp[1:] = g_o[1:] != g_o[:-1]
        pos = np.arange(n, dtype=np.float64)
        grp_start = np.maximum.accumulate(np.where(new_grp, pos, 0.0))
        run_id = np.cumsum(new_run) - 1
        run_start = pos[new_run]
        run_len = np.diff(np.append(run_start, n))
        avg = (run_start - grp_start[new_run]) + (run_len + 1) / 2.0
        r = np.empty(n, dtype=np.float64)
        r[o] = avg[run_id]
        return r

    def auc_from_ranks(self, y, r):
        y = np.asarray(y).astype(np.float64)
        G = self.n_groups
        npos = np.bincount(self.gid, weights=y, minlength=G)
        ntot = np.bincount(self.gid, minlength=G).astype(np.float64)
        nneg = ntot - npos
        rsum = np.bincount(self.gid, weights=r * y, minlength=G)
        ok = (npos > 0) & (nneg > 0)
        if not ok.any():
            return 0.5
        w = npos[ok] * nneg[ok]
        auc = (rsum[ok] - npos[ok] * (npos[ok] + 1) / 2.0) / w
        return float((w * auc).sum() / w.sum())


def ts_auc(y, t, score):
    rk = TSRanker(t)
    return rk.auc_from_ranks(y, rk.ranks(score))


def _folded_ts_auc_table(X, y, t, feature_names, ranker=None):
    rk = ranker or TSRanker(t)
    rows = []
    for k, nm in enumerate(feature_names):
        a = rk.auc_from_ranks(y, rk.ranks(X[:, k]))
        rows.append((nm, max(a, 1 - a), a))
    return pd.DataFrame(rows, columns=["feature", "ts_auc_abs", "ts_auc_raw"])


def permutation_null_ts_auc(X, y, t, g, feature_idx, feature_names, n_perm=100, seed=0, ranker=None):
    """Same permutation scheme and same RNG stream as the original."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); t = np.asarray(t); g = np.asarray(g)
    uniq_series, inv = np.unique(g, return_inverse=True)
    S = uniq_series.size
    breaks_arr = np.zeros(S, dtype=np.int64)
    np.maximum.at(breaks_arr, inv, y.astype(np.int64))

    # per-series "cut" = tt[len(tt)//2] in the series' own row order (as the original)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=S)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    cut = t[order[starts + counts // 2]]
    cut_row = cut[inv]
    synth_y = (t >= cut_row).astype(y.dtype)
    orig_break_row = breaks_arr[inv] == 1

    rk = ranker or TSRanker(t)
    R = [rk.ranks(X[:, j]) for j in feature_idx]
    null = [[] for _ in feature_idx]
    for _ in range(n_perm):
        pb_row = rng.permutation(breaks_arr)[inv] == 1
        y_perm = np.where(pb_row, np.where(orig_break_row, y, synth_y), 0).astype(y.dtype)
        for k, r in enumerate(R):
            a = rk.auc_from_ranks(y_perm, r)
            null[k].append(max(a, 1 - a))
    rows = [(nm, float(np.mean(v)), float(np.percentile(v, 95))) for nm, v in zip(feature_names, null)]
    return pd.DataFrame(rows, columns=["feature", "null_mean", "null_p95"])
