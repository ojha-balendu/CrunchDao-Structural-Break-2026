"""PHASE 3 -- fold fast_metrics.py into the notebook.

`ts_auc` keeps its name and signature, so `_fit_and_score`, `train()`, the local-test
cell and the sup-KS comparison all pick up the fast version with no other change.

Run AFTER tools/make_reference_impl.py (which froze the sklearn-based originals).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell

TSRANKER = '''class TSRanker:
    """Groups rows by `t` once, so a TS-AUC over that row set costs two bincounts.

    Within a t-slice, AUC = (sum of positive ranks - P(P+1)/2) / (P*N). The within-t
    ranks of a feature do not depend on the labels at all, so the permutation null can
    rank each feature ONCE and then re-score every permutation from the same ranks --
    which is what turns an O(n_perm * n_features * n log n) loop into O(n_features *
    n log n) plus O(n_perm * n_features * n).

    Build one per row set and share it: `run_feature_diagnostic` hands the same
    instance to `_folded_ts_auc_table` and `permutation_null_ts_auc`."""

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
        """Row-pair-weighted mean of the per-t AUCs, from precomputed ranks."""
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
    """The leaderboard metric: AUC within each t-slice, weighted by that slice's
    positive-negative pair count. Same value as the sklearn roc_auc_score loop it
    replaces, without the per-slice Python call."""
    rk = TSRanker(t)
    return rk.auc_from_ranks(y, rk.ranks(score))


def _folded_ts_auc_table(X, y, t, feature_names, ranker=None):
    rk = ranker if ranker is not None else TSRanker(t)
    rows = []
    for k, nm in enumerate(feature_names):
        a = rk.auc_from_ranks(y, rk.ranks(X[:, k]))
        rows.append((nm, max(a, 1 - a), a))
    return pd.DataFrame(rows, columns=["feature", "ts_auc_abs", "ts_auc_raw"])
'''

PERM_NULL = '''def permutation_null_ts_auc(X, y, t, g, feature_idx, feature_names, n_perm=100, seed=0,
                             ranker=None):
    """Series-level label permutation: shuffle which SERIES are treated as
    breaking, keep each series' own row/t structure intact. A relabeled
    "breaker" reuses its own original first-break t as a synthetic tau (so the
    permuted series still has a well-defined post-break region); a real
    breaker relabeled non-breaking has all its rows set to y=0. Never shuffle
    rows independently -- that destroys the within-series correlation that is
    exactly why the naive (row-count) null is too tight.

    Same permutation scheme and the same RNG stream as the row-masking version it
    replaces: one rng.permutation(breaks_arr) per iteration, on an array of the same
    length. The per-series `g == sid` mask loop is gone -- the branch is now a
    vectorized np.where over rows, and the feature ranks are computed once up front
    because they do not depend on the labels."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); t = np.asarray(t); g = np.asarray(g)
    uniq_series, inv = np.unique(g, return_inverse=True)
    S = uniq_series.size
    breaks_arr = np.zeros(S, dtype=np.int64)
    np.maximum.at(breaks_arr, inv, y.astype(np.int64))

    # per-series synthetic tau = tt[len(tt) // 2] in the series' own row order
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=S)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    cut = t[order[starts + counts // 2]]
    synth_y = (t >= cut[inv]).astype(y.dtype)
    orig_break_row = breaks_arr[inv] == 1

    rk = ranker if ranker is not None else TSRanker(t)
    R = [rk.ranks(X[:, j]) for j in feature_idx]
    null = [[] for _ in feature_idx]
    for _ in range(n_perm):
        pb_row = rng.permutation(breaks_arr)[inv] == 1
        y_perm = np.where(pb_row, np.where(orig_break_row, y, synth_y), 0).astype(y.dtype)
        for k, r in enumerate(R):
            a = rk.auc_from_ranks(y_perm, r)
            null[k].append(max(a, 1 - a))

    rows = [(nm, float(np.mean(v)), float(np.percentile(v, 95)))
            for nm, v in zip(feature_names, null)]
    return pd.DataFrame(rows, columns=["feature", "null_mean", "null_p95"])
'''


def main():
    nb = nbedit.load()

    i = find_cell(nb, "def ts_auc(", "code")
    text = src(nb["cells"][i])
    # Drop the old table function FIRST: the TSRANKER block defines its own, and
    # remove_blocks() matches the first occurrence -- doing this after the insert
    # would delete the new one and leave the old.
    text, _ = nbedit.remove_blocks(text, ["def _folded_ts_auc_table("])
    text = nbedit.replace_block(text, "def ts_auc(", TSRANKER)
    text = nbedit.replace_block(text, "def permutation_null_ts_auc(", PERM_NULL)

    # one ranker for the whole diagnostic, shared by the table and the null
    old = "    diag = _folded_ts_auc_table(X, y, t, feature_names)\n"
    assert old in text
    text = text.replace(old,
                        "    # one grouping of the rows by t, reused by the table and the null\n"
                        "    ranker = TSRanker(t)\n"
                        "    diag = _folded_ts_auc_table(X, y, t, feature_names, ranker=ranker)\n")

    old = ("    null_df = permutation_null_ts_auc(X, y, t, g, perm_idx, perm_features, "
           "n_perm=n_perm, seed=seed)\n")
    assert old in text
    text = text.replace(old,
                        "    null_df = permutation_null_ts_auc(X, y, t, g, perm_idx, perm_features,\n"
                        "                                       n_perm=n_perm, seed=seed, ranker=ranker)\n")
    set_src(nb["cells"][i], text)

    # the section-3 markdown still says "unchanged"
    i = find_cell(nb, "## 3. TS-AUC, and a per-feature diagnostic", "markdown")
    text = src(nb["cells"][i])
    text = text.replace(
        "`ts_auc` is the leaderboard metric, unchanged.",
        "`ts_auc` is the leaderboard metric. Same number as the sklearn `roc_auc_score`\n"
        "loop it replaces, computed from within-`t` ranks instead: inside a `t`-slice,\n"
        "`AUC = (sum of positive ranks - P(P+1)/2) / (P*N)`. `TSRanker` does the grouping by\n"
        "`t` once and `run_feature_diagnostic` shares one instance between the results table\n"
        "and the permutation null.\n\n"
        "That sharing is what makes the null cheap: **the within-`t` ranks of a feature do not\n"
        "depend on the labels**, so each feature is ranked once and every permutation is two\n"
        "`np.bincount` calls over those same ranks. The permutation scheme, the RNG stream and\n"
        "the output are identical to the previous version.")
    set_src(nb["cells"][i], text)

    nbedit.save(nb)
    print("phase 3 applied")


if __name__ == "__main__":
    main()
