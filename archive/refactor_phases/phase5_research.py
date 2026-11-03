"""PHASE 5 -- section 3d, the one-call research loop.

Adds research_step() / local_test(), and splits DIAG_SEED / DIAG_N_SERIES out of the
diagnostic's run cell so research_step's default n_series works without having run the
10k diagnostic first.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell, new_code_cell, new_markdown_cell

DIAG_CONST_CELL = '''# Sample size / seed for the diagnostic AND for research_step (section 3d), so a
# research step scores its candidate on exactly the series the diagnostic reported on.
DIAG_SEED = 0
DIAG_N_SERIES = 2500
'''

DIAG_RUN_CELL = '''diag, family_rollup, CUTOFF, diag_cache = run_feature_diagnostic(
    train_data, seed=DIAG_SEED, n_series=DIAG_N_SERIES, n_perm=100,
)
Xd, yd, td, wd, gd, FEATURES_D = diag_cache   # full-registry matrix for the sampled series --
                                               # reusable by quick_train/ablate (section 3c) and
                                               # by plot_feature_auc_by_d (bottom of notebook)
print("CUTOFF (permutation-null P95):", CUTOFF)
diag.head(30)
'''

RESEARCH_MD = '''## 3d. Research loop -- one call per idea

`research_step` is the whole keep/drop decision for one candidate family, in one call:
smoke test -> extract only what is missing -> univariate TS-AUC vs. the permutation-null
cutoff -> paired K-fold CV against the current registry. Minutes, not hours, because
every expensive thing it touches is cached by section 2b.

```python
class MyFamily(TwoSampleFamily): ...
research_step(MyFamily())                  # minutes, not hours
research_step(MyFamily(), register=True)   # keep it
local_test()                               # occasional end-to-end check
```

**What each stage costs on a repeat call.** Extraction: only the candidate family
(everything else is a `.npy` load). Univariate: the permutation null is computed once
per (sample, baseline, n_perm, seed) and cached in `null.joblib`, so the cutoff is a
dict lookup. CV: the baseline's per-fold AUCs are cached by (feature set, folds, LGB
params), so only the candidate model is trained.

**The folds are fixed by `seed`, not resampled.** Baseline and candidate see exactly
the same series in exactly the same folds, so the per-fold differences are *paired* and
the right noise estimate is the std of the per-fold deltas -- not the std of either
arm's AUCs, which is dominated by fold-to-fold difficulty that cancels in the pairing.
The verdict line uses that paired std.

`fast_lgb=True` (the default) uses `FAST_LGB` -- `learning_rate=0.1`, `max_bin=63`,
`num_boost_round=400`, `early_stopping(30)` -- which is for *ranking candidates against
each other*, not for estimating the final score. Re-run the finalists with
`fast_lgb=False` (`FINAL_LGB`: the real `LGB_PARAMS` / `NUM_ROUNDS`) before committing.

A `KEEP` verdict here is evidence, not proof: it is one seed, one sample of
`n_series`, and a subsampled validation set (section 3b's note applies). Confirm a real
decision with `fast_lgb=False`, a second `seed`, and finally `local_test()`.
'''

RESEARCH_CELL = r'''# =====================================================================
#  3d. RESEARCH LOOP
# =====================================================================
import time

# Two named fit configurations. `overrides` are applied on top of LGB_PARAMS, which is
# defined in section 4 -- read at call time, never at import time, so this cell can sit
# above it.
FAST_LGB = dict(
    overrides=dict(learning_rate=0.1, max_bin=63),
    num_boost_round=400,
    early_stopping=30,
)
FINAL_LGB = dict(
    overrides={},
    num_boost_round=None,        # None -> NUM_ROUNDS
    early_stopping=80,
)


def _lgb_config(fast_lgb=True):
    cfg = FAST_LGB if fast_lgb else FINAL_LGB
    params = dict(LGB_PARAMS)
    params.update(cfg["overrides"])
    return params, cfg["num_boost_round"] or NUM_ROUNDS, cfg["early_stopping"]


def _sample_for(datasets, n_series, seed):
    """The same seeded sample run_feature_diagnostic draws, so a research step scores
    its candidate on the series the diagnostic reported on."""
    items = list(datasets)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n_series, len(items)), replace=False)
    return [items[i] for i in idx]


def smoke_test_family(fam, datasets, n_series=3, verbose=True):
    """Cheap sanity checks before a candidate costs a full extraction.

    Catches the four ways a new family wastes an hour: duplicate names inside itself,
    a name that collides with an existing feature, a name/width disagreement, and a
    column that is constant (which scores exactly 0.500000 and looks like 'no signal')."""
    names = list(fam.names())
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise AssertionError("%s.names() has duplicates: %s" % (fam.name, dupes))

    existing_families = {f.name for f in REGISTRY}
    if fam.name in existing_families:
        raise AssertionError(
            "family name %r is already in REGISTRY -- pick another .name, or drop the "
            "old one first with research_step(..., drop={%r})" % (fam.name, fam.name))
    clash = sorted(set(names) & set(feature_to_family()))
    if clash:
        raise AssertionError(
            "%s emits feature name(s) that already exist: %s" % (fam.name, clash))

    reg = list(REGISTRY) + [fam]
    rows = []
    for did, xh, xo, tau in list(datasets)[:n_series]:
        fx = StreamFeatures(xh, active_families={fam.name}, registry=reg)
        if fx.names != names:
            raise AssertionError(
                "%s: driver emits %d columns, names() promises %d"
                % (fam.name, len(fx.names), len(names)))
        rows.append(np.asarray([fx.update(v) for v in xo], dtype=np.float64))
    R = np.concatenate(rows)

    if R.shape[1] != len(names):
        raise AssertionError("%s.update() returned %d values, names() has %d"
                             % (fam.name, R.shape[1], len(names)))
    bad = ~np.isfinite(R)
    if bad.any():
        cols = [names[c] for c in np.unique(np.nonzero(bad)[1])]
        raise AssertionError("%s produced non-finite values in: %s" % (fam.name, cols))
    const = [names[c] for c in range(R.shape[1]) if R[:, c].std() == 0]
    if const:
        raise AssertionError(
            "%s produced CONSTANT column(s) over %d smoke rows: %s -- a constant column "
            "scores exactly 0.5 and is indistinguishable from 'no signal' in the "
            "diagnostic, so fix it before spending an extraction on it"
            % (fam.name, R.shape[0], const))

    if verbose:
        print("  smoke test OK: %d column(s), %d rows over %d series, all finite, none constant"
              % (R.shape[1], R.shape[0], min(n_series, len(list(datasets)))))
    return R


def _fixed_folds(g, n_folds, seed):
    """Series-level folds that depend only on (series set, n_folds, seed) -- so the
    baseline and the candidate are always scored on identical splits."""
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    return [tuple(sorted(uniq[ix].tolist())) for ix in np.array_split(perm, n_folds)]


def _cv_fold_aucs(X, y, t, w, g, names, idx, folds, params, num_rounds, early_stop):
    Xsel = X[:, idx]
    sel = [names[i] for i in idx]
    out = []
    for va_ids in folds:
        va = np.isin(g, np.asarray(va_ids))
        tr = ~va
        dtr = lgb.Dataset(Xsel[tr], label=y[tr], weight=w[tr],
                          feature_name=sel, free_raw_data=False)
        dva = lgb.Dataset(Xsel[va], label=y[va], feature_name=sel,
                          reference=dtr, free_raw_data=False)
        booster = lgb.train(
            params, dtr, num_boost_round=num_rounds,
            valid_sets=[dva], valid_names=["valid"],
            callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)],
        )
        pva = booster.predict(Xsel[va], num_iteration=booster.best_iteration)
        out.append(ts_auc(y[va], t[va], pva))
    return np.asarray(out, dtype=np.float64)


def _cached_null(skey, X, y, t, g, names, fam_names, n_perm, seed, ranker, cache_dir,
                  verbose=True):
    """Permutation-null cutoff over one representative feature per BASELINE family.

    Keyed on the baseline family set (not the candidate's), so adding a family does not
    invalidate the cutoff and every later research_step is a dict lookup."""
    key = "null::" + _short_hash({"families": sorted(fam_names), "n_perm": n_perm,
                                   "seed": seed, "n_rows": int(len(y))})
    hit = load_null(skey, key, cache_dir=cache_dir)
    if hit is not None:
        if verbose:
            print("  null: cached (cutoff %.5f, %d perms)" % (hit["cutoff"], hit["n_perm"]))
        return hit["cutoff"], hit["null_df"]

    f2f = feature_to_family()
    base_cols = [i for i, nm in enumerate(names) if f2f.get(nm) in fam_names]
    tab = _folded_ts_auc_table(X[:, base_cols], y, t, [names[i] for i in base_cols],
                                ranker=ranker)
    tab["family"] = tab["feature"].map(f2f)
    rep = tab.sort_values("ts_auc_abs", ascending=False).drop_duplicates(subset="family")
    perm_features = rep["feature"].tolist()
    perm_idx = [names.index(nm) for nm in perm_features]
    if verbose:
        print("  null: computing over %d representative features x %d perms ..."
              % (len(perm_features), n_perm))
    null_df = permutation_null_ts_auc(X, y, t, g, perm_idx, perm_features,
                                       n_perm=n_perm, seed=seed, ranker=ranker)
    cutoff = float(null_df["null_p95"].max())
    save_null(skey, key, {"cutoff": cutoff, "null_df": null_df, "n_perm": n_perm},
              cache_dir=cache_dir)
    return cutoff, null_df


def research_step(new_family=None, drop=None, n_series=None, seed=0,
                   n_folds=3, fast_lgb=True, register=False,
                   datasets=None, n_perm=100, cache_dir=CACHE_DIR, verbose=True):
    """Score one idea against the current registry and print a verdict.

    new_family : a FeatureFamily INSTANCE to evaluate, or None to only measure `drop`.
    drop       : family names to remove from the baseline.
    register   : keep `new_family` in REGISTRY afterwards (default: restore the registry).

    Returns the univariate table for the candidate's columns; the CV numbers, the
    verdict and the stage timings are also on `df.attrs`.
    """
    datasets = train_data if datasets is None else datasets
    n_series = DIAG_N_SERIES if n_series is None else n_series
    drop = set(drop or ())
    sample = _sample_for(datasets, n_series, seed)

    registry_backup = list(REGISTRY)
    timings = {}
    try:
        # ---------------------------------------------------- 0. smoke test
        if new_family is not None:
            if verbose:
                print("== smoke test: %s (%d columns) ==" % (new_family.name,
                                                             len(new_family.names())))
            t0 = time.perf_counter()
            smoke_test_family(new_family, sample, n_series=3, verbose=verbose)
            timings["smoke"] = time.perf_counter() - t0
            REGISTRY.append(new_family)

        base_families = {f.name for f in REGISTRY} - drop
        if new_family is not None:
            base_families -= {new_family.name}
            cand_families = base_families | {new_family.name}
            label = "+%s" % new_family.name
        else:
            cand_families = base_families | drop
            label = "+%s" % ",".join(sorted(drop)) if drop else "(no change)"
        assert cand_families != base_families or not drop, \
            "nothing to compare: pass new_family= and/or drop="

        # ---------------------------------------------------- 1. extraction
        if verbose:
            print("\n== extraction (%d series, seed %d) ==" % (len(sample), seed))
        t0 = time.perf_counter()
        X, y, t, w, g, names = get_cached_matrix(sample, subsample=True, seed=seed,
                                                  cache_dir=cache_dir, verbose=verbose)
        timings["extract"] = time.perf_counter() - t0
        skey = sample_key([r[0] for r in sample], True, seed)
        f2f = feature_to_family()

        # ---------------------------------------------------- 2. univariate
        if verbose:
            print("\n== univariate ==")
        t0 = time.perf_counter()
        ranker = TSRanker(t)
        cutoff, _null_df = _cached_null(skey, X, y, t, g, names, base_families,
                                         n_perm, seed, ranker, cache_dir, verbose)
        if new_family is not None:
            cols = [i for i, nm in enumerate(names) if f2f.get(nm) == new_family.name]
            uni = _folded_ts_auc_table(X[:, cols], y, t, [names[i] for i in cols],
                                        ranker=ranker)
        else:
            uni = _folded_ts_auc_table(X[:, :0], y, t, [], ranker=ranker)
        uni["cutoff"] = cutoff
        uni["above_null"] = uni["ts_auc_abs"] >= cutoff
        uni = uni.sort_values("ts_auc_abs", ascending=False).reset_index(drop=True)
        timings["univariate"] = time.perf_counter() - t0
        if verbose and len(uni):
            print(uni.to_string(index=False))
            print("  %d/%d of the candidate's columns clear the null cutoff %.5f"
                  % (int(uni["above_null"].sum()), len(uni), cutoff))

        # ---------------------------------------------------- 3. CV
        if verbose:
            print("\n== %d-fold CV (folds fixed by seed=%d, %s) =="
                  % (n_folds, seed, "FAST_LGB" if fast_lgb else "FINAL_LGB"))
        t0 = time.perf_counter()
        params, n_rounds, early = _lgb_config(fast_lgb)
        folds = _fixed_folds(g, n_folds, seed)

        base_idx = np.array([i for i, nm in enumerate(names)
                             if f2f.get(nm) in base_families], dtype=np.int64)
        cand_idx = np.array([i for i, nm in enumerate(names)
                             if f2f.get(nm) in cand_families], dtype=np.int64)

        fit_key = {"params": {k: params[k] for k in sorted(params)},
                   "num_rounds": n_rounds, "early": early,
                   "folds": [list(f) for f in folds]}
        base_key = "cv::" + _short_hash(dict(fit_key, features=[names[i] for i in base_idx]))
        base_fold = load_null(skey, base_key, cache_dir=cache_dir)
        if base_fold is None:
            base_fold = _cv_fold_aucs(X, y, t, w, g, names, base_idx, folds,
                                       params, n_rounds, early)
            save_null(skey, base_key, base_fold, cache_dir=cache_dir)
            base_src = "trained"
        else:
            base_src = "cached"
        cand_key = "cv::" + _short_hash(dict(fit_key, features=[names[i] for i in cand_idx]))
        cand_fold = load_null(skey, cand_key, cache_dir=cache_dir)
        if cand_fold is None:
            cand_fold = _cv_fold_aucs(X, y, t, w, g, names, cand_idx, folds,
                                       params, n_rounds, early)
            save_null(skey, cand_key, cand_fold, cache_dir=cache_dir)
            cand_src = "trained"
        else:
            cand_src = "cached"
        timings["cv"] = time.perf_counter() - t0

        deltas = cand_fold - base_fold
        mean_delta = float(deltas.mean())
        paired_std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        keep = mean_delta > paired_std

        if verbose:
            print("  baseline  (%2d families, %3d features, %s): %s | mean %.5f"
                  % (len(base_families), len(base_idx), base_src,
                     " ".join("%.5f" % a for a in base_fold), base_fold.mean()))
            print("  candidate (%2d families, %3d features, %s): %s | mean %.5f"
                  % (len(cand_families), len(cand_idx), cand_src,
                     " ".join("%.5f" % a for a in cand_fold), cand_fold.mean()))
            print("  per-fold delta: %s" % " ".join("%+.5f" % d for d in deltas))
            print("  mean delta %+.5f | paired std %.5f (ddof=1)" % (mean_delta, paired_std))
            print()
            print("  %-9s %s  mean delta %+.5f vs paired std %.5f   "
                  "[smoke %.1fs | extract %.1fs | univariate %.1fs | cv %.1fs]"
                  % ("KEEP" if keep else "NO GAIN", label, mean_delta, paired_std,
                     timings.get("smoke", 0.0), timings["extract"],
                     timings["univariate"], timings["cv"]))
            if register and new_family is not None:
                print("  registered %r -- REGISTRY is now %d families"
                      % (new_family.name, len(REGISTRY)))
            elif new_family is not None:
                print("  registry restored (pass register=True to keep %r)" % new_family.name)

        uni.attrs.update(
            verdict="KEEP" if keep else "NO GAIN",
            label=label,
            baseline_folds=base_fold, candidate_folds=cand_fold, deltas=deltas,
            mean_delta=mean_delta, paired_std=paired_std, cutoff=cutoff,
            timings=timings, folds=folds, sample_key=skey,
            baseline_families=sorted(base_families),
            candidate_families=sorted(cand_families),
        )
        return uni
    finally:
        if not (register and new_family is not None):
            REGISTRY[:] = registry_backup


def local_test(**kwargs):
    """The real end-to-end crunch run: train() + infer() + determinism check on the
    actual competition data. Minutes to hours -- this is the occasional final smoke
    test, not part of the iteration loop. Everything above exists so that this is
    rarely the thing you are waiting on."""
    return crunch_tools.test(**kwargs)


print("research_step / local_test defined -- see the markdown above for the 4-line workflow.")
'''


def main():
    nb = nbedit.load()

    # split the diagnostic constants out of the run cell
    i = find_cell(nb, "DIAG_SEED = 0", "code")
    set_src(nb["cells"][i], DIAG_CONST_CELL)
    nb["cells"].insert(i + 1, nbedit.new_code_cell(DIAG_RUN_CELL))

    # section 3d goes after 3c's example cell, before section 4
    i = find_cell(nb, "quick_train / ablate defined", "code")
    nb["cells"].insert(i + 1, new_markdown_cell(RESEARCH_MD))
    nb["cells"].insert(i + 2, new_code_cell(RESEARCH_CELL))

    nbedit.save(nb)
    print("phase 5 applied")


if __name__ == "__main__":
    main()
