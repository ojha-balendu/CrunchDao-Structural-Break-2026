# Experiments

Standalone scripts for the slower questions: hyperparameter search, model blending, feature
pruning and family screening. Every script converts `submission.ipynb` the same way CrunchDAO
does and works off the feature cache in `cache/`, so a notebook edit or a kernel restart can't
change what an experiment is measuring.

**Run them from the repository root**, for example `python experiments/blend_test.py`. They read
`cache/` and `submission.ipynb` by relative path. `cache/` is built by the notebook's feature
extraction and is not committed.

| Script | Question it answers |
|---|---|
| `tune_overnight.py` | Optuna search for the LightGBM and CatBoost parameters, stored in SQLite so a study can resume |
| `blend_test.py` | Does LightGBM + CatBoost beat LightGBM alone on a fresh split? |
| `blend_experiment.py` | LightGBM + XGBoost + CatBoost blends, by rank average and by probability average |
| `test_families.py`, `test_families_lit3.py` | Screen new literature families: LightGBM screen first, then confirmation on the full model |
| `prune_features.py` | Within-time-step correlation clusters, and CV of pruned feature sets against the full set |
| `balance_features.py` | Does giving every family the same number of features help the trees? |
| `break_age_weighting.py` | Does down-weighting the first rows after a break help training? |
| `bucket3_pruning.py` | Keep-one-in ablation over the redundant scale tests (`ftest`, `bartlett`, `levene`, `sigtuk`, `lscale`) |

All of them use the same decision rule: paired TS-AUC differences over 6 fold-halves (3 folds,
each split into an early-stopping half and a scoring half), and a change is adopted only if the
mean difference is more than 2 standard errors **and** it wins on at least 5 of the 6 halves.

`RUNBOOK.md` is the run order and output guide for one round of these experiments.
