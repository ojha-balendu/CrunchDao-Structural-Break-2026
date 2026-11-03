# Feature research

Candidate feature families, written and tested **outside** the submission notebook. Each one
was scored with the paired cross-validation harness (`research.ipynb` and the scripts in
`experiments/`) before anything was copied into `submission.ipynb`. Nothing in this folder is
imported by `train()` or `infer()`: CrunchDAO submits the notebook alone, so a family only
ships once its code is pasted into the notebook's registry.

| File | Families | Outcome |
|---|---|---|
| `volatility_families.py` | `garch`, `argarch`, `tgarch`, `gjr` | `garch`, `argarch`, `tgarch` shipped; `gjr` dropped in a later prune |
| `lit3_families.py` | `ordpat`, `wavvar`, `recres`, `hmm`, `evt` | `recres` and `hmm` shipped; `wavvar` shipped, then pruned |
| `lit4_families.py` | `ssa`, `mchain`, `xrec` | tested with `experiments/test_families.py` |
| `lit1_ctm_newma_bocpd2.py` | `ctm` (conformal test martingales), `newma`, `bocpd2` | early literature round |
| `archived_families.py` | the families removed in the Phase 1 cull (location, dependence, trend and shape tests) | parts bin, with restore instructions at the top |
| `fast_families.py`, `fast_metrics.py` | optimized versions of the surviving families and of the TS-AUC metric | folded into the notebook; output-identical to about 1e-12 |
| `reference/` | research versions of `dbocpd`, the GRU null model and the RankNet early-step ranker | ported into the notebook (section 1k and section 4) |

Most of these files are written to be pasted or `exec`'d into the converted notebook's
namespace, so they rely on names it defines (`FeatureFamily`, `HistoricalRef`, `_CondBase`,
and so on) rather than importing them.
