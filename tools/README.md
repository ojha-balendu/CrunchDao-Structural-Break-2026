# Tools

Checks that the submission works the way CrunchDAO will actually run it.

| Script | What it checks |
|---|---|
| `check_submission.py` | Converts the notebook with CrunchDAO's converter, imports the result in a fresh process, trains on synthetic series, streams `infer()` point by point, and checks determinism (CrunchDAO re-runs 10% of inference with a 1e-8 tolerance) |
| `parity_blend.py` | Train/serve parity: batch feature extraction vs. streaming `infer()` on held-out series, max abs diff < 1e-6 |
| `parity_vol.py` | The notebook's volatility families reproduce the cached research extractions column by column |
| `profile_infer.py` | Seconds per series for `infer()`, projected to the full 10,000-series test set |
| `retrain_blend.py` | Retrains the production model in a fresh process (`train()` then `train_full()`) and verifies the artifact |

Run from the repository root, for example `python tools/check_submission.py submission.ipynb`.
