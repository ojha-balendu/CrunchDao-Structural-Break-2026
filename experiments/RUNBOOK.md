# Runbook: Experiments A and B (151-feature baseline)

Run these from the **repository root** with the venv active, in this order.
**Run A and B one after the other, never at the same time.** Each one uses all 8 cores and several GB
of the 16 GB of RAM. Running both at once would slow them to a crawl or run out of memory.

`-u` keeps Python from buffering its output, so the `tee` log fills as the run goes. Both experiments
are safe to stop with Ctrl-C. Re-running B resumes where it stopped. A always starts over.

## 1. Pre-flight check (about 5 s)

```bash
caffeinate -i python -u experiments/tune_overnight.py --check 2>&1 | tee cache/check151.log
```

- **What it does:** loads the caches and checks each of the 7 extra feature blocks against the converted
  `submission.ipynb` registry (same names, same order).
- **Writes:** only the log.
- **Verdict line:** the last line must be `TOTAL 151 OK`. If there's a mismatch, it stops with an
  `AssertionError`. Don't start A or B until this passes.

## 2. Experiment A: LightGBM alone vs LightGBM + CatBoost on 151 features (about 35–45 min)

```bash
caffeinate -i python -u experiments/blend_test.py --fold-seed 19 2>&1 | tee cache/blend_test151.log
```

- **What it does:**
  - Trains on 6 paired fold-halves using fold seed 19, which no earlier study has used.
  - Compares LightGBM with the shipped parameters, LightGBM with the Optuna-best parameters, CatBoost
    alone, and both LightGBMs blended with CatBoost at weights 0.2–0.6.
  - Each model is judged against **`lgb shipped`**, which is LightGBM alone with no CatBoost. The 109-feature
    run of this script took 24.7 min.
- **Writes:** `cache/blend_choice_19.json`. It never touches the shipped `cache/blend_choice.json`.
- **Results table:** under `RESULTS vs the SHIPPED model`. The row that answers the question is
  `lgb shipped + cat 0.3`: its delta, sigma and "better" count against LightGBM alone.
- **Verdict line:** `VERDICT: SHIP '<choice>'`.
  - `lgb shipped`: CatBoost doesn't clear the bar (more than 2 sigma, and better on at least 5 of the
    6 halves), so drop it.
  - `lgb shipped + cat 0.3`: keep the current blend.

## 3. Experiment B: `ssa`, `mchain` and `xrec` on top of the 151-feature blend (about 1–3 h)

```bash
caffeinate -i python -u experiments/test_families.py 2>&1 | tee cache/family_tests_lit4.log
```

- **What it does, in stages:**
  - **Stage 0:** all checks.
  - **Stage 1:** extracts the three families over all 10,000 series. That's about 2 min in total, based
    on the 200-series dry run. It also prints a table of how well each new column predicts a break on
    its own.
  - **Stage 2a, screen:** trains LightGBM on 151 features vs 151 + each family, on 6 fold-halves with fold
    seed 23. About 45 min.
  - **Stage 2b, confirm:** repeats the comparison with the full blend, for each family that beat the
    baseline by more than 1 sigma in the screen. About 10–15 min for the baseline, plus about the same
    per passing family.
  - **Stage 2c, combined:** tests the passing families together, only if two or more pass.
  - The run takes about 1 h if nothing passes the screen, and up to about 3 h if everything passes.
- **Writes:**
  - The extracted features go to `cache/new_ssa.npz`, `cache/new_mchain.npz` and `cache/new_xrec.npz`.
  - Results go to `cache/family_tests_lit4.json`, which is what makes the run resumable.
  - Predictions from each training run go to `cache/family_tests_lit4/`.
- **What to read in the output:**
  - The `SCREEN (LightGBM x3)` table shows whether each family goes on to the blend confirm.
  - The `CONSISTENCY` line compares the blend with LightGBM alone. It will probably show the
    `!! THE BLEND DOES NOT BEAT LIGHTGBM ALONE` warning. That was expected at 151 features, and
    Experiment A settles it.
  - The `CONFIRM (blend)` table ends each row with `ADOPT` or `REJECT`.
- **Verdict line:** the last `VERDICT:` line, either `adopt +<family>, ...` or `adopt nothing`.

## Test commands (already run, for reference)

```bash
python experiments/test_families.py --check      # stage 0 only (under 1 min)
python experiments/test_families.py --dry-run    # every stage on 200 series, in cache/dryrun/ (about 2 min, wiped each run)
```
