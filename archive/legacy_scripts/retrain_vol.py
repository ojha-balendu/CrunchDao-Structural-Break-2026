"""Retrain the submission model after the volatility families were added.

  python tools/retrain_vol.py

Converts submission.ipynb, imports it, then runs the two production calls in order:
    train(train_data, "resources")        -> early-stopped, writes holdout_scores.npz
    train_full(train_data, "resources")   -> refit on all series (keeps model.joblib.holdout_backup)
and asserts the artifact has 3 boosters and 109 features.
"""
import contextlib, importlib, io, os, subprocess, sys, tempfile, time

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.chdir(ROOT)
    d = tempfile.mkdtemp(prefix="retrain_")
    out = os.path.join(d, "_vol_main.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", "submission.ipynb",
                        out, "--override"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    sys.path.insert(0, d)                      # importable BY NAME so parallel workers can too
    with contextlib.redirect_stdout(io.StringIO()):
        m = importlib.import_module("_vol_main")
    print("registry: %d families | %d features | seeds %s"
          % (len(m.REGISTRY), m.N_FEATURES, m.SEEDS), flush=True)

    import crunch
    crunch_tools = crunch.load_notebook()
    train_data, _ = crunch_tools.load_data()

    t0 = time.time()
    print("\n=== train() ===", flush=True)
    m.train(train_data, "resources")
    t1 = time.time()
    print("train() took %.1f min" % ((t1 - t0) / 60), flush=True)

    print("\n=== train_full() ===", flush=True)
    m.train_full(train_data, "resources")
    print("train_full() took %.1f min" % ((time.time() - t1) / 60), flush=True)

    art = joblib.load("resources/model.joblib")
    print("\nartifact keys:", sorted(art))
    print("boosters: %d | features: %d | best_iterations: %s | active_families: %s"
          % (len(art["boosters"]), len(art["features"]), art["best_iterations"], art["active_families"]))
    assert len(art["boosters"]) == 3, len(art["boosters"])
    assert len(art["features"]) == 109, len(art["features"])
    for fam in ("garch", "argarch", "tgarch", "gjr"):
        n = sum(1 for f in art["features"] if f.startswith(fam + "_"))
        print("  %-8s %d features in artifact" % (fam, n))
    assert os.path.exists("resources/model.joblib.holdout_backup"), "holdout backup missing"
    print("\nARTIFACT OK | total %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":     # required: spawn/loky workers re-import this file
    main()
