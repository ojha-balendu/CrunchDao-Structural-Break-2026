"""Profile infer() cost per series, for the current model and (optionally) the pre-change one.

  python tools/profile_infer.py [--series 100] [--compare resources/model_pre_vol.joblib]

Reports seconds/series and the projected wall clock for 10,000 series at INFER_PARALLELISM.
The pre-change model is profiled with the SAME converted notebook, so it exercises only the
families its own artifact asks for (infer() rebuilds the extractor from art["active_families"]).
"""
import argparse, contextlib, importlib, io, json, os, shutil, subprocess, sys, tempfile, time

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(nb="submission.ipynb"):
    d = tempfile.mkdtemp(prefix="prof_")
    out = os.path.join(d, "_prof_main.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", nb,
                        out, "--override"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    sys.path.insert(0, d)
    with contextlib.redirect_stdout(io.StringIO()):
        return importlib.import_module("_prof_main")


def profile(m, model_dir, items, label):
    t0 = time.time()
    gen = m.infer([(d[1], iter(d[2])) for d in items], model_dir)
    next(gen)                                   # ready signal
    n = sum(1 for _ in gen)
    dt = time.time() - t0
    per = dt / len(items)
    par = getattr(m, "INFER_PARALLELISM", 1) or 1
    proj = per * 10000 / par / 60
    print("%-12s %6.3f s/series | %7d predictions | %5.1f s total | projected 10k at parallelism %d: %.1f min"
          % (label, per, n, dt, par, proj), flush=True)
    return per, proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=100)
    ap.add_argument("--notebook", default="submission.ipynb")
    ap.add_argument("--model", default="resources/model.joblib")
    ap.add_argument("--label", default="with GARCH")
    ap.add_argument("--compare", default="")
    a = ap.parse_args()
    os.chdir(ROOT)

    m = load_module(a.notebook)
    import crunch
    crunch_tools = crunch.load_notebook()
    with contextlib.redirect_stdout(io.StringIO()):
        train_data, _ = crunch_tools.load_data()
    items = list(train_data)[:a.series]
    print("profiling %d series | INFER_PARALLELISM = %s | registry %d families, %d features"
          % (len(items), getattr(m, "INFER_PARALLELISM", None), len(m.REGISTRY), m.N_FEATURES))

    art = joblib.load(a.model)
    print("artifact %s: %d boosters, %d features, %d trees/booster"
          % (a.model, len(art.get("boosters", [1])), len(art["features"]),
             int(np.mean(art.get("best_iterations", [0])))))
    assert len(art["features"]) == m.N_FEATURES, (
        "artifact has %d features but this notebook's registry emits %d -- profile the model "
        "with the notebook it was trained from" % (len(art["features"]), m.N_FEATURES))
    d = tempfile.mkdtemp(prefix="profmodel_")
    shutil.copy(a.model, os.path.join(d, "model.joblib"))
    now_per, now_proj = profile(m, d, items, a.label)

    if a.compare:
        print("to compare, run this script again with the pre-change pair, e.g.:\n"
              "  python tools/profile_infer.py --notebook submission.BACKUP_pre_vol.ipynb "
              "--model resources/model_pre_vol.joblib --label pre-change")


if __name__ == "__main__":
    main()
