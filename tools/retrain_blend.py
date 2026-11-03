"""Retrain the LightGBM + CatBoost blend.

  python tools/retrain_blend.py

train(train_data, "resources") then train_full(train_data, "resources"), in a fresh process that
imports the converted notebook; then verifies the artifact.
"""
import contextlib, importlib, io, os, subprocess, sys, tempfile, time

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.chdir(ROOT)
    d = tempfile.mkdtemp(prefix="retrain_blend_")
    out = os.path.join(d, "_blend_main.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", "submission.ipynb",
                        out, "--override"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    sys.path.insert(0, d)                          # importable by name, so loky workers can too
    with contextlib.redirect_stdout(io.StringIO()):
        m = importlib.import_module("_blend_main")
    print("registry %d families | %d features | SEEDS %s | CAT_WEIGHT %s"
          % (len(m.REGISTRY), m.N_FEATURES, m.SEEDS, m.CAT_WEIGHT), flush=True)
    print("LGB_PARAMS:", {k: m.LGB_PARAMS[k] for k in ("learning_rate", "num_leaves", "min_data_in_leaf",
          "feature_fraction", "bagging_fraction", "lambda_l2", "extra_trees", "min_gain_to_split")})
    print("CAT_PARAMS:", m.CAT_PARAMS, flush=True)

    import crunch
    crunch_tools = crunch.load_notebook()
    with contextlib.redirect_stdout(io.StringIO()):
        train_data, _ = crunch_tools.load_data()

    t0 = time.time()
    print("\n=== train() ===", flush=True)
    m.train(train_data, "resources")
    t1 = time.time()
    print("train() took %.1f min" % ((t1 - t0) / 60), flush=True)

    print("\n=== train_full() ===", flush=True)
    m.train_full(train_data, "resources")
    print("train_full() took %.1f min" % ((time.time() - t1) / 60), flush=True)

    from catboost import CatBoostClassifier
    for p in ("resources/model.joblib.holdout_backup", "resources/model.joblib"):
        art = joblib.load(p)
        print("\n%s" % p)
        print("  keys: %s" % sorted(art))
        print("  lgb boosters %d | best_iterations %s | features %d | catboost %s | cat_weight %s | cat_iters %s"
              % (len(art["boosters"]), art["best_iterations"], len(art["features"]),
                 type(art.get("catboost")).__name__, art.get("cat_weight"), art.get("cat_iters")))
        assert len(art["boosters"]) == 3
        assert len(art["features"]) == m.N_FEATURES == len(m.FEATURES), (len(art["features"]), m.N_FEATURES)
        assert list(art["features"]) == list(m.FEATURES), "artifact feature order != notebook FEATURES"
        assert isinstance(art["catboost"], CatBoostClassifier)
        assert art["cat_weight"] == 0.3
        assert art["cat_iters"] and art["cat_iters"] > 0
        print("  catboost trees actually in model: %d" % art["catboost"].tree_count_)
    print("\nARTIFACT OK | total %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
