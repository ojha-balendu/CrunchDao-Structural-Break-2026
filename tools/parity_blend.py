"""Train-vs-infer parity for the LightGBM + CatBoost blend.

  python tools/parity_blend.py [--model resources/model.joblib] [--series 20]

On N series from train()'s hold-out split (same rng(0) / VALID_FRACTION draw as train()):
  batch : extract_series(subsample=False) -> mean(lgb seeds) and catboost -> (1-w)*lgb + w*cat,
          then the same stride hold infer() applies (score refreshed only when t % stride == 0)
  stream: infer() fed point by point
Max abs diff must be < 1e-6.
"""
import argparse, contextlib, importlib, io, os, shutil, subprocess, sys, tempfile

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-6


def load_module():
    d = tempfile.mkdtemp(prefix="parity_blend_")
    out = os.path.join(d, "_pb_main.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", "submission.ipynb",
                        out, "--override"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    sys.path.insert(0, d)
    with contextlib.redirect_stdout(io.StringIO()):
        return importlib.import_module("_pb_main")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resources/model.joblib")
    ap.add_argument("--series", type=int, default=20)
    a = ap.parse_args()
    os.chdir(ROOT)

    m = load_module()
    assert m.SMOOTH_ALPHA == 0.0 and not m.RUNNING_MAX, "post-processing is on; the batch side assumes it is off"

    import crunch
    crunch_tools = crunch.load_notebook()
    with contextlib.redirect_stdout(io.StringIO()):
        train_data, _ = crunch_tools.load_data()

    # train()'s split, reproduced exactly
    items = list(train_data)
    if m.TRAIN_MAX_SERIES:
        items = items[:m.TRAIN_MAX_SERIES]
    is_val = np.random.default_rng(0).random(len(items)) < m.VALID_FRACTION
    va = [r for r, v in zip(items, is_val) if v]
    pick = [va[i] for i in np.linspace(0, len(va) - 1, a.series).astype(int)]
    print("hold-out split: %d series | testing %d spread across it" % (len(va), len(pick)))

    art = joblib.load(a.model)
    boosters, its = art["boosters"], art["best_iterations"]
    cat, w = art.get("catboost"), float(art.get("cat_weight", 0.0))
    active = set(art["active_families"]) if art.get("active_families") else None
    print("artifact %s: %d lgb boosters %s | catboost %s | cat_weight %.2f"
          % (a.model, len(boosters), its, "yes" if cat is not None else "NO", w))

    # ---- batch side
    batch = []
    for sid, xh, xo, tau in pick:
        xo = np.asarray(list(xo), dtype=np.float64)
        X, _, t, _ = m.extract_series(xh, xo, tau, subsample=False, active_families=active)
        pl = np.mean([b.predict(X, num_iteration=it) for b, it in zip(boosters, its)], axis=0)
        p = pl if cat is None else (1.0 - w) * pl + w * cat.predict_proba(X)[:, 1]
        held, cur = np.empty_like(p), 0.0
        for k in range(p.size):
            if t[k] % m._stride_for(int(t[k])) == 0:
                cur = p[k]
            held[k] = cur
        batch.append(np.clip(held, 0.0, 1.0))

    # ---- streamed side
    md = tempfile.mkdtemp(prefix="pb_model_")
    shutil.copy(a.model, os.path.join(md, "model.joblib"))
    gen = m.infer([(xh, iter(np.asarray(list(xo), dtype=np.float64))) for _, xh, xo, _ in pick], md)
    next(gen)
    streamed = np.array([float(v) for v in gen])

    ref = np.concatenate(batch)
    assert streamed.size == ref.size, "streamed %d values, batch %d" % (streamed.size, ref.size)
    diffs, off = [], 0
    for (sid, *_), b in zip(pick, batch):
        s = streamed[off:off + b.size]; off += b.size
        diffs.append((float(np.max(np.abs(s - b))), int(sid), b.size))
    worst = max(diffs)
    print("rows compared: %d over %d series" % (ref.size, len(pick)))
    print("per-series max abs diff: min %.2e | median %.2e | max %.2e (series %d, %d rows)"
          % (min(d[0] for d in diffs), float(np.median([d[0] for d in diffs])), worst[0], worst[1], worst[2]))
    ok = worst[0] < TOL
    print("PARITY %s (tolerance %.0e)" % ("OK" if ok else "FAILED", TOL))
    if not ok:
        s0 = 0
        for (sid, *_), b in zip(pick, batch):
            s = streamed[s0:s0 + b.size]; s0 += b.size
            if int(sid) == worst[1]:
                k = int(np.argmax(np.abs(s - b)))
                print("first divergence in series %d at row %d: batch %.9f stream %.9f" % (sid, k, b[k], s[k]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
