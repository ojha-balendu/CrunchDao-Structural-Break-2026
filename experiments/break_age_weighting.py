"""Break-age sample weighting: does downweighting the first steps after a break help?

    python experiments/break_age_weighting.py                  # full comparison (baseline + 4 ramps)
    python experiments/break_age_weighting.py --k 10 --floor 0.25
    python experiments/break_age_weighting.py --force          # ignore cached fold results

Rows right after a break are labelled positive but carry almost no post-break evidence.
This multiplies the training weight of rows with 0 <= age < k by
    floor + (1 - floor) * age / k          (age = t - tau, tau = first t with y == 1)
and compares against the current weighting with the paired 3-fold x 2-half CV that
research.ipynb uses (same folds, same row order, same LightGBM call), so the baseline here
reproduces research.ipynb's `cv_cols("base (85)", ...)` numbers exactly.

Normalization: the ramped weights are rescaled so the TOTAL training weight equals the
current total (not mean 1). The current stride weights average ~3.3, and with lambda_l2 ~44
rescaling every weight changes the effective regularization, which would confound the test
and break "k = 0 reproduces current behaviour exactly". Preserving the total moves weight
between rows and changes nothing else.

Reads cache/ (built by research.ipynb); never modifies either notebook.
"""
import argparse
import ast
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
CACHE = os.path.join(ROOT, "cache")
RESULTS = os.path.join(CACHE, "age_weight_results.json")
NOTEBOOK = os.path.join(ROOT, "submission.ipynb")

N_FOLDS = 3
PATIENCE = 100
DEFAULT_CONFIGS = [(5, 0.25), (10, 0.25), (20, 0.25), (10, 0.5)]
KEEP_MIN_BETTER = 5


# ----------------------------------------------------------------------------- inputs
def load_cache():
    need = ["X_all.npy", "y_all.npy", "t_all.npy", "g_all.npy", "names.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(CACHE, f))]
    if missing:
        sys.exit("missing %s in %s -- run the cache cell (section 1) of research.ipynb first"
                 % (", ".join(missing), CACHE))
    X = np.load(os.path.join(CACHE, "X_all.npy"), mmap_mode="r")
    y = np.load(os.path.join(CACHE, "y_all.npy"))
    t = np.load(os.path.join(CACHE, "t_all.npy"))
    g = np.load(os.path.join(CACHE, "g_all.npy"))
    names = json.load(open(os.path.join(CACHE, "names.json")))
    assert X.shape[0] == len(y) == len(t) == len(g), "cache arrays disagree on row count"
    assert X.shape[1] == len(names), "X_all has %d columns, names.json %d" % (X.shape[1], len(names))
    return X, y, t, g, names


def import_engine():
    """Convert submission.ipynb exactly as Crunch does and import the result.

    Guards against the import training anything: every module-level call in the converted
    source is inspected, and a call to train/infer/test/build_dataset aborts the run."""
    out = os.path.join(tempfile.mkdtemp(prefix="age_weight_"), "_age_engine.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", NOTEBOOK, out, "--override"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("crunch_convert failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])

    forbidden = {"train", "infer", "test", "build_dataset", "local_test", "research_step"}
    for node in ast.parse(open(out).read()).body:
        call = getattr(node, "value", None)
        if isinstance(node, (ast.Expr, ast.Assign)) and isinstance(call, ast.Call):
            f = call.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in forbidden:
                sys.exit("converted notebook calls %s() at import time (line %d) -- refusing to import"
                         % (name, node.lineno))

    spec = importlib.util.spec_from_file_location("_age_engine", out)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):          # the module prints a banner
        spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- folds
def stride_arr(t, keep_every):
    s = np.full(t.shape, keep_every[-1][1], dtype=np.int32)
    for up_to, stride in reversed(keep_every):
        s[t < up_to] = stride
    return s


def build_folds(t, g, keep_every, seed=0):
    """Identical to research.ipynb build_folds(): same RNG streams, same row order."""
    def sorted_by_t(mask):
        idx = np.flatnonzero(mask)
        return idx[np.argsort(t[idx], kind="stable")]

    uniq = np.unique(g)
    pos = np.searchsorted(uniq, g)
    fold_u = np.empty(uniq.size, dtype=np.int32)
    for k, part in enumerate(np.array_split(np.random.default_rng(seed).permutation(uniq.size), N_FOLDS)):
        fold_u[part] = k
    fold_row = fold_u[pos]
    half_row = (np.random.default_rng(seed + 1).random(uniq.size) < 0.5)[pos]
    sub = (t % stride_arr(t, keep_every)) == 0
    folds = []
    for k in range(N_FOLDS):
        folds.append(dict(tr=sorted_by_t((fold_row != k) & sub),
                          st=sorted_by_t((fold_row == k) & half_row),
                          sc=sorted_by_t((fold_row == k) & ~half_row)))
        print("fold %d: train %s | stop %s | score %s" % (
            k, format(folds[-1]["tr"].size, ","), format(folds[-1]["st"].size, ","),
            format(folds[-1]["sc"].size, ",")))
    return folds


# ----------------------------------------------------------------------------- weights
def break_age(y, t, g):
    """age = t - tau per row, tau = smallest t with y == 1 in that series.
    Rows of series that never break get age -1 (never ramped)."""
    _, inv = np.unique(g, return_inverse=True)
    big = np.iinfo(np.int64).max
    tau = np.full(inv.max() + 1, big, dtype=np.int64)
    pos = y == 1
    np.minimum.at(tau, inv[pos], t[pos].astype(np.int64))
    tau_row = tau[inv]
    return np.where(tau_row == big, -1, t.astype(np.int64) - tau_row)


def age_weights(stride_w, age, k, floor):
    """Stride weight x linear ramp over the first k post-break steps, rescaled so the total
    weight is unchanged. k = 0 returns stride_w itself."""
    if k == 0:
        return stride_w
    ramp = (age >= 0) & (age < k)
    m = np.ones(stride_w.shape, dtype=np.float64)
    m[ramp] = floor + (1.0 - floor) * age[ramp] / k
    w = stride_w.astype(np.float64) * m
    w *= stride_w.astype(np.float64).sum() / w.sum()
    return w.astype(np.float32)


# ----------------------------------------------------------------------------- one fit
def fit_one(E, X, y, t, names, F, w_tr, swap):
    import lightgbm as lgb
    si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
    params = dict(E.LGB_PARAMS)
    params.update(objective="binary", metric="binary_logloss")

    Xtr = np.ascontiguousarray(X[F["tr"]])
    dtr = lgb.Dataset(Xtr, label=y[F["tr"]], weight=w_tr, feature_name=names)
    Xst = np.ascontiguousarray(X[si])
    dva = lgb.Dataset(Xst, label=y[si], reference=dtr)
    bst = lgb.train(params, dtr, num_boost_round=E.NUM_ROUNDS, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    del Xtr, Xst, dtr, dva; gc.collect()

    Xe = np.ascontiguousarray(X[ei])
    auc = float(E.ts_auc(y[ei], t[ei], bst.predict(Xe, num_iteration=bst.best_iteration)))
    it = int(bst.best_iteration)
    del Xe, bst; gc.collect()
    return auc, it


# ----------------------------------------------------------------------------- cache
def fingerprint(E, X):
    st = os.stat(os.path.join(CACHE, "X_all.npy"))
    payload = dict(params={k: E.LGB_PARAMS[k] for k in sorted(E.LGB_PARAMS)},
                   num_rounds=E.NUM_ROUNDS, keep_every=E.KEEP_EVERY, patience=PATIENCE,
                   shape=list(X.shape), x_size=st.st_size, x_mtime=int(st.st_mtime), folds=N_FOLDS)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_results(fp, force):
    if force or not os.path.exists(RESULTS):
        return {}
    data = json.load(open(RESULTS))
    if data.get("fingerprint") != fp:
        print("cached results are for a different setup (params/cache changed) -- recomputing")
        return {}
    return data.get("configs", {})


def save_results(fp, configs):
    tmp = RESULTS + ".tmp"
    json.dump({"fingerprint": fp, "configs": configs}, open(tmp, "w"), indent=1)
    os.replace(tmp, RESULTS)


def label(k, floor):
    return "baseline (k=0)" if k == 0 else "k=%d floor=%.2f" % (k, floor)


# ----------------------------------------------------------------------------- report
def print_edit(k, floor):
    print("""
Edit to train() in submission.ipynb (NOT applied):

    # was:  Xtr, ytr, ttr, wtr, _ = build_dataset(tr_items, subsample=True)
    Xtr, ytr, ttr, wtr, gtr = build_dataset(tr_items, subsample=True)

    # break-age ramp: rows 0 <= t - tau < AGE_K get floor..1, total weight preserved
    AGE_K, AGE_FLOOR = %d, %s
    _, _inv = np.unique(gtr, return_inverse=True)
    _tau = np.full(_inv.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(_tau, _inv[ytr == 1], ttr[ytr == 1].astype(np.int64))
    _age = ttr.astype(np.int64) - _tau[_inv]            # negative for series that never break
    _ramp = (_age >= 0) & (_age < AGE_K)
    _m = np.ones(len(wtr)); _m[_ramp] = AGE_FLOOR + (1.0 - AGE_FLOOR) * _age[_ramp] / AGE_K
    _w = wtr.astype(np.float64) * _m
    wtr = (_w * (wtr.sum(dtype=np.float64) / _w.sum())).astype(np.float32)

    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feat_names, free_raw_data=False)
""" % (k, repr(floor)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, help="run only this ramp length (plus the baseline)")
    ap.add_argument("--floor", type=float, default=0.25, help="weight at age 0 (with --k)")
    ap.add_argument("--force", action="store_true", help="ignore cache/age_weight_results.json")
    a = ap.parse_args()
    if a.k is not None and not (a.k > 0 and 0.0 <= a.floor < 1.0):
        sys.exit("--k must be > 0 and --floor in [0, 1)")
    configs = [(0, 1.0)] + ([(a.k, a.floor)] if a.k is not None else DEFAULT_CONFIGS)

    t_start = time.time()
    X, y, t, g, names = load_cache()
    print("cache: X_all %s (memmap) | %s series | %.1f%% positive rows"
          % (X.shape, format(np.unique(g).size, ","), 100 * y.mean()))
    E = import_engine()
    print("engine: NUM_ROUNDS %d | KEEP_EVERY %s | early_stopping(%d) | single booster per fit"
          % (E.NUM_ROUNDS, E.KEEP_EVERY, PATIENCE))

    folds = build_folds(t, g, E.KEEP_EVERY, seed=0)
    stride_all = stride_arr(t, E.KEEP_EVERY)
    age_all = break_age(y, t, g)
    n_break = int((np.unique(g[y == 1])).size)
    print("breaks: %s of %s series | first positive row per series has age 0 by construction: %s"
          % (format(n_break, ","), format(np.unique(g).size, ","),
             bool((age_all[y == 1] >= 0).all() and (age_all[y == 0] < 0).all())))

    # k = 0 must be exactly what research.ipynb / train() use today
    for F in folds:
        cur = stride_all[F["tr"]].astype(np.float32)
        assert np.array_equal(age_weights(cur, age_all[F["tr"]], 0, 1.0), cur), "k=0 changed the weights"

    fp = fingerprint(E, X)
    done = load_results(fp, a.force)

    for k, floor in configs:
        lab = label(k, floor)
        fits = done.get(lab, {}).get("fits", [])
        if len(fits) == 2 * N_FOLDS:
            print("\n[%s] cached: %s" % (lab, " ".join("%.5f" % f["auc"] for f in fits)))
            continue
        print("\n[%s]" % lab)
        fits = fits[:]                                         # resume a partly-finished config
        for fi, F in enumerate(folds):
            w_tr = age_weights(stride_all[F["tr"]].astype(np.float32), age_all[F["tr"]], k, floor)
            if k:
                r = (age_all[F["tr"]] >= 0) & (age_all[F["tr"]] < k)
                print("  fold %d: %s ramped rows (%.2f%% of train), total weight preserved: %s"
                      % (fi, format(int(r.sum()), ","), 100 * r.mean(),
                         bool(np.isclose(w_tr.sum(dtype=np.float64),
                                         stride_all[F["tr"]].sum(dtype=np.float64), rtol=1e-5))))
            for swap in (False, True):
                if any(f["fold"] == fi and f["swap"] == swap for f in fits):
                    continue
                t0 = time.time()
                auc, it = fit_one(E, X, y, t, names, F, w_tr, swap)
                fits.append(dict(fold=fi, swap=swap, auc=auc, trees=it, secs=round(time.time() - t0, 1)))
                print("  fold %d %s: TS-AUC %.5f | trees %4d | %5.1fs"
                      % (fi, "score-half" if not swap else "stop-half ", auc, it, time.time() - t0),
                      flush=True)
                done[lab] = dict(k=k, floor=floor, fits=fits)
                save_results(fp, done)                         # after every fit: safe to interrupt
            del w_tr; gc.collect()

    # ------------------------------------------------------------------ verdict
    def vec(lab):
        fs = sorted(done[lab]["fits"], key=lambda f: (f["fold"], f["swap"]))
        return np.array([f["auc"] for f in fs]), np.array([f["trees"] for f in fs])

    base, base_trees = vec(label(0, 1.0))
    print("\nbaseline per fit: %s" % " ".join("%.5f" % v for v in base))
    print("\n%-18s %9s %9s %8s %7s %6s  %s" % ("config", "mean", "delta", "se", "better", "trees", "verdict"))
    print("%-18s %9.5f %9s %8s %7s %6.0f  %s" % (label(0, 1.0), base.mean(), "-", "-", "-", base_trees.mean(), "-"))
    keeps = []
    for k, floor in configs[1:]:
        lab = label(k, floor)
        a_, tr_ = vec(lab)
        d = a_ - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        better = int((d > 0).sum())
        keep = d.mean() > 2 * se and better >= KEEP_MIN_BETTER
        print("%-18s %9.5f %+9.5f %8.5f %5d/%d %6.0f  %s"
              % (lab, a_.mean(), d.mean(), se, better, len(d), tr_.mean(), "KEEP" if keep else "REJECT"))
        print("%-18s per-fit delta: %s" % ("", " ".join("%+.5f" % v for v in d)))
        if keep:
            keeps.append((d.mean(), k, floor))

    print("\nrule: KEEP only if delta > 2*se and >= %d/6 fits improved" % KEEP_MIN_BETTER)
    print("wall clock: %.1f min" % ((time.time() - t_start) / 60))
    if keeps:
        _, k, floor = max(keeps)
        print("\nKEEP: k=%d floor=%s" % (k, floor))
        print_edit(k, floor)
    else:
        print("\nno configuration passes -- nothing to change in train()")


if __name__ == "__main__":
    main()
