"""LightGBM + XGBoost + CatBoost blend on the cached 73-feature matrix.

    python experiments/blend_experiment.py                      # stages 1-4 end to end
    python experiments/blend_experiment.py --stage 1            # tune XGB/Cat only
    python experiments/blend_experiment.py --trials 25 --fold-seed 7

Stage 1  tune XGBoost and CatBoost with Optuna (LightGBM is already tuned in LGB_PARAMS)
Stage 2  solo TS-AUC for each member on 6 paired fold-halves; eligibility = within 0.003 of LGB
Stage 3  blends (pairs + triple) by rank average and by probability average, reusing the
         member predictions already computed for stage 2 -- no member is trained twice
Stage 4  verdict table vs "LightGBM alone"

Folds, row order and weighting match research.ipynb exactly, so the LightGBM solo numbers are
directly comparable to that notebook's cv_cols() output (on the same fold seed).
Reads cache/ only; never modifies submission.ipynb or research.ipynb.
"""
import argparse
import ast
import contextlib
import gc
import hashlib
import importlib.util
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
CACHE = os.path.join(ROOT, "cache")
NOTEBOOK = os.path.join(ROOT, "submission.ipynb")
PARAMS_PATH = os.path.join(CACHE, "blend_params.json")
RESULTS_PATH = os.path.join(CACHE, "blend_results.json")
TUNING_DB = "sqlite:///" + os.path.join(CACHE, "blend_tuning.db")

DROP_FAMILIES = {"mood", "ansari", "iqr", "fligner"}
N_EXPECTED_FEATURES = 73
N_FOLDS = 3
PATIENCE = 100
LGB_SEEDS = (0, 1, 2)                 # the LightGBM member is the average of these
ELIGIBLE_MARGIN = 0.003               # a member must land within this of LightGBM to blend
KEEP_MIN_BETTER = 5
N_JOBS = 8
MEMBERS = ("lgb", "xgb", "cat")
COMBOS = [("lgb", "xgb"), ("lgb", "cat"), ("lgb", "xgb", "cat")]
RULES = ("rank", "prob")


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
    """Convert submission.ipynb the way Crunch does and import it, refusing to run training."""
    out = os.path.join(tempfile.mkdtemp(prefix="blend_"), "_blend_engine.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", NOTEBOOK, out, "--override"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("crunch_convert failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    forbidden = {"train", "infer", "test", "build_dataset", "local_test", "research_step"}
    for node in ast.parse(open(out).read()).body:
        call = getattr(node, "value", None)
        if isinstance(node, (ast.Expr, ast.Assign)) and isinstance(call, ast.Call):
            f = call.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if nm in forbidden:
                sys.exit("converted notebook calls %s() at import time (line %d)" % (nm, node.lineno))
    spec = importlib.util.spec_from_file_location("_blend_engine", out)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def production_columns(E, names):
    """Column indices of the 73 production features, from the REGISTRY mapping (never hardcoded).

    cache/X_all.npy predates the pruning: it has 85 columns, while the current submission.ipynb
    registers 14 families / 73 features. A cached name with no family in today's REGISTRY is
    therefore a pruned one -- checked here to be exactly the four families we mean to drop."""
    fam = {nm: f.name for f in E.REGISTRY for nm in f.names()}
    cols = [i for i, n in enumerate(names) if fam.get(n) is not None and fam[n] not in DROP_FAMILIES]
    dropped = [n for n in names if n not in cols and (fam.get(n) in DROP_FAMILIES or n not in fam)]
    stray = [n for n in dropped if not any(n.startswith(d + "_") for d in DROP_FAMILIES)]
    assert not stray, "dropping features that are not in %s: %s" % (sorted(DROP_FAMILIES), stray[:5])
    assert len(cols) == N_EXPECTED_FEATURES, \
        "expected %d production features, got %d (registry has %d families, %d features)" \
        % (N_EXPECTED_FEATURES, len(cols), len(E.REGISTRY), E.N_FEATURES)
    assert len(cols) == E.N_FEATURES, \
        "kept %d columns but submission.ipynb reports N_FEATURES=%d" % (len(cols), E.N_FEATURES)
    return np.array(cols), [names[i] for i in cols]


# ----------------------------------------------------------------------------- folds
def stride_arr(t, keep_every):
    s = np.full(t.shape, keep_every[-1][1], dtype=np.int32)
    for up_to, stride in reversed(keep_every):
        s[t < up_to] = stride
    return s


def build_folds(t, g, keep_every, seed):
    """Identical construction to research.ipynb build_folds()."""
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
        print("fold %d: train %s | stop %s | score %s"
              % (k, format(folds[-1]["tr"].size, ","), format(folds[-1]["st"].size, ","),
                 format(folds[-1]["sc"].size, ",")), flush=True)
    return folds


# ----------------------------------------------------------------------------- rank average
def ranks_within_t(t, score):
    """Tie-aware average ranks of `score` within each t group. One lexsort, no Python loop.

    TS-AUC only compares series inside one t, so averaging these ranks across members is a
    scale-free combination rule: a member cannot dominate by having wider-spread scores."""
    n = t.size
    order = np.lexsort((score, t))
    ts, ss = t[order], score[order]
    grp_head = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    grp_start = np.repeat(grp_head, np.diff(np.r_[grp_head, n]))
    ordinal = np.arange(n) - grp_start                     # 0-based position inside the t group
    run_head = np.flatnonzero(np.r_[True, (ts[1:] != ts[:-1]) | (ss[1:] != ss[:-1])])
    run_len = np.diff(np.r_[run_head, n])
    avg = ordinal[run_head] + (run_len - 1) / 2.0          # ties share their mean rank
    out = np.empty(n, dtype=np.float64)
    out[order] = np.repeat(avg, run_len)
    return out


def combine(rule, preds, t_eval):
    if rule == "prob":
        return np.mean([preds[m] for m in preds], axis=0)
    return np.mean([ranks_within_t(t_eval, preds[m]) for m in preds], axis=0)


# ----------------------------------------------------------------------------- members
# Each fit_* returns (predict_fn, iterations). The caller frees the training matrices before
# materializing the evaluation slice, so train+stop and eval are never resident together.
def fit_lgb(E, params, Xtr, ytr, wtr, Xst, yst):
    import lightgbm as lgb
    p = dict(E.LGB_PARAMS)
    p.update(objective="binary", metric="binary_logloss")
    models, trees = [], []
    for s in LGB_SEEDS:
        ps = dict(p, seed=s, bagging_seed=s + 100, feature_fraction_seed=s + 200,
                  data_random_seed=s + 300)
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        dva = lgb.Dataset(Xst, label=yst, reference=dtr)
        b = lgb.train(ps, dtr, num_boost_round=E.NUM_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        models.append(b)
        trees.append(int(b.best_iteration))
        del dtr, dva; gc.collect()

    def predict(Xev):
        out = np.zeros(Xev.shape[0])
        for b in models:
            out += b.predict(Xev, num_iteration=b.best_iteration)
        return out / len(models)

    return predict, float(np.mean(trees))


def fit_xgb(E, params, Xtr, ytr, wtr, Xst, yst):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=E.NUM_ROUNDS, tree_method="hist", eval_metric="logloss",
                          early_stopping_rounds=PATIENCE, n_jobs=N_JOBS, random_state=0, **params)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xst, yst)], verbose=False)
    return (lambda Xev: m.predict_proba(Xev)[:, 1]), float(m.best_iteration + 1)


def fit_cat(E, params, Xtr, ytr, wtr, Xst, yst):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=E.NUM_ROUNDS, early_stopping_rounds=PATIENCE,
                           thread_count=N_JOBS, verbose=0, allow_writing_files=False,
                           random_seed=0, **params)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xst, yst))
    return (lambda Xev: m.predict_proba(Xev)[:, 1]), float(m.get_best_iteration() + 1)


# ----------------------------------------------------------------------------- stage 1
def tune(lib, E, X, y, t, cols, folds, stride_all, n_trials, fold_seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest(tr):
        if lib == "xgb":
            return dict(learning_rate=tr.suggest_float("learning_rate", 0.02, 0.12, log=True),
                        max_depth=tr.suggest_int("max_depth", 4, 10),
                        min_child_weight=tr.suggest_float("min_child_weight", 1.0, 400.0, log=True),
                        subsample=tr.suggest_float("subsample", 0.5, 1.0),
                        colsample_bytree=tr.suggest_float("colsample_bytree", 0.3, 1.0),
                        reg_lambda=tr.suggest_float("reg_lambda", 0.5, 100.0, log=True),
                        gamma=tr.suggest_float("gamma", 1e-3, 5.0, log=True))
        return dict(learning_rate=tr.suggest_float("learning_rate", 0.02, 0.12, log=True),
                    depth=tr.suggest_int("depth", 4, 9),
                    l2_leaf_reg=tr.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
                    rsm=tr.suggest_float("rsm", 0.3, 1.0),
                    random_strength=tr.suggest_float("random_strength", 0.1, 10.0, log=True),
                    bagging_temperature=tr.suggest_float("bagging_temperature", 0.0, 2.0))

    fit = fit_xgb if lib == "xgb" else fit_cat

    def objective(trial):
        params = suggest(trial)
        aucs = []
        for fi, F in enumerate(folds):
            t0 = time.time()
            # one fit per fold: early-stop on the "stop" half, score the "score" half (swap=False)
            Xtr = np.ascontiguousarray(X[np.ix_(F["tr"], cols)])
            Xst = np.ascontiguousarray(X[np.ix_(F["st"], cols)])
            predict, it = fit(E, params, Xtr, y[F["tr"]], stride_all[F["tr"]].astype(np.float32),
                              Xst, y[F["st"]])
            del Xtr, Xst; gc.collect()
            Xsc = np.ascontiguousarray(X[np.ix_(F["sc"], cols)])
            a = float(E.ts_auc(y[F["sc"]], t[F["sc"]], predict(Xsc)))
            aucs.append(a)
            del Xsc, predict; gc.collect()
            print("    [tune %s] trial %d fold %d: TS-AUC %.5f | iters %.0f | %.0fs"
                  % (lib, trial.number, fi, a, it, time.time() - t0), flush=True)
            trial.report(float(np.mean(aucs)), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", storage=TUNING_DB,
                                study_name="%s_seed%d" % (lib, fold_seed), load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0))
    done = len([t_ for t_ in study.trials if t_.state.is_finished()])
    todo = max(0, n_trials - done)
    print("  [%s] %d trials already in the study, running %d more" % (lib, done, todo), flush=True)
    if todo:
        study.optimize(objective, n_trials=todo)
    return dict(study.best_params), float(study.best_value)


# ----------------------------------------------------------------------------- results cache
def fingerprint(E, X, cols, fold_seed):
    st = os.stat(os.path.join(CACHE, "X_all.npy"))
    return hashlib.sha256(json.dumps(dict(
        params={k: E.LGB_PARAMS[k] for k in sorted(E.LGB_PARAMS)}, rounds=E.NUM_ROUNDS,
        keep=E.KEEP_EVERY, patience=PATIENCE, seeds=list(LGB_SEEDS), cols=len(cols),
        shape=list(X.shape), size=st.st_size, mtime=int(st.st_mtime), fold_seed=fold_seed,
    ), sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_results(fp):
    if not os.path.exists(RESULTS_PATH):
        return {}
    d = json.load(open(RESULTS_PATH))
    if d.get("fingerprint") != fp:
        print("cached fold results are for a different setup -- recomputing")
        return {}
    return d.get("halves", {})


def save_results(fp, halves):
    tmp = RESULTS_PATH + ".tmp"
    json.dump({"fingerprint": fp, "halves": halves}, open(tmp, "w"), indent=1)
    os.replace(tmp, RESULTS_PATH)


# ----------------------------------------------------------------------------- stages 2+3
def run_fold_halves(E, X, y, t, cols, folds, stride_all, best, fp, halves):
    for fi, F in enumerate(folds):
        for swap in (False, True):
            key = "fold%d_swap%d" % (fi, int(swap))
            if key in halves:
                r = halves[key]
                print("[%s] cached: %s" % (key, " ".join("%s %.5f" % (m, r["members"][m]["auc"])
                                                         for m in MEMBERS)), flush=True)
                continue
            si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
            w = stride_all[F["tr"]].astype(np.float32)
            ytr, yst, yev = y[F["tr"]], y[si], y[ei]
            tev = t[ei]

            print("[%s] train %s | stop %s | eval %s"
                  % (key, format(len(F["tr"]), ","), format(len(si), ","), format(len(ei), ",")), flush=True)

            # train all three members on this fold-half (train+stop resident), then free those
            # matrices and materialize the eval slice once for every member's predictions
            fitted, rec = {}, {"members": {}, "blends": {}}
            Xtr = np.ascontiguousarray(X[np.ix_(F["tr"], cols)])
            Xst = np.ascontiguousarray(X[np.ix_(si, cols)])
            for m in MEMBERS:
                t0 = time.time()
                if m == "lgb":
                    predict, it = fit_lgb(E, None, Xtr, ytr, w, Xst, yst)
                elif m == "xgb":
                    predict, it = fit_xgb(E, best["xgb"]["params"], Xtr, ytr, w, Xst, yst)
                else:
                    predict, it = fit_cat(E, best["cat"]["params"], Xtr, ytr, w, Xst, yst)
                fitted[m] = (predict, it, time.time() - t0)
                gc.collect()
            del Xtr, Xst; gc.collect()

            Xev = np.ascontiguousarray(X[np.ix_(ei, cols)])
            preds = {}
            for m in MEMBERS:
                predict, it, secs = fitted[m]
                preds[m] = predict(Xev)
                a = float(E.ts_auc(yev, tev, preds[m]))
                rec["members"][m] = dict(auc=a, trees=it, secs=round(secs, 1))
                print("  fold %d %-10s %-3s: TS-AUC %.5f | iters %6.0f | %6.1fs"
                      % (fi, "stop-half" if swap else "score-half", m, a, it, secs), flush=True)
            del Xev, fitted; gc.collect()

            for combo in COMBOS:
                for rule in RULES:
                    s = combine(rule, {m: preds[m] for m in combo}, tev)
                    rec["blends"]["%s|%s" % ("+".join(combo), rule)] = float(E.ts_auc(yev, tev, s))
                    del s
            print("  blends: %s" % " ".join("%s %.5f" % (k, v) for k, v in rec["blends"].items()), flush=True)
            del preds; gc.collect()

            halves[key] = rec
            save_results(fp, halves)
    return halves


# ----------------------------------------------------------------------------- stage 4
def order_keys():
    return ["fold%d_swap%d" % (fi, s) for fi in range(N_FOLDS) for s in (0, 1)]


def report(halves, best, fold_seed, t_start):
    ks = order_keys()
    solo = {m: np.array([halves[k]["members"][m]["auc"] for k in ks]) for m in MEMBERS}
    trees = {m: np.mean([halves[k]["members"][m]["trees"] for k in ks]) for m in MEMBERS}
    secs = {m: np.mean([halves[k]["members"][m]["secs"] for k in ks]) for m in MEMBERS}
    base = solo["lgb"]

    print("\n" + "=" * 78)
    print("STAGE 2 -- solo scores (%d fold-halves, fold seed %d)" % (len(ks), fold_seed))
    print("%-22s %9s %10s %9s %8s %8s" % ("member", "mean", "delta", "se", "iters", "fit s"))
    print("%-22s %9.5f %10s %9s %8.0f %8.0f" % ("lgb (3-seed avg)", base.mean(), "-", "-", trees["lgb"], secs["lgb"]))
    eligible = []
    for m in ("xgb", "cat"):
        d = solo[m] - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        ok = d.mean() > -ELIGIBLE_MARGIN
        eligible.append(m) if ok else None
        print("%-22s %9.5f %+10.5f %9.5f %8.0f %8.0f  %s"
              % (m, solo[m].mean(), d.mean(), se, trees[m], secs[m],
                 "ELIGIBLE" if ok else "NOT ELIGIBLE (>%.3f below lgb)" % ELIGIBLE_MARGIN))
    print("eligible for blending: %s" % (", ".join(eligible) if eligible else "none"))

    print("\n" + "=" * 78)
    print("STAGE 3/4 -- blends vs LightGBM alone")
    print("%-28s %9s %10s %9s %7s  %s" % ("configuration", "mean", "delta", "se", "better", "verdict"))
    print("%-28s %9.5f %10s %9s %7s  %s" % ("lgb alone (3-seed)", base.mean(), "-", "-", "-", "-"))
    keeps = []
    for combo in COMBOS:
        extra = [m for m in combo if m != "lgb"]
        usable = all(m in eligible for m in extra)
        for rule in RULES:
            lab = "%s | %s" % ("+".join(combo), rule)
            a = np.array([halves[k]["blends"]["%s|%s" % ("+".join(combo), rule)] for k in ks])
            d = a - base
            se = d.std(ddof=1) / np.sqrt(len(d))
            better = int((d > 0).sum())
            keep = usable and d.mean() > 2 * se and better >= KEEP_MIN_BETTER
            verdict = "KEEP" if keep else ("REJECT (%s not eligible)" % ",".join(m for m in extra if m not in eligible)
                                           if not usable else "REJECT")
            print("%-28s %9.5f %+10.5f %9.5f %5d/%d  %s" % (lab, a.mean(), d.mean(), se, better, len(d), verdict))
            if keep:
                keeps.append((d.mean(), combo, rule, a.mean()))

    print("\nrule: KEEP only if delta > 2*se, >= %d/6 fits improved, and every member is eligible"
          % KEEP_MIN_BETTER)
    print("tuned parameters:")
    for lib in ("xgb", "cat"):
        print("  %s (best tuning CV %.5f): %s" % (lib, best[lib]["value"], json.dumps(best[lib]["params"])))
    print("wall clock: %.1f min" % ((time.time() - t_start) / 60))

    if keeps:
        _, combo, rule, mean = max(keeps)
        n_models = sum(len(LGB_SEEDS) if m == "lgb" else 1 for m in combo)
        print("\nKEEP %s | %s  (mean TS-AUC %.5f)" % ("+".join(combo), rule, mean))
        print("  mean trees/iterations per member: %s"
              % ", ".join("%s %.0f" % (m, trees[m]) for m in combo))
        print("  inference cost: %d model evaluations per step vs %d for the 3-seed LightGBM = %.2fx"
              % (n_models, len(LGB_SEEDS), n_models / len(LGB_SEEDS)))
        print("  (production train() currently fits ONE booster, so vs deployed it is %.2fx)" % n_models)
    else:
        print("\nno blend passes -- keep LightGBM alone")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4],
                    help="highest stage to run (1 = tune only; 2+ = fits, blends and verdict)")
    ap.add_argument("--trials", type=int, default=15, help="Optuna trials per library")
    ap.add_argument("--fold-seed", type=int, default=11)
    a = ap.parse_args()

    t_start = time.time()
    X, y, t, g, names = load_cache()
    E = import_engine()
    cols, feat_names = production_columns(E, names)
    print("cache: X_all %s (memmap) | using %d production features (dropped %s)"
          % (X.shape, len(cols), ", ".join(sorted(DROP_FAMILIES))))
    import lightgbm, xgboost, catboost
    print("versions: lightgbm %s | xgboost %s | catboost %s | NUM_ROUNDS %d | early stop %d"
          % (lightgbm.__version__, xgboost.__version__, catboost.__version__, E.NUM_ROUNDS, PATIENCE))

    folds = build_folds(t, g, E.KEEP_EVERY, a.fold_seed)
    stride_all = stride_arr(t, E.KEEP_EVERY)

    # ---------------- stage 1
    best = {}
    if os.path.exists(PARAMS_PATH):
        d = json.load(open(PARAMS_PATH))
        if d.get("fold_seed") == a.fold_seed and d.get("trials", 0) >= a.trials:
            best = d["best"]
            print("\nSTAGE 1: reusing %s (%d trials, fold seed %d)"
                  % (PARAMS_PATH, d["trials"], a.fold_seed))
    if not best:
        print("\nSTAGE 1 -- tuning XGBoost and CatBoost (%d trials each)" % a.trials)
        for lib in ("xgb", "cat"):
            p, v = tune(lib, E, X, y, t, cols, folds, stride_all, a.trials, a.fold_seed)
            best[lib] = dict(params=p, value=v)
            print("  [%s] best CV %.5f: %s" % (lib, v, json.dumps(p)), flush=True)
            json.dump({"fold_seed": a.fold_seed, "trials": a.trials, "best": best},
                      open(PARAMS_PATH, "w"), indent=1)
    for lib in ("xgb", "cat"):
        print("  %s params: %s" % (lib, json.dumps(best[lib]["params"])))
    if a.stage == 1:
        print("\nstage 1 only -- stop here. wall clock: %.1f min" % ((time.time() - t_start) / 60))
        return

    # ---------------- stages 2 and 3
    fp = fingerprint(E, X, cols, a.fold_seed)
    halves = load_results(fp)
    print("\nSTAGE 2/3 -- fitting members and combining (%d/%d fold-halves cached)"
          % (len(halves), 2 * N_FOLDS))
    halves = run_fold_halves(E, X, y, t, cols, folds, stride_all, best, fp, halves)

    # ---------------- stage 4
    report(halves, best, a.fold_seed, t_start)


if __name__ == "__main__":
    main()
