"""
tune_overnight.py -- Optuna for the shipped 109-feature model (LightGBM now, CatBoost later).

    python experiments/tune_overnight.py --check                    # feature set + names vs the notebook registry, then exit
    python experiments/tune_overnight.py --model lgb --hours 7      # overnight LightGBM study
    python experiments/tune_overnight.py --model cat --hours 7      # CatBoost study on the same features (later)
    python experiments/tune_overnight.py --model lgb --confirm      # re-test trial 0 vs the top 2 on a fresh split

Standalone on purpose: it reads only cache/ files, so a kernel restart, a notebook re-run or a
Claude Code edit cannot change what it is tuning on. Studies are stored in SQLite under cache/, so an
interrupted run resumes where it stopped when you run the same command again.
"""
import argparse, hashlib, json, os, sys, time, gc
import numpy as np

CACHE = "cache"
KEEP_EVERY = ((24, 1), (128, 2), (512, 4), (10 ** 9, 8))          # must match the engine
DROPPED_PREFIXES = ("mood_", "ansari_", "iqr_", "fligner_")        # families removed from production
EXTRA_FAMILIES = [
    ("garch", lambda n: n == "garch_persist"),
    ("tgarch", lambda n: False),
    ("argarch", lambda n: False),
    ("gjr", lambda n: "asym" in n or n.endswith("_gamma")),
    ("recres", lambda n: False),
    ("wavvar", lambda n: False),
    ("hmm", lambda n: False),
]
EXPECTED = {"base": 73, "garch": 9, "tgarch": 11, "argarch": 11, "gjr": 5,
            "recres": 11, "wavvar": 20, "hmm": 11, "total": 151}

# the currently shipped LightGBM parameters (trial 15) -- enqueued as trial 0 of every study
LGB_BASE = dict(objective="binary", metric="binary_logloss", verbosity=-1, num_threads=8,
                bagging_freq=1, feature_pre_filter=False)
LGB_SHIPPED = dict(learning_rate=0.043114, num_leaves=45, min_data_in_leaf=794,
                   feature_fraction=0.539909, bagging_fraction=0.805950, lambda_l2=79.616230,
                   extra_trees=True, min_gain_to_split=0.408390)
NUM_ROUNDS, PATIENCE = int(os.environ.get("TUNE_ROUNDS", 3000)), 100


# ------------------------------------------------------------------ data
def load_features():
    names_all = json.load(open(f"{CACHE}/names.json"))
    base_idx = [i for i, n in enumerate(names_all) if not n.startswith(DROPPED_PREFIXES)]
    X_all = np.load(f"{CACHE}/X_all.npy", mmap_mode="r")
    y = np.load(f"{CACHE}/y_all.npy"); t = np.load(f"{CACHE}/t_all.npy"); g = np.load(f"{CACHE}/g_all.npy")
    extras, groups = [], {"base": [names_all[i] for i in base_idx]}
    for fam, drop in EXTRA_FAMILIES:
        Z = np.load(f"{CACHE}/new_{fam}.npz")
        nm = [str(s) for s in Z["names"]]
        keep = [i for i, n in enumerate(nm) if not drop(n)]
        assert Z["X"].shape[0] == X_all.shape[0], f"{fam}: row count differs from the cache"
        extras.append(np.ascontiguousarray(np.asarray(Z["X"], dtype=np.float32)[:, keep]))
        groups[fam] = [nm[i] for i in keep]
        del Z; gc.collect()
    names = [n for k in groups for n in groups[k]]
    return X_all, base_idx, extras, names, groups, y, t, g


def feature_report(groups, names):
    print("=" * 72)
    print("FEATURE SET")
    ok = True
    for k, v in groups.items():
        flag = "OK" if len(v) == EXPECTED[k] else f"EXPECTED {EXPECTED[k]}"
        ok &= len(v) == EXPECTED[k]
        print(f"  {k:8s} {len(v):3d}  {flag:12s} {v[0]} ... {v[-1]}")
    flag = "OK" if len(names) == EXPECTED["total"] else f"EXPECTED {EXPECTED['total']}"
    ok &= len(names) == EXPECTED["total"] and len(set(names)) == len(names)
    print(f"  {'TOTAL':8s} {len(names):3d}  {flag}   unique: {len(set(names)) == len(names)}")
    for p in DROPPED_PREFIXES:
        assert not any(n.startswith(p) for n in names), f"dropped family {p} leaked into the set"
    print("  dropped families absent:", ", ".join(p.rstrip("_") for p in DROPPED_PREFIXES))
    print("  fingerprint:", hashlib.md5("|".join(names).encode()).hexdigest()[:12])
    print("=" * 72)
    assert ok, "feature set does not match the shipped %d -- not starting" % EXPECTED["total"]


def convert_submission():
    """Convert submission.ipynb (as Crunch does) and import it in this process."""
    import contextlib, importlib.util, io, subprocess, tempfile
    out = os.path.join(tempfile.mkdtemp(prefix="tune_reg_"), "_tune_reg_main.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", "submission.ipynb", out, "--override"],
                       capture_output=True, text=True)
    assert r.returncode == 0, "conversion failed:\n" + r.stdout[-1500:] + r.stderr[-1500:]
    spec = importlib.util.spec_from_file_location("_tune_reg_main", out)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m


def registry_check(groups, names, m=None):
    """Every extra block's cached column names == what the converted notebook's family emits, same
    order; and the whole loaded set == the notebook's FEATURES. Raises on any mismatch."""
    m = m or convert_submission()
    reg = {f.name: f for f in m.REGISTRY}
    for fam, _ in EXTRA_FAMILIES:
        assert fam in reg, f"{fam}: not in the submission registry"
        want = list(reg[fam].names())
        if groups[fam] != want:
            raise AssertionError(f"{fam}: cached names != {type(reg[fam]).__name__}().names()\n"
                                 f"  cache    {groups[fam]}\n  registry {want}")
        print(f"  {fam:8s} {len(want):3d}  == {type(reg[fam]).__name__}().names() (same order)")
    feats = list(m.FEATURES)
    assert len(feats) == len(names) == EXPECTED["total"], (len(feats), len(names))
    if set(feats) != set(names):
        raise AssertionError(f"loaded set != notebook FEATURES | only in cache {sorted(set(names) - set(feats))} "
                             f"| only in notebook {sorted(set(feats) - set(names))}")
    print(f"  all {len(names)} loaded names == the notebook's FEATURES (as a set) | "
          f"{len(m.REGISTRY)} families | column order {'identical' if feats == names else 'differs (as before)'}")
    return m


def stride_arr(t):
    s = np.full(t.shape, KEEP_EVERY[-1][1], dtype=np.int32)
    for up, st in reversed(KEEP_EVERY):
        s[t < up] = st
    return s


def build_folds(y, t, g, seed, n_folds=3):
    uniq = np.unique(g)
    fu = np.empty(uniq.size, np.int32)
    for k, part in enumerate(np.array_split(np.random.default_rng(seed).permutation(uniq.size), n_folds)):
        fu[part] = k
    pos = np.searchsorted(uniq, g)
    fold_row = fu[pos]
    half = (np.random.default_rng(seed + 1).random(uniq.size) < 0.5)[pos]
    sub = (t % stride_arr(t)) == 0
    srt = lambda m: (lambda idx: idx[np.argsort(t[idx], kind="stable")])(np.flatnonzero(m))
    return [dict(tr=srt((fold_row != k) & sub), st=srt((fold_row == k) & half),
                 sc=srt((fold_row == k) & ~half)) for k in range(n_folds)]


def rows(X_all, base_idx, extras, idx):
    return np.ascontiguousarray(np.hstack([np.asarray(X_all[idx])[:, base_idx]] + [e[idx] for e in extras]),
                                dtype=np.float32)


# ------------------------------------------------------------------ metric (vectorized TS-AUC)
def ts_auc(y, t, s):
    t = np.asarray(t); s = np.nan_to_num(np.asarray(s, np.float64))
    _, gid = np.unique(t, return_inverse=True)
    o = np.lexsort((s, gid)); go, so = gid[o], s[o]; n = s.size
    new_run = np.r_[True, (go[1:] != go[:-1]) | (so[1:] != so[:-1])]
    new_grp = np.r_[True, go[1:] != go[:-1]]
    pos = np.arange(n, dtype=np.float64)
    gstart = np.maximum.accumulate(np.where(new_grp, pos, 0.0))
    rid = np.cumsum(new_run) - 1
    rstart = pos[new_run]; rlen = np.diff(np.append(rstart, n))
    r = np.empty(n); r[o] = ((rstart - gstart[new_run]) + (rlen + 1) / 2.0)[rid]
    y = np.asarray(y, np.float64); G = gid.max() + 1
    npos = np.bincount(gid, weights=y, minlength=G); ntot = np.bincount(gid, minlength=G)
    nneg = ntot - npos; rs = np.bincount(gid, weights=r * y, minlength=G)
    ok = (npos > 0) & (nneg > 0); w = npos[ok] * nneg[ok]
    return float((w * (rs[ok] - npos[ok] * (npos[ok] + 1) / 2) / w).sum() / w.sum()) if ok.any() else 0.5


# ------------------------------------------------------------------ fold cache (built once per run)
class FoldData:
    def __init__(self, X_all, base_idx, extras, names, y, t, g, stride, seed, need_score_both=False):
        self.folds = []
        for k, F in enumerate(build_folds(y, t, g, seed)):
            d = dict(Xtr=rows(X_all, base_idx, extras, F["tr"]), ytr=y[F["tr"]], wtr=stride[F["tr"]].astype(np.float32),
                     Xst=rows(X_all, base_idx, extras, F["st"]), yst=y[F["st"]], tst=t[F["st"]],
                     Xsc=rows(X_all, base_idx, extras, F["sc"]), ysc=y[F["sc"]], tsc=t[F["sc"]])
            self.folds.append(d)
            print(f"  fold {k}: train {d['Xtr'].shape} stop {d['Xst'].shape} score {d['Xsc'].shape}", flush=True)
        self.names = names
        self._lgb = {}

    def lgb_sets(self, k):
        import lightgbm as lgb
        if k not in self._lgb:                       # binned once, reused by every trial
            d = self.folds[k]
            tr = lgb.Dataset(d["Xtr"], label=d["ytr"], weight=d["wtr"], feature_name=self.names,
                             params={"feature_pre_filter": False}, free_raw_data=False)
            st = lgb.Dataset(d["Xst"], label=d["yst"], reference=tr, free_raw_data=False)
            tr.construct(); st.construct()
            self._lgb[k] = (tr, st)
        return self._lgb[k]


# ------------------------------------------------------------------ one fit per model
def fit_lgb(fd, k, params, seeds, swap=False):
    import lightgbm as lgb
    d = fd.folds[k]
    tr, st = fd.lgb_sets(k)
    if swap:
        raise NotImplementedError("swap uses separate datasets; only used in --confirm")
    preds, iters = [], []
    for s in seeds:
        p = dict(LGB_BASE); p.update(params)
        p.update(seed=s, bagging_seed=s + 100, feature_fraction_seed=s + 200)
        b = lgb.train(p, tr, num_boost_round=NUM_ROUNDS, valid_sets=[st],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        preds.append(b.predict(d["Xsc"], num_iteration=b.best_iteration)); iters.append(b.best_iteration)
    return ts_auc(d["ysc"], d["tsc"], np.mean(preds, axis=0)), int(np.mean(iters))


def fit_cat(fd, k, params, seeds):
    from catboost import CatBoostClassifier
    d = fd.folds[k]
    preds, iters = [], []
    for s in seeds:
        m = CatBoostClassifier(**params, iterations=NUM_ROUNDS, early_stopping_rounds=PATIENCE, random_seed=s,
                               thread_count=8, verbose=0, allow_writing_files=False)
        m.fit(d["Xtr"], d["ytr"], sample_weight=d["wtr"], eval_set=(d["Xst"], d["yst"]))
        preds.append(m.predict_proba(d["Xsc"])[:, 1]); iters.append(m.get_best_iteration())
    return ts_auc(d["ysc"], d["tsc"], np.mean(preds, axis=0)), int(np.mean(iters))


# ------------------------------------------------------------------ search spaces
def lgb_space(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 20, 160, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 300, 5000, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.3, 0.9),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 0.95),
        lambda_l2=trial.suggest_float("lambda_l2", 5.0, 300.0, log=True),
        extra_trees=trial.suggest_categorical("extra_trees", [True, False]),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    )


def cat_space(trial):
    gp = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
    p = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        depth=trial.suggest_int("depth", 4, 9),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
        rsm=trial.suggest_float("rsm", 0.3, 1.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        bootstrap_type="Bernoulli",
        subsample=trial.suggest_float("subsample", 0.5, 0.95),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 3000, log=True),
        grow_policy=gp,
    )
    if gp == "Lossguide":
        p["max_leaves"] = trial.suggest_int("max_leaves", 16, 128, log=True)
    return p


CAT_DEFAULT = dict(learning_rate=0.05, depth=6, l2_leaf_reg=10.0, rsm=0.8, random_strength=1.0,
                   bootstrap_type="Bernoulli", subsample=0.8, min_data_in_leaf=500, grow_policy="SymmetricTree")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", choices=["lgb", "cat"], default="lgb")
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--fold-seed", type=int, default=11)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()

    X_all, base_idx, extras, names, groups, y, t, g = load_features()
    feature_report(groups, names)
    if a.check:
        print("REGISTRY NAMES (converted submission.ipynb)")
        registry_check(groups, names)
        print(f"TOTAL {len(names)} OK")
        return

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    stride = stride_arr(t)
    store = f"sqlite:///{CACHE}/optuna_{a.model}109.db"
    study_name = f"{a.model}109_fold{a.fold_seed}"

    if a.confirm:
        return confirm(a, X_all, base_idx, extras, names, y, t, g, stride, store, study_name)

    print(f"building folds (seed {a.fold_seed}) ...", flush=True)
    fd = FoldData(X_all, base_idx, extras, names, y, t, g, stride, a.fold_seed)
    seeds = (0, 1, 2) if a.model == "lgb" else (0,)
    fit = fit_lgb if a.model == "lgb" else fit_cat
    space = lgb_space if a.model == "lgb" else cat_space

    def objective(trial):
        params = space(trial)
        t0, aucs, iters = time.time(), [], []
        for k in range(len(fd.folds)):
            auc, it = fit(fd, k, params, seeds)
            aucs.append(auc); iters.append(it)
            trial.report(float(np.mean(aucs)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_aucs", [round(x, 5) for x in aucs])
        trial.set_user_attr("mean_trees", int(np.mean(iters)))
        print(f"trial {trial.number:3d}: {np.mean(aucs):.5f} | trees {int(np.mean(iters))} | "
              f"{(time.time()-t0)/60:.1f} min | {params}", flush=True)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", storage=store, study_name=study_name, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10, multivariate=True, group=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    if not study.trials:
        study.enqueue_trial(LGB_SHIPPED if a.model == "lgb" else
                            {k: v for k, v in CAT_DEFAULT.items() if k != "bootstrap_type"})
    print(f"study {study_name}: {len(study.trials)} trials already stored | running up to {a.hours} h", flush=True)
    study.optimize(objective, n_trials=a.trials, timeout=a.hours * 3600, gc_after_trial=True)

    done = [tr for tr in study.trials if tr.state.name == "COMPLETE"]
    done.sort(key=lambda tr: -tr.value)
    t0v = next((tr.value for tr in study.trials if tr.number == 0 and tr.value is not None), None)
    print("\nTOP 8 (delta vs trial 0 = the shipped parameters)")
    for tr in done[:8]:
        d = f"{tr.value - t0v:+.5f}" if t0v is not None else "  n/a"
        print(f"  #{tr.number:3d} {tr.value:.5f} {d} | trees {tr.user_attrs.get('mean_trees')} | {tr.params}")
    json.dump({"best_number": done[0].number, "best_value": done[0].value, "best_params": done[0].params,
               "trial0_value": t0v}, open(f"{CACHE}/best_{a.model}109.json", "w"), indent=2)
    print(f"\nwrote {CACHE}/best_{a.model}109.json -- run with --confirm before adopting anything")


def confirm(a, X_all, base_idx, extras, names, y, t, g, stride, store, study_name):
    """Trial 0 vs the top 2 on a FRESH split, both halves, 3 seeds: the best of N noisy trials is biased."""
    import optuna, lightgbm as lgb
    study = optuna.load_study(study_name=study_name, storage=store)
    done = sorted([tr for tr in study.trials if tr.state.name == "COMPLETE"], key=lambda tr: -tr.value)
    cands = {"trial0": study.trials[0].params}
    for tr in done[:2]:
        if tr.number != 0:
            cands[f"trial{tr.number}"] = tr.params
    folds = build_folds(y, t, g, a.fold_seed + 2)
    res = {k: [] for k in cands}
    for fi, F in enumerate(folds):
        Xtr = rows(X_all, base_idx, extras, F["tr"])
        for swap in (False, True):
            si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
            Xs, Xe = rows(X_all, base_idx, extras, si), rows(X_all, base_idx, extras, ei)
            for lab, params in cands.items():
                if a.model == "lgb":
                    tr = lgb.Dataset(Xtr, label=y[F["tr"]], weight=stride[F["tr"]].astype(np.float32),
                                     feature_name=names, params={"feature_pre_filter": False})
                    pr = []
                    for s in (0, 1, 2):
                        p = dict(LGB_BASE); p.update(params); p.update(seed=s, bagging_seed=s + 100, feature_fraction_seed=s + 200)
                        b = lgb.train(p, tr, num_boost_round=NUM_ROUNDS,
                                      valid_sets=[lgb.Dataset(Xs, label=y[si], reference=tr)],
                                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
                        pr.append(b.predict(Xe, num_iteration=b.best_iteration))
                else:
                    from catboost import CatBoostClassifier
                    pp = dict(params); pp.setdefault("bootstrap_type", "Bernoulli")
                    m = CatBoostClassifier(**pp, iterations=NUM_ROUNDS, early_stopping_rounds=PATIENCE, random_seed=0,
                                           thread_count=8, verbose=0, allow_writing_files=False)
                    m.fit(Xtr, y[F["tr"]], sample_weight=stride[F["tr"]].astype(np.float32), eval_set=(Xs, y[si]))
                    pr = [m.predict_proba(Xe)[:, 1]]
                res[lab].append(ts_auc(y[ei], t[ei], np.mean(pr, axis=0)))
                print(f"  fold {fi} swap={swap} {lab}: {res[lab][-1]:.5f}", flush=True)
    base = np.array(res["trial0"])
    print("\nCONFIRMATION (fresh split, both halves)")
    for lab, v in res.items():
        d = np.array(v) - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if lab != "trial0" else 0.0
        verdict = "" if lab == "trial0" else ("ADOPT" if d.mean() > 2 * se and (d > 0).sum() >= 5 else "keep current")
        print(f"  {lab:10s} {np.mean(v):.5f} | delta {d.mean():+.5f} | se {se:.5f} | {int((d>0).sum())}/{len(d)} {verdict}")


if __name__ == "__main__":
    main()
