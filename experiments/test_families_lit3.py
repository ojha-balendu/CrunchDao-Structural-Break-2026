"""Test five literature feature families against the model we ship (LightGBM x3 + CatBoost, 109 features).

    caffeinate -i python -u experiments/test_families_lit3.py > cache/family_tests.log 2>&1

Stage 0  checks (feature set, garch names vs the shipped registry, x_series, the five new families)
Stage 1  extract each new family once (cache/new_<name>.npz), univariate table
Stage 2  screen: LightGBM only, 109 vs 109+family, fold seed 17, 6 paired fold-halves
         confirm: full blend (0.7 * mean(lgb seeds) + 0.3 * catboost) for families with delta > 1 sigma
         combined: all confirmed families together, if two or more pass

Reads cache/ and the converted submission.ipynb; never modifies the notebook or the shipped model.
Everything is resumable: results -> cache/family_tests.json, per-fit predictions -> cache/family_tests/.
Memory: one new family in RAM at a time, fold matrices built per fold-half and freed after it.
"""
import argparse, contextlib, gc, hashlib, importlib, io, json, math, os, subprocess, sys, tempfile, time

import numpy as np

import tune_overnight as TO
from tune_overnight import (load_features, feature_report, build_folds, rows, ts_auc, stride_arr,
                            LGB_BASE, LGB_SHIPPED, NUM_ROUNDS, PATIENCE)

CACHE = "cache"
RESULTS = f"{CACHE}/family_tests.json"
PRED_DIR = f"{CACHE}/family_tests"
FOLD_SEED = 17
FAMILIES = ["recres", "wavvar", "ordpat", "evt", "hmm"]
CLASSES = {"recres": "RecursiveResidualFamily", "wavvar": "WaveletVarianceFamily",
           "ordpat": "OrdinalPatternFamily", "evt": "EVTTailFamily", "hmm": "HMMRegimeFamily"}
GARCH_FAMS = ["garch", "tgarch", "argarch", "gjr"]
PER_SERIES_CONSTANTS = {"evt_xi"}          # legitimately constant within one series
SCREEN_SIGMA = 1.0
KEEP_MIN_BETTER = 5
LGB_PATIENCE = 80                          # production train(): lgb.early_stopping(80)
CAT_PATIENCE = 100                         # production train(): early_stopping_rounds=100
UNI_SAMPLE = 400_000


def log(*a):
    print(*a, flush=True)


# ============================================================================ engine
def load_engine():
    """Convert submission.ipynb, import it by name (fork workers inherit it), exec lit3_families.py
    inside the module's own namespace so the classes see _CondBase, _history, _safe, ..."""
    d = tempfile.mkdtemp(prefix="famtest_")
    out = os.path.join(d, "_famtest_main.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", "submission.ipynb", out,
                        "--override"], capture_output=True, text=True)
    assert r.returncode == 0, "conversion failed:\n" + r.stdout[-1500:] + r.stderr[-1500:]
    sys.path.insert(0, d)
    with contextlib.redirect_stdout(io.StringIO()):
        m = importlib.import_module("_famtest_main")
    n_before = len(m.REGISTRY)
    exec(compile(open("feature_research/lit3_families.py").read(), "feature_research/lit3_families.py", "exec"), m.__dict__)
    assert len(m.REGISTRY) == n_before, "lit3_families.py registered something on import"
    return m


def new_family(m, name):
    fam = getattr(m, CLASSES[name])()
    assert fam.name == name, (fam.name, name)
    return fam


# ============================================================================ results store
def load_store(fp):
    if os.path.exists(RESULTS):
        s = json.load(open(RESULTS))
        if s.get("fingerprint") == fp:
            return s
        log("!! %s is for a different setup (fingerprint changed) -- starting fresh" % RESULTS)
    return {"fingerprint": fp, "fold_seed": FOLD_SEED, "extract": {}, "univariate": {},
            "screen": {}, "confirm": {}, "combined": {}}


def save_store(s):
    tmp = RESULTS + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1)
    os.replace(tmp, RESULTS)


def pred_path(key, label, model):
    return os.path.join(PRED_DIR, "%s__%s__%s.npy" % (key, label.replace("+", "-"), model))


# ============================================================================ stage 0
def stage0(m, names, groups, lgb_params, cat_params, cat_weight):
    log("\n" + "#" * 78 + "\nSTAGE 0 -- checks\n" + "#" * 78)
    feature_report(groups, names)                                           # 0.1 (asserts inside)

    # 0.1b column order = the shipped artifact's order (LightGBM column sampling depends on it)
    prod = list(m.FEATURES)
    assert len(prod) == 109 and set(prod) == set(names), "shipped registry features != the 109 loaded"
    order = [names.index(n) for n in prod]
    log("0.1b shipped FEATURES order %s the loaded order -> training uses the shipped order"
        % ("matches" if order == list(range(109)) else "DIFFERS from"))

    # 0.2 garch-family names == the shipped registry's, same order
    reg = {f.name: f for f in m.REGISTRY}
    for fam in GARCH_FAMS:
        assert groups[fam] == list(reg[fam].names()), "%s: cache names != registry names" % fam
        log("0.2  %-8s %2d names == shipped registry (same order)" % (fam, len(groups[fam])))

    # 0.3 HistoricalRef.x_series
    xh = np.random.default_rng(0).standard_t(5, 600)
    xh[17] = np.nan
    h = m.HistoricalRef(xh)
    assert hasattr(h, "x_series"), "HistoricalRef has no x_series"
    assert np.array_equal(h.x_series, xh[np.isfinite(xh)]), "x_series is not the ordered finite history"
    log("0.3  HistoricalRef.x_series present: %d ordered finite values (of 600 incl. one NaN)" % h.x_series.size)

    # 0.4 the five families
    base_set = set(names)
    rng = np.random.default_rng(7)
    synth = []
    for df, nh, T in ((3.0, 300, 400), (6.0, 900, 400), (40.0, 2000, 400)):
        xh = rng.standard_t(df, nh)
        xo = rng.standard_t(df, T); xo[200:] *= 1.7                        # a break halfway
        synth.append((df, xh, xo))
    for name in FAMILIES:
        fam = new_family(m, name)
        nm = list(fam.names())
        assert len(nm) == len(set(nm)), "%s: duplicate names" % name
        clash = sorted(set(nm) & base_set)
        assert not clash, "%s collides with baseline names %s" % (name, clash)
        reg_names = {x for f in m.REGISTRY for x in f.names()}
        assert not (set(nm) & reg_names), "%s collides with the shipped registry" % name
        cost = getattr(fam, "cost", None)
        if cost != "O1":
            # declared cost vs forced O1: identical on the steps the declared stride recomputes
            for df, xh, xo in synth:
                decl = m.StreamFeatures(xh, active_families={name}, registry=[fam])
                forced = new_family(m, name); forced.cost = "O1"
                o1 = m.StreamFeatures(xh, active_families={name}, registry=[forced])
                for t, v in enumerate(xo):
                    a, b = decl.update(v), o1.update(v)
                    if m.stride_for(cost, t) == 1 or t % m.stride_for(cost, t) == 0:
                        assert np.allclose(a, b, rtol=0, atol=1e-12), "%s not stride-safe at t=%d" % (name, t)
            stride_msg = "cost=%s, recompute steps identical to forced O1" % cost
        else:
            stride_msg = "cost=O1 (recomputes every step)"
        R = []
        for df, xh, xo in synth:
            fx = m.StreamFeatures(xh, active_families={name}, registry=[fam])
            assert fx.names == nm, (name, fx.names)
            for v in xo:
                r = fx.update(v)
                assert len(r) == len(nm), "%s: width %d != %d names" % (name, len(r), len(nm))
                R.append(r)
        R = np.asarray(R, dtype=np.float64)
        bad = [nm[c] for c in np.unique(np.nonzero(~np.isfinite(R))[1])]
        assert not bad, "%s: non-finite in %s" % (name, bad)
        const = [nm[c] for c in range(R.shape[1]) if R[:, c].std() == 0 and nm[c] not in PER_SERIES_CONSTANTS]
        assert not const, "%s: constant columns %s" % (name, const)
        log("0.4  %-7s %2d cols | 3 synthetic series (t df 3/6/40, hist 300/900/2000) | finite, none constant"
            " | no name collisions | %s" % (name, len(nm), stride_msg))

    # 0.5 the baseline model == production
    prod_lgb = dict(m.LGB_PARAMS)
    to_lgb = dict(LGB_BASE); to_lgb.update(LGB_SHIPPED)
    diffs = {k: (to_lgb.get(k), prod_lgb.get(k)) for k in sorted(set(to_lgb) | set(prod_lgb))
             if to_lgb.get(k) != prod_lgb.get(k)}
    for k in LGB_SHIPPED:
        assert prod_lgb[k] == LGB_SHIPPED[k], "shipped notebook LGB %s=%s != LGB_SHIPPED %s" % (k, prod_lgb[k], LGB_SHIPPED[k])
    log("0.5  LGB tuned params == LGB_SHIPPED: OK")
    log("     tune_overnight LGB_BASE+LGB_SHIPPED vs the shipped LGB_PARAMS (tune_overnight, shipped):")
    for k, (a, b) in diffs.items():
        log("       %-20s %-14s %s" % (k, a, b))
    log("     -> training uses the SHIPPED LGB_PARAMS, early stopping %d (production train())" % LGB_PATIENCE)
    assert dict(m.CAT_PARAMS) == cat_params, "notebook CAT_PARAMS != blend_choice.json cat_params"
    assert m.CAT_WEIGHT == cat_weight == 0.3
    log("0.5  CatBoost params == blend_choice.json == notebook CAT_PARAMS | weight %.1f: OK" % cat_weight)
    log("     SEEDS %s | NUM_ROUNDS %d" % (tuple(m.SEEDS), NUM_ROUNDS))
    return order


# ============================================================================ stage 1
def stage1(m, store, y_all, t_all, g_all, extras, groups, X_all, names):
    log("\n" + "#" * 78 + "\nSTAGE 1 -- extraction and univariate screen\n" + "#" * 78)
    train_data = None
    for name in FAMILIES:
        path = f"{CACHE}/new_{name}.npz"
        want = list(new_family(m, name).names())
        if os.path.exists(path):
            with np.load(path) as Z:
                have = [str(s) for s in Z["names"]]
            if have != want:
                log("  %s: cached names differ from names() -- deleting stale %s" % (name, path))
                os.remove(path)
        if not os.path.exists(path):
            if train_data is None:
                import crunch
                with contextlib.redirect_stdout(io.StringIO()):
                    train_data, _ = crunch.load_notebook().load_data()
                train_data = list(train_data)
            fam = new_family(m, name)
            m.REGISTRY[:] = [f for f in m.REGISTRY if f.name != name] + [fam]
            t0 = time.time()
            X, y, t, _, g = m.build_dataset(train_data, subsample=False, active_families={name},
                                            n_jobs=8, backend="fork")
            secs = time.time() - t0
            m.REGISTRY[:] = [f for f in m.REGISTRY if f.name != name]
            assert X.shape == (y_all.size, len(want)), (X.shape, y_all.size, len(want))
            assert np.array_equal(y, y_all) and np.array_equal(t, t_all) and np.array_equal(g, g_all), \
                "%s: y/t/g differ from the cache -- row order mismatch" % name
            assert np.isfinite(X).all(), "%s: non-finite values in the extraction" % name
            np.savez(path, X=X.astype(np.float32), names=np.array(want))
            store["extract"][name] = dict(secs=round(secs, 1), shape=list(X.shape))
            save_store(store)
            log("  %s: extracted %s in %.1f min -> %s" % (name, X.shape, secs / 60, path))
            del X, y, t, g; gc.collect()
        else:
            log("  %s: reusing %s (names match names())" % (name, path))
    del train_data; gc.collect()

    # univariate: TS-AUC per column + max |corr| with the 36 garch columns and with e2_w256
    rng = np.random.default_rng(0)
    samp = np.sort(rng.choice(y_all.size, size=UNI_SAMPLE, replace=False))
    garch_cols = np.hstack([e[samp] for e in extras]).astype(np.float64)
    e2 = np.asarray(X_all[samp, names_all_index("e2_w256")], dtype=np.float64)
    for name in FAMILIES:
        if name in store["univariate"]:
            continue
        with np.load(f"{CACHE}/new_{name}.npz") as Z:
            X = Z["X"]; nm = [str(s) for s in Z["names"]]
        tab = []
        for j, col in enumerate(nm):
            a = ts_auc(y_all, t_all, X[:, j])
            s = X[samp, j].astype(np.float64)
            cg = max(abs_corr(s, garch_cols[:, i]) for i in range(garch_cols.shape[1]))
            ce = abs_corr(s, e2)
            tab.append(dict(col=col, auc=max(a, 1 - a), raw=a, corr_garch=cg, corr_e2w256=ce))
        store["univariate"][name] = tab
        save_store(store)
        del X; gc.collect()
    for name in FAMILIES:
        log("\n  %s -- univariate (TS-AUC over all %s rows; |corr| on %dk sampled rows)"
            % (name, format(y_all.size, ","), UNI_SAMPLE // 1000))
        log("  %-24s %8s %8s %12s %12s" % ("column", "ts_auc", "raw", "max|c| garch", "|c| e2_w256"))
        for r in sorted(store["univariate"][name], key=lambda r: -r["auc"]):
            log("  %-24s %8.4f %8.4f %12.3f %12.3f" % (r["col"], r["auc"], r["raw"], r["corr_garch"], r["corr_e2w256"]))
    del garch_cols, e2; gc.collect()


_NAMES_ALL = None
def names_all_index(n):
    global _NAMES_ALL
    if _NAMES_ALL is None:
        _NAMES_ALL = json.load(open(f"{CACHE}/names.json"))
    return _NAMES_ALL.index(n)


def abs_corr(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(abs(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)))


# ============================================================================ fits
def fit_lgb(m, Xtr, ytr, wtr, Xst, yst, Xev, feat_names):
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feat_names, free_raw_data=False)
    dst = lgb.Dataset(Xst, label=yst, feature_name=feat_names, reference=dtr, free_raw_data=False)
    preds, iters = [], []
    for s in m.SEEDS:                      # production train(): same params, same seed offsets
        p = dict(m.LGB_PARAMS)
        p.update(seed=s, bagging_seed=s + 100, feature_fraction_seed=s + 200)
        b = lgb.train(p, dtr, num_boost_round=NUM_ROUNDS, valid_sets=[dst],
                      callbacks=[lgb.early_stopping(LGB_PATIENCE, verbose=False)])
        preds.append(b.predict(Xev, num_iteration=b.best_iteration)); iters.append(int(b.best_iteration))
        del b
    del dtr, dst; gc.collect()
    return np.mean(preds, axis=0), iters


def fit_cat(m, Xtr, ytr, wtr, Xst, yst, Xev):
    from catboost import CatBoostClassifier
    c = CatBoostClassifier(**m.CAT_PARAMS, iterations=NUM_ROUNDS, early_stopping_rounds=CAT_PATIENCE,
                           random_seed=0, thread_count=8, verbose=0, allow_writing_files=False)
    c.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xst, yst))
    p = c.predict_proba(Xev)[:, 1]
    it = int(c.get_best_iteration())
    del c; gc.collect()
    return p, it


def fold_halves(y, t, g):
    out = []
    for k, F in enumerate(build_folds(y, t, g, FOLD_SEED)):
        for swap in (False, True):
            st, ev = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
            out.append(("f%d%s" % (k, "b" if swap else "a"), F["tr"], st, ev))
    return out


def load_family_block(name):
    with np.load(f"{CACHE}/new_{name}.npz") as Z:
        return np.asarray(Z["X"], dtype=np.float32), [str(s) for s in Z["names"]]


class Half:
    """The 109-column matrices of one fold-half (in the shipped column order), built once."""
    def __init__(self, X_all, base_idx, extras, order, y, t, stride, tr, st, ev):
        self.tr, self.st, self.ev = tr, st, ev
        self.Xtr = rows(X_all, base_idx, extras, tr)[:, order]
        self.Xst = rows(X_all, base_idx, extras, st)[:, order]
        self.Xev = rows(X_all, base_idx, extras, ev)[:, order]
        self.ytr, self.wtr = y[tr], stride[tr].astype(np.float32)
        self.yst, self.yev, self.tev = y[st], y[ev], t[ev]

    def plus(self, blocks):
        """Base + extra family columns, built fresh (the caller frees them after the fit)."""
        if not blocks:
            return self.Xtr, self.Xst, self.Xev
        cat = lambda B, idx: np.ascontiguousarray(np.hstack([B] + [b[idx] for b in blocks]), dtype=np.float32)
        return cat(self.Xtr, self.tr), cat(self.Xst, self.st), cat(self.Xev, self.ev)


def cached_fit(kind, key, label, fn):
    """Run fn() -> (pred, iters) unless the prediction file exists; return pred, iters, secs, cached."""
    path = pred_path(key, label, kind)
    meta = path + ".json"
    if os.path.exists(path) and os.path.exists(meta):
        md = json.load(open(meta))
        return np.load(path), md["iters"], md["secs"], True
    t0 = time.time()
    p, it = fn()
    secs = round(time.time() - t0, 1)
    np.save(path, np.asarray(p, dtype=np.float64))
    json.dump({"iters": it, "secs": secs}, open(meta, "w"))
    return p, it, secs, False


# ============================================================================ stage 2
def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / math.sqrt(d.size)
    return float(d.mean()), float(se), (float(d.mean() / se) if se > 0 else float("inf")), int((d > 0).sum())


def stage2(m, store, X_all, base_idx, extras, order, names, y, t, g, stride):
    os.makedirs(PRED_DIR, exist_ok=True)
    halves = fold_halves(y, t, g)
    base_names = [names[i] for i in order]
    fam_names = {n: list(new_family(m, n).names()) for n in FAMILIES}
    t_start = time.time()

    # ------------------------------------------------------------- screen (LightGBM only)
    log("\n" + "#" * 78 + "\nSTAGE 2a -- screen: LightGBM (3 seeds), 109 vs 109 + family, fold seed %d\n" % FOLD_SEED + "#" * 78)
    for hi, (key, tr, st, ev) in enumerate(halves):
        rec = store["screen"].setdefault(key, {})
        if all(l in rec for l in ["base"] + FAMILIES):
            log("[screen %s] cached: base %.5f | %s" % (key, rec["base"]["auc"],
                " ".join("%s %+.5f" % (f, rec[f]["auc"] - rec["base"]["auc"]) for f in FAMILIES)))
            continue
        th = time.time()
        H = Half(X_all, base_idx, extras, order, y, t, stride, tr, st, ev)
        log("[screen %s] train %s | stop %s | eval %s" % (key, H.Xtr.shape, H.Xst.shape, H.Xev.shape))
        for label in ["base"] + FAMILIES:
            if label in rec:
                continue
            if label == "base":
                fn = lambda: fit_lgb(m, H.Xtr, H.ytr, H.wtr, H.Xst, H.yst, H.Xev, base_names)
            else:
                blk, _ = load_family_block(label)
                Xtr, Xst, Xev = H.plus([blk]); del blk
                fn = lambda: fit_lgb(m, Xtr, H.ytr, H.wtr, Xst, H.yst, Xev, base_names + fam_names[label])
            p, it, secs, was = cached_fit("lgb", key, label, fn)
            a = ts_auc(H.yev, H.tev, p)
            rec[label] = dict(auc=a, trees=it, secs=secs)
            save_store(store)
            log("  screen | %-6s | fold %s half %s | lgb x3 | TS-AUC %.5f | trees %s | %5.0fs%s"
                % (label, key[1], key[2], a, it, secs, " (cached)" if was else ""))
            if label != "base":
                del Xtr, Xst, Xev
            del p; gc.collect()
        del H; gc.collect()
        if hi == 0:
            el = time.time() - th
            log("\n  >> first fold-half took %.1f min -> screening ~%.0f min total; confirm adds ~%.0f min per"
                " passing family plus ~%.0f min for the baseline blend\n"
                % (el / 60, el * 6 / 60, 6 * 1.5, 6 * 1.5))

    log("\nSCREEN (LightGBM x3) -- delta vs 109-feature baseline, fold seed %d" % FOLD_SEED)
    log("%-8s %6s %10s %9s %7s %8s  %s" % ("family", "cols", "delta", "se", "sigma", "better", "-> confirm?"))
    ks = [h[0] for h in halves]
    base_lgb = [store["screen"][k]["base"]["auc"] for k in ks]
    log("%-8s %6d %10s %9s %7s %8s    mean %.5f" % ("base", 109, "-", "-", "-", "-", np.mean(base_lgb)))
    passing = []
    for f in FAMILIES:
        d, se, sig, better = paired([store["screen"][k][f]["auc"] for k in ks], base_lgb)
        go = sig > SCREEN_SIGMA and d > 0
        passing += [f] if go else []
        log("%-8s %6d %+10.5f %9.5f %7.2f %6d/6  %s" % (f, len(fam_names[f]), d, se, sig, better, "YES" if go else "no"))
    store["screen_passing"] = passing
    save_store(store)

    # ------------------------------------------------------------- confirm (full blend)
    log("\n" + "#" * 78 + "\nSTAGE 2b -- confirm: blend 0.7 * lgb + 0.3 * catboost, for: %s\n" % (passing or "none") + "#" * 78)
    w = m.CAT_WEIGHT
    combos = [(f, [f]) for f in passing]
    for key, tr, st, ev in halves:
        rec = store["confirm"].setdefault(key, {})
        todo = ["base"] + passing
        if all(l in rec for l in todo):
            continue
        H = Half(X_all, base_idx, extras, order, y, t, stride, tr, st, ev)
        for label in todo:
            if label in rec:
                continue
            if label == "base":
                pl, lit, _, _ = cached_fit("lgb", key, "base", None)
                fn = lambda: fit_cat(m, H.Xtr, H.ytr, H.wtr, H.Xst, H.yst, H.Xev)
                pc, cit, secs, was = cached_fit("cat", key, "base", fn)
            else:
                pl, lit, _, _ = cached_fit("lgb", key, label, None)       # reuse the screening LGB
                blk, _ = load_family_block(label)
                Xtr, Xst, Xev = H.plus([blk]); del blk
                fn = lambda: fit_cat(m, Xtr, H.ytr, H.wtr, Xst, H.yst, Xev)
                pc, cit, secs, was = cached_fit("cat", key, label, fn)
                del Xtr, Xst, Xev
            pb = (1.0 - w) * pl + w * pc
            rec[label] = dict(blend=ts_auc(H.yev, H.tev, pb), lgb=ts_auc(H.yev, H.tev, pl),
                              cat=ts_auc(H.yev, H.tev, pc), lgb_trees=lit, cat_iters=cit, cat_secs=secs)
            save_store(store)
            log("  confirm | %-6s | fold %s half %s | blend %.5f (lgb %.5f, cat %.5f) | cat iters %d | %5.0fs%s"
                % (label, key[1], key[2], rec[label]["blend"], rec[label]["lgb"], rec[label]["cat"], cit, secs,
                   " (cached)" if was else ""))
            del pl, pc, pb; gc.collect()
        del H; gc.collect()

    # ------------------------------------------------------------- combined
    confirmed = []
    if passing:
        bb = [store["confirm"][k]["base"]["blend"] for k in ks]
        for f in passing:
            d, se, sig, better = paired([store["confirm"][k][f]["blend"] for k in ks], bb)
            if d > 2 * se and better >= KEEP_MIN_BETTER:
                confirmed.append(f)
    if len(confirmed) >= 2:
        label = "+".join(confirmed)
        log("\n" + "#" * 78 + "\nSTAGE 2c -- combined: %s (full blend)\n" % label + "#" * 78)
        for key, tr, st, ev in halves:
            rec = store["combined"].setdefault(key, {})
            if label in rec:
                continue
            H = Half(X_all, base_idx, extras, order, y, t, stride, tr, st, ev)
            blks = []
            for f in confirmed:
                b, _ = load_family_block(f); blks.append(b)
            Xtr, Xst, Xev = H.plus(blks); del blks; gc.collect()
            fnames = base_names + [n for f in confirmed for n in fam_names[f]]
            pl, lit, ls, _ = cached_fit("lgb", key, label, lambda: fit_lgb(m, Xtr, H.ytr, H.wtr, Xst, H.yst, Xev, fnames))
            pc, cit, cs, _ = cached_fit("cat", key, label, lambda: fit_cat(m, Xtr, H.ytr, H.wtr, Xst, H.yst, Xev))
            pb = (1.0 - w) * pl + w * pc
            rec[label] = dict(blend=ts_auc(H.yev, H.tev, pb), lgb=ts_auc(H.yev, H.tev, pl), cat=ts_auc(H.yev, H.tev, pc))
            save_store(store)
            log("  combined | fold %s half %s | blend %.5f | %5.0fs" % (key[1], key[2], rec[label]["blend"], ls + cs))
            del Xtr, Xst, Xev, H, pl, pc, pb; gc.collect()
    return halves, passing, confirmed, time.time() - t_start


# ============================================================================ report
def report(store, halves, passing, confirmed):
    ks = [h[0] for h in halves]
    log("\n" + "#" * 78 + "\nRESULTS (fold seed %d, 6 paired fold-halves)\n" % FOLD_SEED + "#" * 78)
    if not passing:
        log("no family passed the LightGBM screen (> %.0f sigma) -- nothing to confirm" % SCREEN_SIGMA)
        return
    bb = [store["confirm"][k]["base"]["blend"] for k in ks]
    bl = [store["confirm"][k]["base"]["lgb"] for k in ks]

    d, se, sig, better = paired(bb, bl)
    log("\nCONSISTENCY -- baseline blend vs baseline LightGBM on these folds: %+.5f (se %.5f, %.1f sigma, %d/6)"
        % (d, se, sig, better))
    if not (d > 2 * se and better >= KEEP_MIN_BETTER):
        log("!" * 78 + "\n!! THE BLEND DOES NOT BEAT LIGHTGBM HERE (seed 13 showed +0.001 at 6.7 sigma).\n"
            "!! Something in this baseline may differ from production -- read the verdicts with care.\n" + "!" * 78)
    else:
        log("   OK -- consistent with the seed-13 measurement that justified shipping the blend")

    log("\nCONFIRM (blend) -- delta vs the 109-feature blend")
    log("%-22s %10s %10s %9s %7s %8s  %s" % ("configuration", "mean", "delta", "se", "sigma", "better", "verdict"))
    log("%-22s %10.5f %10s %9s %7s %8s" % ("baseline blend", np.mean(bb), "-", "-", "-", "-"))
    rows_ = [(f, [store["confirm"][k][f]["blend"] for k in ks]) for f in passing]
    for key in (store["combined"].get(ks[0]) or {}):
        rows_.append((key, [store["combined"][k][key]["blend"] for k in ks]))
    for lab, a in rows_:
        d, se, sig, better = paired(a, bb)
        keep = d > 2 * se and better >= KEEP_MIN_BETTER
        log("%-22s %10.5f %+10.5f %9.5f %7.2f %6d/6  %s" % ("+" + lab, np.mean(a), d, se, sig, better,
                                                          "ADOPT" if keep else "REJECT"))
    log("\nrule: ADOPT only if delta > 2*se and >= %d/6 fold-halves better, measured on the BLEND" % KEEP_MIN_BETTER)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="stage 0 only")
    a = ap.parse_args()
    T0 = time.time()

    cfg = json.load(open(f"{CACHE}/blend_choice.json"))
    cat_params, cat_weight = cfg["cat_params"], cfg["cat_weight"]
    assert cat_params["grow_policy"] == "Depthwise" and cat_params["depth"] == 8
    assert cat_params["bootstrap_type"] == "Bernoulli" and cat_weight == 0.3

    m = load_engine()
    log("engine: %d families, %d features | lit3 classes: %s" % (len(m.REGISTRY), m.N_FEATURES,
        ", ".join(CLASSES[f] for f in FAMILIES)))
    X_all, base_idx, extras, names, groups, y, t, g = load_features()
    order = stage0(m, names, groups, dict(m.LGB_PARAMS), cat_params, cat_weight)
    log("\nSTAGE 0 PASSED")
    if a.check:
        return

    fp = hashlib.md5(json.dumps(dict(names=[names[i] for i in order], lgb={k: str(v) for k, v in m.LGB_PARAMS.items()},
                                     cat=m.CAT_PARAMS, seed=FOLD_SEED, fams={f: list(new_family(m, f).names()) for f in FAMILIES},
                                     rounds=NUM_ROUNDS, lp=LGB_PATIENCE, cp=CAT_PATIENCE), sort_keys=True).encode()).hexdigest()[:12]
    store = load_store(fp)
    stage1(m, store, y, t, g, extras, groups, X_all, names)
    stride = stride_arr(t)
    halves, passing, confirmed, secs2 = stage2(m, store, X_all, base_idx, extras, order, names, y, t, g, stride)
    report(store, halves, passing, confirmed)
    log("\ntotal wall clock this run: %.1f min" % ((time.time() - T0) / 60))


if __name__ == "__main__":
    main()
