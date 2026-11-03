"""
balance_features.py -- give every family the SAME number of features, and test whether the trees learn
better when no property is over-represented.

    python experiments/balance_features.py --counts          # features per family + within-family redundancy (fast)
    python experiments/balance_features.py --cv              # LightGBM (3 seeds): full vs balanced variants
    python experiments/balance_features.py --cv --blend      # the same on the shipped LightGBM + CatBoost 0.3 blend

Variants (all selections/fits use ONLY each fold's training rows -- no leakage into the score half):
    top2 / top3 / top5   the k features with the highest LightGBM gain within each family
                         (gain from a quick fit on the fold's training rows); families with <= k keep all
    pca3                 each family (> 3 features) replaced by its first 3 principal components,
                         computed within t on the fold's training rows; smaller families kept as is
The `time` family (t_log etc.) is always kept whole: it carries no signal of its own, it calibrates.

Family membership comes from the converted submission.ipynb registry, so it always matches production.
Results resumable in cache/balance_cv[_blend].json.
"""
import argparse, json, os, subprocess, sys, time, gc, importlib.util
import numpy as np
from tune_overnight import (load_features, feature_report, build_folds, rows, ts_auc, stride_arr,
                            LGB_BASE, LGB_SHIPPED, NUM_ROUNDS, PATIENCE, CACHE)

KS = (2, 3, 5)
PCA_K = 3
ALWAYS_KEEP = {"time"}


def family_map(names, prefix_fallback=False):
    if prefix_fallback:                                  # only for synthetic tests
        return {n: n.split("_")[0] for n in names}
    out = "/tmp/_engine_balance.py"
    subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", "submission.ipynb", out, "--override"],
                   check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("_engine_balance", out)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    fam = {nm: f.name for f in m.REGISTRY for nm in f.names()}
    missing = [n for n in names if n not in fam]
    assert not missing, f"features not in the submission registry: {missing[:8]}"
    return {n: fam[n] for n in names}


def property_map(names, fmap, thr):
    """Group features by WHAT THEY MEASURE, across families: average-linkage clusters of the within-t
    rank correlation matrix from `prune_features.py --corr`, cut at |r| = thr. The time family stays
    its own group (it calibrates, it doesn't measure a property)."""
    p = f"{CACHE}/feature_corr.npz"
    assert os.path.exists(p), "run `python experiments/prune_features.py --corr` first"
    Z = np.load(p); cn = [str(x) for x in Z["names"]]
    assert cn == list(names), "feature_corr.npz is for a different feature set -- rerun prune_features --corr"
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1.0 - np.abs(np.nan_to_num(Z["C"])); np.fill_diagonal(D, 0); D = np.clip((D + D.T) / 2, 0, None)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=1.0 - thr, criterion="distance")
    return {n: ("time" if fmap[n] == "time" else f"p{int(l):02d}") for n, l in zip(names, lab)}


def describe_groups(names, gmap, fmap):
    grp = {}
    for n in names:
        grp.setdefault(gmap[n], []).append(n)
    print(f"\n{len(grp)} property groups (sizes {sorted((len(v) for v in grp.values()), reverse=True)})")
    for g_, ms in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        fams = {}
        for n in ms:
            fams[fmap[n]] = fams.get(fmap[n], 0) + 1
        print(f"  {g_:5s} {len(ms):3d}  " + ", ".join(f"{f} {c}" for f, c in sorted(fams.items(), key=lambda kv: -kv[1])))


def families(names, fmap):
    out = {}
    for i, n in enumerate(names):
        out.setdefault(fmap[n], []).append(i)
    return out


def do_counts(prefix_fallback):
    X_all, base_idx, extras, names, groups, y, t, g = load_features()
    feature_report(groups, names)
    fmap = family_map(names, prefix_fallback)
    fams = families(names, fmap)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(y.size, size=min(200_000, y.size), replace=False))
    Xs = rows(X_all, base_idx, extras, idx).astype(np.float64)
    for j in range(Xs.shape[1]):                       # ranks: immune to a few huge values
        Xs[:, j] = np.argsort(np.argsort(Xs[:, j], kind="stable"), kind="stable")
    _, inv = np.unique(t[idx], return_inverse=True); cnt = np.bincount(inv).astype(np.float64)
    for j in range(Xs.shape[1]):
        Xs[:, j] -= (np.bincount(inv, weights=Xs[:, j]) / cnt)[inv]
    print(f"\n{len(fams)} families, {len(names)} features")
    print(f"{'family':10s} {'n':>3s}  {'mean |r| within':>15s}  {'eff. dims':>9s}")
    for f, cols in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        if len(cols) == 1:
            print(f"{f:10s} {1:3d}  {'-':>15s}  {1.0:9.1f}"); continue
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.nan_to_num(np.corrcoef(Xs[:, cols].T))
        A = np.abs(C); np.fill_diagonal(A, np.nan)
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        print(f"{f:10s} {len(cols):3d}  {np.nanmean(A):15.2f}  {ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12):9.1f}")
    print("\n'eff. dims' = how many independent directions the family really has; a family with n = 20 and "
          "eff. dims = 2 is voting ~10x per direction.")


def select_topk(Xtr, ytr, wtr, names, fams, k):
    import lightgbm as lgb
    p = dict(LGB_BASE); p.update(LGB_SHIPPED); p.update(learning_rate=0.1, seed=0)
    b = lgb.train(p, lgb.Dataset(Xtr, label=ytr, weight=wtr, params={"feature_pre_filter": False}),
                  num_boost_round=150)
    gain = b.feature_importance("gain")
    keep = []
    for f, cols in fams.items():
        if f in ALWAYS_KEEP or len(cols) <= k:
            keep += cols
        else:
            keep += sorted(cols, key=lambda c: -gain[c])[:k]
    return sorted(keep)


def fit_pca(Xtr, ttr, fams, k):
    """Per-family PCA (within-t centred, standardized) fitted on training rows only."""
    _, inv = np.unique(ttr, return_inverse=True); cnt = np.bincount(inv).astype(np.float64)
    plan = []
    for f, cols in fams.items():
        if f in ALWAYS_KEEP or len(cols) <= k:
            plan.append(("raw", cols, None)); continue
        Z = Xtr[:, cols].astype(np.float64)
        mu, sd = Z.mean(axis=0), Z.std(axis=0); sd[sd == 0] = 1
        Zs = (Z - mu) / sd
        for j in range(Zs.shape[1]):
            Zs[:, j] -= (np.bincount(inv, weights=Zs[:, j]) / cnt)[inv]
        w, V = np.linalg.eigh(Zs.T @ Zs / Zs.shape[0])
        plan.append(("pca", cols, (mu, sd, V[:, np.argsort(w)[::-1][:k]])))
    return plan


def apply_pca(X, plan):
    parts = []
    for kind, cols, par in plan:
        if kind == "raw":
            parts.append(X[:, cols])
        else:
            mu, sd, V = par
            parts.append(((X[:, cols] - mu) / sd) @ V)
    return np.ascontiguousarray(np.hstack(parts), dtype=np.float32)


def do_cv(blend, fold_seed, prefix_fallback, grouping="family", thr=0.6):
    import lightgbm as lgb
    X_all, base_idx, extras, names, groups, y, t, g = load_features()
    feature_report(groups, names)
    fmap = family_map(names, prefix_fallback)
    if grouping == "property":
        gmap = property_map(names, fmap, thr)
        describe_groups(names, gmap, fmap)
        fams = families(names, gmap)
    else:
        fams = families(names, fmap)
    print("families:", {f: len(c) for f, c in fams.items()})
    stride = stride_arr(t)
    if blend:
        from catboost import CatBoostClassifier
        bc = json.load(open(f"{CACHE}/blend_choice.json"))
        CATP, CW = bc["cat_params"], float(bc["cat_weight"])
        assert abs(CW - 0.3) < 1e-9, CW
    tag = "" if grouping == "family" else f"_prop{int(thr * 100)}"
    out_path = f"{CACHE}/balance_cv{tag}{'_blend' if blend else ''}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    variants = ["full"] + [f"top{k}" for k in KS] + [f"pca{PCA_K}"]
    T0 = time.time()
    for fi, F in enumerate(build_folds(y, t, g, fold_seed)):
        Xtr_f = rows(X_all, base_idx, extras, F["tr"]); ytr = y[F["tr"]]; wtr = stride[F["tr"]].astype(np.float32)
        sel = {f"top{k}": select_topk(Xtr_f, ytr, wtr, names, fams, k) for k in KS}
        plan = fit_pca(Xtr_f, t[F["tr"]], fams, PCA_K)
        for swap in (False, True):
            key = f"{fi}_{int(swap)}"
            si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
            Xs_f, Xe_f = rows(X_all, base_idx, extras, si), rows(X_all, base_idx, extras, ei)
            for v in variants:
                if key in res.get(v, {}):
                    continue
                tf = time.time()
                if v == "full":
                    Xtr, Xs, Xe = Xtr_f, Xs_f, Xe_f
                elif v.startswith("top"):
                    c = sel[v]; Xtr, Xs, Xe = (np.ascontiguousarray(M[:, c]) for M in (Xtr_f, Xs_f, Xe_f))
                else:
                    Xtr, Xs, Xe = (apply_pca(M, plan) for M in (Xtr_f, Xs_f, Xe_f))
                dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, params={"feature_pre_filter": False})
                pl, trees = [], []
                for s in (0, 1, 2):
                    p = dict(LGB_BASE); p.update(LGB_SHIPPED); p.update(seed=s, bagging_seed=s + 100, feature_fraction_seed=s + 200)
                    b = lgb.train(p, dtr, num_boost_round=NUM_ROUNDS,
                                  valid_sets=[lgb.Dataset(Xs, label=y[si], reference=dtr)],
                                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
                    pl.append(b.predict(Xe, num_iteration=b.best_iteration)); trees.append(b.best_iteration)
                pf = np.mean(pl, axis=0)
                if blend:
                    m = CatBoostClassifier(**CATP, iterations=NUM_ROUNDS, early_stopping_rounds=PATIENCE, random_seed=0,
                                           thread_count=8, verbose=0, allow_writing_files=False)
                    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xs, y[si]))
                    pf = (1 - CW) * pf + CW * m.predict_proba(Xe)[:, 1]
                a = ts_auc(y[ei], t[ei], pf)
                res.setdefault(v, {})[key] = [a, int(np.mean(trees)), int(Xtr.shape[1])]
                json.dump(res, open(out_path, "w"))
                print(f"fold {fi} swap={swap} {v:5s} ({Xtr.shape[1]:3d} feats): {a:.5f} | trees {int(np.mean(trees))} "
                      f"| {(time.time()-tf)/60:.1f} min", flush=True)
                if v != "full":
                    del Xtr, Xs, Xe
                del dtr, pl; gc.collect()
            del Xs_f, Xe_f; gc.collect()
        del Xtr_f; gc.collect()

    keys = sorted(res["full"])
    base = np.array([res["full"][k][0] for k in keys])
    print(f"\nRESULTS ({'blend' if blend else 'LightGBM'}, fold seed {fold_seed}) | {(time.time()-T0)/60:.0f} min")
    for v in variants:
        a = np.array([res[v][k][0] for k in keys]); d = a - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if v != "full" else 0.0
        nf = int(np.mean([res[v][k][2] for k in keys])); tr = int(np.mean([res[v][k][1] for k in keys]))
        verdict = ("" if v == "full" else "BETTER" if d.mean() > 2 * se and (d > 0).sum() >= 5
                   else "no loss" if d.mean() >= -2 * se else "WORSE")
        print(f"  {v:5s} {nf:3d} feats | {a.mean():.5f} | delta {d.mean():+.5f} | se {se:.5f} "
              f"| {int((d>0).sum())}/{len(d)} | trees {tr} | {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--fold-seed", type=int, default=37)
    ap.add_argument("--groups", choices=["family", "property"], default="family")
    ap.add_argument("--thr", type=float, default=0.6, help="|r| cut for property groups")
    ap.add_argument("--show-groups", action="store_true")
    ap.add_argument("--prefix-families", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.counts:
        do_counts(a.prefix_families)
    if a.show_groups:
        X_all, base_idx, extras, names, groups, y, t, g = load_features()
        fm = family_map(names, a.prefix_families)
        describe_groups(names, property_map(names, fm, a.thr), fm)
    if a.cv:
        do_cv(a.blend, a.fold_seed, a.prefix_families, a.groups, a.thr)
    if not (a.counts or a.cv or a.show_groups):
        ap.print_help()
