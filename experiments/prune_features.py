"""
prune_features.py -- correlation matrix of the production features, redundancy clusters, and a CV test
of pruned feature sets against the full set.

    python experiments/prune_features.py --corr              # correlation matrix + clusters + pruned sets (minutes)
    python experiments/prune_features.py --cv                # LightGBM (3 seeds): full vs each pruned set
    python experiments/prune_features.py --cv --blend        # the same, on the shipped LightGBM + CatBoost 0.3 blend

Correlation is computed WITHIN each time step (every column is demeaned per t first), because
TS-AUC only compares series at the same t: two features that both just rise with t look correlated
globally but may carry independent information where it matters. Rank (Spearman-style) correlation,
so heavy-tailed features don't dominate.

Within each redundancy cluster the feature with the highest univariate TS-AUC is kept.
Outputs: cache/feature_corr.npz, cache/feature_corr_pairs.csv, cache/prune_sets.json,
cache/prune_cv.json (resumable).
"""
import argparse, json, os, time, gc
import numpy as np
from tune_overnight import (load_features, feature_report, build_folds, rows, ts_auc, stride_arr,
                            LGB_BASE, LGB_SHIPPED, NUM_ROUNDS, PATIENCE, CACHE)

THRESHOLDS = (0.98, 0.95, 0.90, 0.85)
SAMPLE = 400_000


def within_t_rank_corr(Xs, ts):
    """Rank-transform each column, demean within each t, then Pearson."""
    Z = np.empty_like(Xs, dtype=np.float64)
    for j in range(Xs.shape[1]):
        Z[:, j] = np.argsort(np.argsort(Xs[:, j], kind="stable"), kind="stable")
    _, inv = np.unique(ts, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    for j in range(Z.shape[1]):
        Z[:, j] -= (np.bincount(inv, weights=Z[:, j]) / cnt)[inv]
    sd = Z.std(axis=0); sd[sd == 0] = 1.0
    Z /= sd
    return (Z.T @ Z) / Z.shape[0]


def do_corr():
    X_all, base_idx, extras, names, groups, y, t, g = load_features()
    feature_report(groups, names)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(y.size, size=min(SAMPLE, y.size), replace=False))
    Xs = rows(X_all, base_idx, extras, idx).astype(np.float64)
    ys, ts = y[idx], t[idx]
    print(f"sample {Xs.shape}; computing within-t rank correlation ...", flush=True)
    t0 = time.time()
    C = within_t_rank_corr(Xs, ts)
    print(f"  done in {time.time()-t0:.0f}s")

    auc = np.array([ts_auc(ys, ts, Xs[:, j]) for j in range(Xs.shape[1])])
    auc_abs = np.maximum(auc, 1 - auc)
    fam = {n: k for k, v in groups.items() for n in v}

    A = np.abs(C.copy()); np.fill_diagonal(A, 0)
    ev = np.linalg.eigvalsh(C)
    print(f"\npairs |r| > 0.98: {int((A > 0.98).sum() // 2)} | > 0.95: {int((A > 0.95).sum() // 2)} "
          f"| > 0.90: {int((A > 0.90).sum() // 2)} | > 0.80: {int((A > 0.80).sum() // 2)}")
    print(f"effective dimensions (participation ratio): {ev.sum() ** 2 / (ev ** 2).sum():.1f} of {len(names)}")

    iu = np.triu_indices_from(A, 1)
    order = np.argsort(-A[iu])
    with open(f"{CACHE}/feature_corr_pairs.csv", "w") as f:
        f.write("feature_a,family_a,feature_b,family_b,abs_r\n")
        for k in order[:400]:
            i, j = iu[0][k], iu[1][k]
            f.write(f"{names[i]},{fam[names[i]]},{names[j]},{fam[names[j]]},{A[i, j]:.4f}\n")
    print("\ntop 15 pairs:")
    for k in order[:15]:
        i, j = iu[0][k], iu[1][k]
        print(f"  {A[i, j]:.4f}  {names[i]:28s} ({fam[names[i]]})  ~  {names[j]:28s} ({fam[names[j]]})")

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1.0 - np.abs(C); np.fill_diagonal(D, 0); D = np.clip((D + D.T) / 2, 0, None)
    Lk = linkage(squareform(D, checks=False), method="complete")   # complete: EVERY pair in a cluster >= thr
    sets = {"full": list(names)}
    for thr in THRESHOLDS:
        lab = fcluster(Lk, t=1.0 - thr, criterion="distance")
        keep, dropped = [], []
        for c in np.unique(lab):
            mem = np.flatnonzero(lab == c)
            best = mem[np.argmax(auc_abs[mem])]
            keep.append(int(best))
            dropped += [int(m) for m in mem if m != best]
        keep.sort()
        sets[f"r{int(thr * 100)}"] = [names[i] for i in keep]
        print(f"\nthreshold |r| >= {thr}: {len(keep)} kept, {len(dropped)} dropped")
        per_fam = {}
        for i in dropped:
            per_fam[fam[names[i]]] = per_fam.get(fam[names[i]], 0) + 1
        print("   dropped by family:", dict(sorted(per_fam.items(), key=lambda kv: -kv[1])))
    np.savez(f"{CACHE}/feature_corr.npz", C=C, names=np.array(names), auc=auc_abs)
    json.dump(sets, open(f"{CACHE}/prune_sets.json", "w"), indent=1)
    print(f"\nwrote {CACHE}/feature_corr.npz, feature_corr_pairs.csv, prune_sets.json")


def do_cv(blend, fold_seed):
    import lightgbm as lgb
    sets = json.load(open(f"{CACHE}/prune_sets.json"))
    X_all, base_idx, extras, names, groups, y, t, g = load_features()
    feature_report(groups, names)
    assert sets["full"] == names, "prune_sets.json was built on a different feature set -- rerun --corr"
    cols = {k: [names.index(n) for n in v] for k, v in sets.items()}
    seen = {}
    for k in list(cols):                      # a threshold that drops nothing is the full set again
        key = tuple(cols[k])
        if key in seen:
            print(f"  {k} is identical to {seen[key]} -- skipped")
            del cols[k]
        else:
            seen[key] = k
    stride = stride_arr(t)
    if blend:
        from catboost import CatBoostClassifier
        bc = json.load(open(f"{CACHE}/blend_choice.json"))
        CATP, CW = bc["cat_params"], float(bc["cat_weight"])
        assert abs(CW - 0.3) < 1e-9, CW
    out_path = f"{CACHE}/prune_cv{'_blend' if blend else ''}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    T0 = time.time()
    for fi, F in enumerate(build_folds(y, t, g, fold_seed)):
        Xtr_full = rows(X_all, base_idx, extras, F["tr"])
        for swap in (False, True):
            key = f"{fi}_{int(swap)}"
            si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
            Xs_full = rows(X_all, base_idx, extras, si); Xe_full = rows(X_all, base_idx, extras, ei)
            for name, c in cols.items():
                if key in res.get(name, {}):
                    continue
                tf = time.time()
                Xtr, Xs, Xe = (np.ascontiguousarray(M[:, c]) for M in (Xtr_full, Xs_full, Xe_full))
                nm = [names[i] for i in c]
                dtr = lgb.Dataset(Xtr, label=y[F["tr"]], weight=stride[F["tr"]].astype(np.float32),
                                  feature_name=nm, params={"feature_pre_filter": False})
                pl, trees = [], []
                for s in (0, 1, 2):
                    p = dict(LGB_BASE); p.update(LGB_SHIPPED); p.update(seed=s, bagging_seed=s + 100, feature_fraction_seed=s + 200)
                    b = lgb.train(p, dtr, num_boost_round=NUM_ROUNDS,
                                  valid_sets=[lgb.Dataset(Xs, label=y[si], reference=dtr)],
                                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
                    pl.append(b.predict(Xe, num_iteration=b.best_iteration)); trees.append(b.best_iteration)
                p_fin = np.mean(pl, axis=0)
                if blend:
                    m = CatBoostClassifier(**CATP, iterations=NUM_ROUNDS, early_stopping_rounds=PATIENCE, random_seed=0,
                                           thread_count=8, verbose=0, allow_writing_files=False)
                    m.fit(Xtr, y[F["tr"]], sample_weight=stride[F["tr"]].astype(np.float32), eval_set=(Xs, y[si]))
                    p_fin = (1 - CW) * p_fin + CW * m.predict_proba(Xe)[:, 1]
                a = ts_auc(y[ei], t[ei], p_fin)
                res.setdefault(name, {})[key] = [a, int(np.mean(trees)), len(c)]
                json.dump(res, open(out_path, "w"))
                print(f"fold {fi} swap={swap} {name:6s} ({len(c):3d} feats): {a:.5f} | trees {int(np.mean(trees))} "
                      f"| {(time.time()-tf)/60:.1f} min", flush=True)
                del Xtr, Xs, Xe, dtr, pl; gc.collect()
            del Xs_full, Xe_full; gc.collect()
        del Xtr_full; gc.collect()

    keys = sorted(res["full"])
    base = np.array([res["full"][k][0] for k in keys])
    print(f"\nRESULTS ({'blend' if blend else 'LightGBM'}, fold seed {fold_seed}, {len(keys)} halves) | {(time.time()-T0)/60:.0f} min")
    for name in cols:
        v = np.array([res[name][k][0] for k in keys]); d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if name != "full" else 0.0
        tr = int(np.mean([res[name][k][1] for k in keys]))
        verdict = ("" if name == "full" else
                   "BETTER" if d.mean() > 2 * se and (d > 0).sum() >= 5 else
                   "no loss" if d.mean() >= -2 * se else "WORSE")
        print(f"  {name:6s} {len(cols[name]):3d} feats | {v.mean():.5f} | delta {d.mean():+.5f} | se {se:.5f} "
              f"| {int((d>0).sum())}/{len(d)} | trees {tr} | {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--fold-seed", type=int, default=31)
    a = ap.parse_args()
    if a.corr:
        do_corr()
    if a.cv:
        do_cv(a.blend, a.fold_seed)
    if not (a.corr or a.cv):
        ap.print_help()
