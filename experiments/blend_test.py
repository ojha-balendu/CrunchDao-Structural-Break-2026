"""
blend_test.py -- shipped LightGBM vs Optuna-best LightGBM vs either + CatBoost, on a FRESH split.

    python experiments/blend_test.py --params          # print best params from both Optuna studies, then exit
    python experiments/blend_test.py                   # full test (both halves of 3 folds = 6 paired measurements)

Each model is trained ONCE per fold-half; every blend weight is then evaluated on the same stored
predictions, so testing 6 weights costs nothing extra. Blending is on probabilities, because that is
the only rule infer() can apply (it never sees the other series at the same t).
Writes cache/blend_choice_<fold-seed>.json with the chosen weight and both parameter sets
(never cache/blend_choice.json, which holds the shipped choice).
"""
import argparse, json, time, gc
import numpy as np
import optuna
from tune_overnight import (load_features, feature_report, build_folds, rows, ts_auc, stride_arr,
                            LGB_BASE, LGB_SHIPPED, NUM_ROUNDS, PATIENCE, CACHE)

WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)           # weight on CatBoost; 0.0 = LightGBM alone


def best_params(model):
    s = optuna.load_study(study_name=f"{model}109_fold11", storage=f"sqlite:///{CACHE}/optuna_{model}109.db")
    done = sorted([t for t in s.trials if t.state.name == "COMPLETE"], key=lambda t: -t.value)
    t0 = next((t.value for t in s.trials if t.number == 0 and t.value is not None), None)
    return done[0], t0, len(done)


def cat_params(p):
    p = dict(p)
    p["bootstrap_type"] = "Bernoulli"                  # subsample requires it; not stored as a trial param
    if p.get("grow_policy") != "Lossguide":
        p.pop("max_leaves", None)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", action="store_true")
    ap.add_argument("--fold-seed", type=int, default=13)
    ap.add_argument("--halves", type=int, choices=[1, 2], default=2)
    a = ap.parse_args()

    bl, l0, nl = best_params("lgb")
    bc, c0, nc = best_params("cat")
    print(f"LightGBM study: {nl} complete | shipped (trial 0) {l0:.5f} | best #{bl.number} {bl.value:.5f} "
          f"({bl.value - l0:+.5f})\n  {bl.params}")
    print(f"CatBoost study: {nc} complete | default (trial 0) {c0:.5f} | best #{bc.number} {bc.value:.5f} "
          f"({bc.value - c0:+.5f})\n  {cat_params(bc.params)}")
    if a.params:
        return

    LGBS = {"shipped": dict(LGB_SHIPPED), "best": dict(bl.params)}
    CATP = cat_params(bc.params)
    print(f"\nLightGBM variants: shipped + best #{bl.number} | CatBoost = best #{bc.number}")

    X_all, base_idx, extras, names, groups, y, t, g = load_features()
    feature_report(groups, names)
    stride = stride_arr(t)

    import lightgbm as lgb
    from catboost import CatBoostClassifier
    res = {}
    rhos, lgb_trees, cat_trees = [], [], []
    T0 = time.time()
    for fi, F in enumerate(build_folds(y, t, g, a.fold_seed)):
        Xtr = rows(X_all, base_idx, extras, F["tr"]); ytr = y[F["tr"]]; wtr = stride[F["tr"]].astype(np.float32)
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=names,
                          params={"feature_pre_filter": False}, free_raw_data=False)
        for swap in ((False, True) if a.halves == 2 else (False,)):
            tf = time.time()
            si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
            Xs, Xe = rows(X_all, base_idx, extras, si), rows(X_all, base_idx, extras, ei)
            p_l = {}
            for lab, LGBP in LGBS.items():
                pl = []
                for s_ in (0, 1, 2):
                    p = dict(LGB_BASE); p.update(LGBP); p.update(seed=s_, bagging_seed=s_ + 100, feature_fraction_seed=s_ + 200)
                    b = lgb.train(p, dtr, num_boost_round=NUM_ROUNDS,
                                  valid_sets=[lgb.Dataset(Xs, label=y[si], reference=dtr)],
                                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
                    pl.append(b.predict(Xe, num_iteration=b.best_iteration)); lgb_trees.append(b.best_iteration)
                p_l[lab] = np.mean(pl, axis=0)
            m = CatBoostClassifier(**CATP, iterations=NUM_ROUNDS, early_stopping_rounds=PATIENCE, random_seed=0,
                                   thread_count=8, verbose=0, allow_writing_files=False)
            m.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xs, y[si]))
            p_cat = m.predict_proba(Xe)[:, 1]; cat_trees.append(m.get_best_iteration())
            ye, te = y[ei], t[ei]
            def add(k, v):
                res.setdefault(k, []).append(ts_auc(ye, te, v))
            add("cat alone", p_cat)
            for lab in LGBS:
                for w in WEIGHTS:
                    add(f"lgb {lab}" if w == 0.0 else f"lgb {lab} + cat {w:.1f}", (1 - w) * p_l[lab] + w * p_cat)
            rho = np.corrcoef(np.argsort(np.argsort(p_l["best"])), np.argsort(np.argsort(p_cat)))[0, 1]
            rhos.append(rho)
            print(f"fold {fi} swap={swap}: shipped {res['lgb shipped'][-1]:.5f} | best {res['lgb best'][-1]:.5f} | "
                  f"cat {res['cat alone'][-1]:.5f} | best+cat0.3 {res['lgb best + cat 0.3'][-1]:.5f} | "
                  f"spearman {rho:.4f} | {(time.time()-tf)/60:.1f} min", flush=True)
            del Xs, Xe, p_l, p_cat, m; gc.collect()
        del Xtr, dtr; gc.collect()

    base = np.array(res["lgb shipped"])            # the model on the leaderboard today
    n = len(base)
    print(f"\nRESULTS vs the SHIPPED model ({n} paired measurements, fresh split seed {a.fold_seed}) "
          f"| mean spearman lgb_best~cat {np.mean(rhos):.4f} | mean trees lgb {int(np.mean(lgb_trees))} cat {int(np.mean(cat_trees))}")
    table = []
    for k, v in res.items():
        v = np.array(v); d = v - base
        se = d.std(ddof=1) / np.sqrt(n) if k != "lgb shipped" else 0.0
        table.append((k, v.mean(), d.mean(), se, (d.mean() / se if se else 0.0), int((d > 0).sum())))
    for k, m_, d_, se, sg, b_ in sorted(table, key=lambda r: -r[1]):
        print(f"  {k:22s} {m_:.5f} | delta {d_:+.5f} | se {se:.5f} | {sg:5.1f} sigma | {b_}/{n}")

    # decision: candidates must beat the shipped model by 2 sigma with >= 5/6 (all, with one half);
    # among them prefer the simplest config within 0.0003 of the best (shipped params, then lower cat weight)
    need = 5 if n == 6 else n
    ok = [r for r in table if r[0] != "lgb shipped" and r[0] != "cat alone" and r[2] > 2 * r[3] and r[5] >= need]
    def complexity(k):
        w = float(k.split("cat ")[1]) if "+ cat" in k else 0.0
        return (0 if "shipped" in k else 1, w)
    if ok:
        top = max(r[1] for r in ok)
        choice = min((r for r in ok if r[1] >= top - 0.0003), key=lambda r: complexity(r[0]))[0]
    else:
        choice = "lgb shipped"
    lgb_src = "best" if "best" in choice else "shipped"
    w = float(choice.split("cat ")[1]) if "+ cat" in choice else 0.0
    LGBP = LGBS[lgb_src]
    out = f"{CACHE}/blend_choice_{a.fold_seed}.json"     # the shipped cache/blend_choice.json is never overwritten
    print(f"\nVERDICT: SHIP '{choice}'  (LightGBM params = {lgb_src}, CatBoost weight = {w:.1f})")
    json.dump({"cat_weight": w, "lgb_params": LGBP, "cat_params": CATP,
               "lgb_source": lgb_src, "choice": choice, "cat_trial": bc.number,
               "lgb_mean_trees": int(np.mean(lgb_trees)), "cat_mean_trees": int(np.mean(cat_trees))},
              open(out, "w"), indent=2)
    print(f"wrote {out} | total {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()
