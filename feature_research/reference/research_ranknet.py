# ===================== RANKING OBJECTIVE FOR THE MAIN MODEL -- LightGBM trained on TS-AUC's own pairs =====================
# Paste after the BLIND-SPOT SCAN cell (uses the base's cached out-of-fold predictions). WITH_CATBOOST = False.
#
# Your LightGBM today minimizes logloss: "get each row's probability right". TS-AUC only asks "at each step, is every
# broken series above every not-broken one". This trains the same full-bank LightGBM on exactly that question, with a
# custom objective (RankNet / pairwise logistic loss):
#   loss = sum over (broken i, not-broken j) at the SAME step t of  w_ij * log(1 + exp(-(s_i - s_j)))
# * pairs: each broken row is paired with PAIRS_K random not-broken rows at its step, RE-SAMPLED EVERY ITERATION
#   (cheap, and acts like extra data: across 1,000+ trees the model sees many different pairs)
# * weights: stride(t) * n_neg(t) / PAIRS_K per pair, so every step counts in proportion to its number of pairs --
#   exactly TS-AUC's weights (a kept step stands for `stride` original steps)
# * early stopping on TS-AUC itself (the stop half, training-stride rows), not on a loss
# The raw score is a ranking score; for infer() any monotone map works (sigmoid; or an affine fit then sigmoid when
# blending with a probability model) -- TS-AUC is unchanged by it.
# Reported: ranker alone vs the logloss base, and a 50/50 average of the two (standardized) -- if that beats both,
# the ranker makes DIFFERENT errors and is worth keeping as a blend member even if it does not win alone.
PAIRS_K = 8
RANK_SEEDS = SEEDS
EVAL_EVERY = 10                     # compute the TS-AUC early-stopping metric every N iterations
RN_PATH = f"{CACHE}/ranknet_{BASE_LABEL}_s{FOLD_SEED}_k{PAIRS_K}.json"


class _PairSampler:
    """Same-step (broken, not-broken) pairs for a fixed row set, re-sampled on demand."""
    def __init__(self, t, y, stride, k, seed):
        self.n = t.size
        self.rng = np.random.default_rng(seed)
        neg = np.flatnonzero(y == 0)
        neg = neg[np.argsort(t[neg], kind="stable")]
        tn = t[neg]
        pos = np.flatnonzero(y == 1)
        lo = np.searchsorted(tn, t[pos], side="left")
        hi = np.searchsorted(tn, t[pos], side="right")
        ok = hi > lo
        self.pos, self.lo, self.cnt = np.repeat(pos[ok], k), np.repeat(lo[ok], k), np.repeat((hi - lo)[ok], k)
        self.neg = neg
        w = stride[pos[ok]].astype(np.float64) * (hi - lo)[ok] / k
        self.w = np.repeat(w / w.mean(), k)

    def sample(self):
        j = self.neg[self.lo + (self.rng.random(self.pos.size) * self.cnt).astype(np.int64)]
        return self.pos, j, self.w


def _ranknet_objective(sampler):
    def fobj(preds, data):
        i, j, w = sampler.sample()
        d = preds[i] - preds[j]
        sg = 1.0 / (1.0 + np.exp(d))                      # sigma(-d)
        h = w * sg * (1.0 - sg)
        grad = np.bincount(i, weights=-w * sg, minlength=sampler.n) + np.bincount(j, weights=w * sg, minlength=sampler.n)
        hess = np.bincount(i, weights=h, minlength=sampler.n) + np.bincount(j, weights=h, minlength=sampler.n)
        return grad, np.maximum(hess, 1e-6)
    return fobj


def _tsauc_feval(y, t):
    state = {"it": 0, "last": 0.5}
    def feval(preds, data):
        if state["it"] % EVAL_EVERY == 0:
            state["last"] = float(ts_auc(y, t, preds))
        state["it"] += 1
        return "ts_auc", state["last"], True
    return feval


def _zs(x, ref):
    return (x - ref.mean()) / (ref.std() + 1e-12)


prod_idx, blocks, names = selection()
RN = json.load(open(RN_PATH)) if os.path.exists(RN_PATH) else {}
for fi, swap in HALVES:
    key = f"f{fi}{'b' if swap else 'a'}"
    RN_PRED = f"{CACHE}/ranknet_pred_{BASE_LABEL}_s{FOLD_SEED}_k{PAIRS_K}_{key}.npz"
    if key in RN and os.path.exists(RN_PRED):
        continue
    t0 = time.time()
    F = FOLDS[fi]
    si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
    tr = F["tr"]
    Xtr, Xe = hstack_rows(tr, prod_idx, blocks), hstack_rows(ei, prod_idx, blocks)
    ks = (t_all[si] % STRIDE_ALL[si]) == 0
    rs = si[ks]
    Xs = hstack_rows(rs, prod_idx, blocks)
    ytr, ttr = y_all[tr], t_all[tr]
    scores_s, scores_e, trees = [], [], []
    for s in RANK_SEEDS:
        p = _lgb_params(None, s)
        p.pop("metric", None)
        sampler = _PairSampler(ttr, ytr, STRIDE_ALL[tr], PAIRS_K, seed=1000 + s)
        p["objective"] = _ranknet_objective(sampler)
        p["metric"] = "None"
        dtr = lgb.Dataset(Xtr, label=ytr, feature_name=names)
        dva = lgb.Dataset(Xs, label=y_all[rs], reference=dtr)
        b = lgb.train(p, dtr, num_boost_round=SMOKE_ROUNDS if SMOKE else NUM_ROUNDS, valid_sets=[dva],
                      feval=_tsauc_feval(y_all[rs], t_all[rs]),
                      callbacks=[lgb.early_stopping(max(PATIENCE, 3 * EVAL_EVERY), verbose=False)])
        scores_s.append(b.predict(Xs, num_iteration=b.best_iteration))
        scores_e.append(b.predict(Xe, num_iteration=b.best_iteration))
        trees.append(int(b.best_iteration))
        del dtr, dva, b
    del Xtr, Xs, Xe
    gc.collect()
    r_s, r_e = np.mean(scores_s, axis=0), np.mean(scores_e, axis=0)
    np.savez_compressed(RN_PRED, r_s=r_s, r_e=r_e, rows_s=rs)       # for blending experiments without retraining
    Z = np.load(f"{CACHE}/oof_{BASE_LABEL}_s{FOLD_SEED}_{key}.npz")
    b_e = _logit(Z["p_e"]) if "_logit" in globals() else np.log(np.clip(Z["p_e"], 1e-6, 1) / np.clip(1 - Z["p_e"], 1e-6, 1))
    b_s = np.log(np.clip(Z["p_s"], 1e-6, 1) / np.clip(1 - Z["p_s"], 1e-6, 1))[ks]
    avg = 0.5 * _zs(r_e, r_s) + 0.5 * _zs(b_e, b_s)          # standardized with stop-half statistics
    ye, te = y_all[ei], t_all[ei]
    RN[key] = dict(base=by_bucket(ye, te, b_e), ranker=by_bucket(ye, te, r_e), avg=by_bucket(ye, te, avg),
                   trees=trees, corr=float(pd.Series(r_e).corr(pd.Series(b_e), method="spearman")))
    RN[key] = {k: ({kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in RN[key].items()}
    with open(RN_PATH + ".tmp", "w") as fh:
        json.dump(RN, fh, indent=1)
    os.replace(RN_PATH + ".tmp", RN_PATH)
    print(f"  [ranknet] {key}: base {RN[key]['base']['all']:.4f} | ranker {RN[key]['ranker']['all']:.4f} | "
          f"50/50 {RN[key]['avg']['all']:.4f} | trees {trees} | rank corr with base {RN[key]['corr']:.3f} | "
          f"{(time.time() - t0) / 60:.1f} min")

keys = [f"f{fi}{'b' if sw else 'a'}" for fi, sw in HALVES]
print(f"\n=== RANKING OBJECTIVE (pairwise within step) vs logloss | {BASE_LABEL} | {PAIRS_K} pairs per broken row")
for cand in ("ranker", "avg"):
    parts = []
    for b_ in ["all"] + BUCKET_KEYS:
        d = np.array([RN[k][cand][b_] - RN[k]["base"][b_] for k in keys], dtype=np.float64)
        d = d[np.isfinite(d)]
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(d.size)
        parts.append(f"{b_} {m:+.5f} ({m / se if se else 0:+.1f}s, {(d > 0).sum()}/{d.size})")
        if b_ == "all":
            ok = m > 2 * se and (d > 0).sum() >= d.size - 1
    print(f"  {cand:7s}: " + " | ".join(parts) + f"  -> {'ADOPT' if ok else 'REJECT'}")
print(f"  mean rank correlation ranker vs base: {np.mean([RN[k]['corr'] for k in keys]):.3f} "
      f"(lower = more different errors = more useful as a blend member)")


# ---------------- time-varying blend: how much ranker vs logloss model, per stage of the online segment ----------------
# The ranker wins early and loses late, so one fixed 50/50 weight wastes both. For each time bucket (log2 steps), the
# weight on the ranker is chosen on the STOP half (grid 0..1, both scores standardized per bucket on the stop half),
# then applied to the untouched SCORE half. Also: the rank correlation WITHIN each step (the pooled one above is
# inflated by the logloss model's rise with t, which the ranker does not have).
TB_EDGES = (0, 16, 32, 64, 128, 256, 512, 10 ** 9)
W_GRID = np.round(np.linspace(0, 1, 11), 2)


def _tbin(t):
    return np.digitize(t, TB_EDGES[1:-1])


out, wt_log, wcorr = [], [], []
for fi, swap in HALVES:
    key = f"f{fi}{'b' if swap else 'a'}"
    F = FOLDS[fi]
    si, ei = (F["sc"], F["st"]) if swap else (F["st"], F["sc"])
    P = np.load(f"{CACHE}/ranknet_pred_{BASE_LABEL}_s{FOLD_SEED}_k{PAIRS_K}_{key}.npz")
    Z = np.load(f"{CACHE}/oof_{BASE_LABEL}_s{FOLD_SEED}_{key}.npz")
    rs = P["rows_s"]
    ks = (t_all[si] % STRIDE_ALL[si]) == 0
    b_s = np.log(np.clip(Z["p_s"], 1e-6, 1) / np.clip(1 - Z["p_s"], 1e-6, 1))[ks]
    b_e = np.log(np.clip(Z["p_e"], 1e-6, 1) / np.clip(1 - Z["p_e"], 1e-6, 1))
    r_s, r_e = P["r_s"], P["r_e"]
    ts_, ys = t_all[rs], y_all[rs]
    te, ye = t_all[ei], y_all[ei]
    bs, be = _tbin(ts_), _tbin(te)
    blend = np.zeros(ei.size)
    ws = [np.nan] * (len(TB_EDGES) - 1)
    for b in range(len(TB_EDGES) - 1):
        ms, me = bs == b, be == b
        if not me.any():
            continue
        if ms.sum() < 50 or ys[ms].sum() < 5 or (1 - ys[ms]).sum() < 5:
            w = 0.5
        else:
            zr = lambda x, ref: (x - ref.mean()) / (ref.std() + 1e-12)
            sc = [ts_auc(ys[ms], ts_[ms], w_ * zr(r_s[ms], r_s[ms]) + (1 - w_) * zr(b_s[ms], b_s[ms])) for w_ in W_GRID]
            w = float(W_GRID[int(np.argmax(sc))])
        ws[b] = w
        blend[me] = w * (r_e[me] - r_s[ms].mean()) / (r_s[ms].std() + 1e-12) + \
                    (1 - w) * (b_e[me] - b_s[ms].mean()) / (b_s[ms].std() + 1e-12) if ms.any() else r_e[me]
    wt_log.append(ws)
    out.append({"base": by_bucket(ye, te, b_e), "tv_blend": by_bucket(ye, te, blend)})
    cs = []
    for tt in np.unique(te)[::5]:
        m = te == tt
        if m.sum() > (5 if SMOKE else 50):
            cs.append(pd.Series(r_e[m]).corr(pd.Series(b_e[m]), method="spearman"))
    wcorr.append(np.nanmean(cs))
print(f"\n=== TIME-VARYING BLEND ranker / logloss (weights picked on the stop half per time bucket)")
print("  ranker weight per bucket " + str([f"{a}-{b - 1}" if b < 10 ** 9 else f"{a}+" for a, b in zip(TB_EDGES[:-1], TB_EDGES[1:])])
      + ": mean " + str(np.round(np.nanmean(np.array(wt_log, dtype=float), axis=0), 2).tolist()))
for b_ in ["all"] + BUCKET_KEYS:
    d = np.array([o["tv_blend"][b_] - o["base"][b_] for o in out], dtype=np.float64)
    d = d[np.isfinite(d)]
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(d.size)
    print(f"  tv_blend vs base {b_:9s}: {m:+.5f} | se {se:.5f} | {m / se if se else 0:+.1f} sigma | better {(d > 0).sum()}/{d.size}")
d = np.array([o["tv_blend"]["all"] - o["base"]["all"] for o in out])
print("  VERDICT:", "ADOPT" if (d.mean() > 2 * d.std(ddof=1) / np.sqrt(d.size) and (d > 0).sum() >= d.size - 1) else "REJECT")
print(f"  rank correlation ranker vs base WITHIN a step (the fair diversity measure): {np.mean(wcorr):.3f}")
