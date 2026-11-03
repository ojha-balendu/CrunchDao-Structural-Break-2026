# ===================== B1: LEARNED NULL MODEL -- "how does THIS kind of series move when nothing happens?" =====================
# Paste after the BLIND-SPOT SCAN cell. Needs PyTorch (`python -m pip install torch`). WITH_CATBOOST = False.
#
# Every scale feature you have judges online data against a hand-written null model (AR(3) + rank normalization,
# GARCH, a 2-state HMM). The false-alarm types from Step 0 (heavy tails, clustered or drifting volatility) are exactly
# where those are wrong. Here a small neural network LEARNS the null from ~30M break-free history points:
#   * input: the series in the engine's `z` form (each value's rank in its own history -> normal score)
#   * a 2-layer GRU reads the past and outputs a 5-component Gaussian mixture for the NEXT value
#   * trained only on HISTORIES (guaranteed break-free) to maximize the likelihood of what comes next
# Online, three per-step "surprise" streams under that learned model:
#   nll  -log p(z_t | past)              how unlikely the new value is
#   g    normal score of its predictive CDF position (signed: above / below what was expected)
#   g2   g^2 - 1                         scale surprise in the model's own units
# each standardized by the SAME stream over the series' own history (volser principle), then aggregated like your
# other streams: w32 / w128 / w512 / ew / exp (window mean x sqrt(size)), plus CUSUM up / down on g2. = 17 features.
#
# Cross-fitting (no series is ever scored by a model that saw its history): one model per CV fold, trained on the
# histories of the OTHER two folds' series, and used to compute the features of its own fold's series. Histories
# carry no labels, so this is leak-free; it also mirrors production, where test series are unseen.
# Then one paired CV: base vs base + these 17 features.
import torch
import torch.nn as nn
from scipy.special import ndtri as _nm_ndtri
from scipy.signal import lfilter as _nm_lfilter

NM_H, NM_LAYERS, NM_K = 64, 2, 5
NM_WIN, NM_BURN = 256, 32                 # training window length; first steps of each window not scored
NM_STEPS = 60 if SMOKE else 2500          # optimizer steps per fold model
NM_BATCH, NM_LR = 256, 2e-3
NM_WARM = 1024                            # history steps run through the GRU before the online segment
NM_PATH = f"{CACHE}/nullmodel_feats_{BASE_LABEL}.npz"
torch.set_num_threads(8)
NM_DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)


class _NullGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(3, NM_H)
        self.gru = nn.GRU(NM_H, NM_H, num_layers=NM_LAYERS, batch_first=True)
        self.out = nn.Linear(NM_H, 3 * NM_K)

    def forward(self, z):                                    # z: (B, T) -> mixture params for z_{t+1} at each t
        x = torch.stack([z, z.abs(), z * z], dim=-1)
        h, _ = self.gru(torch.relu(self.inp(x)))
        lg, mu, ls = self.out(h).chunk(3, dim=-1)
        return torch.log_softmax(lg, dim=-1), mu, ls.clamp(-4.0, 2.0)


def _mdn_nll(lw, mu, ls, y):
    y = y.unsqueeze(-1)
    lp = -0.5 * ((y - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(lw + lp, dim=-1)


def _mdn_cdf(lw, mu, ls, y):
    y = y.unsqueeze(-1)
    return (lw.exp() * 0.5 * (1 + torch.erf((y - mu) / (ls.exp() * math.sqrt(2))))).sum(-1)


if os.path.exists(NM_PATH):
    _Z = np.load(NM_PATH)
    NM_X, NM_NAMES = _Z["X"], [str(s) for s in _Z["names"]]
    print(f"[null model] loaded features {NM_X.shape}")
else:
    t_all0 = time.time()
    # ---- every series in z form, and which CV fold it belongs to ----
    GU, GI = np.unique(g_all, return_inverse=True)
    fold_of = np.full(GU.size, -1)
    for f, F in enumerate(FOLDS):
        fold_of[np.unique(GI[np.r_[F["st"], F["sc"]]])] = f
    ZH, ZO = {}, {}
    for did, xh, xo, _tau in get_train_data():
        k = int(np.searchsorted(GU, did))
        if k >= GU.size or GU[k] != did:
            continue
        xh = np.asarray(xh, dtype=np.float64)
        xh = xh[np.isfinite(xh)]
        xo = np.asarray(xo, dtype=np.float64).copy()
        xo[~np.isfinite(xo)] = xh.mean()
        srt, n = np.sort(xh), xh.size
        zf = lambda v: _nm_ndtri((np.searchsorted(srt, v, side="left") + 0.5) / (n + 1.0)).astype(np.float32)
        ZH[k], ZO[k] = zf(xh), zf(xo)
    print(f"[null model] {len(ZH):,} series in z form | device {NM_DEV}")

    NAMES = [f"nm_{s}_{a}" for s in ("nll", "g", "g2") for a in ("w32", "w128", "w512", "ew", "exp")] + \
            ["nm_g2_cusup", "nm_g2_cusdn"]
    NM_X = np.zeros((y_all.size, len(NAMES)), dtype=np.float32)

    def _aggregate(s):
        """window mean x sqrt(size) for 32/128/512, EW (alpha .02) x sqrt(min(n,100)), expanding mean x sqrt(n)."""
        n = np.arange(1, s.size + 1, dtype=np.float64)
        cs = np.r_[0.0, np.cumsum(s)]
        cols = []
        for W in (32, 128, 512):
            m = np.minimum(n, W)
            idx = np.arange(1, s.size + 1)
            cols.append((cs[idx] - cs[np.maximum(idx - W, 0)]) / m * np.sqrt(m))
        ew = _nm_lfilter([0.02], [1, -0.98], s)
        ew = ew / (1 - 0.98 ** n)
        cols.append(ew * np.sqrt(np.minimum(n, 100)))
        cols.append(cs[1:] / n * np.sqrt(n))
        return cols

    def _cusum(s, drift=0.5):
        up = np.zeros(s.size)
        dn = np.zeros(s.size)
        cu = cd = 0.0
        for i, v in enumerate(s):
            cu = max(0.0, cu + v - drift)
            cd = max(0.0, cd - v - drift)
            up[i], dn[i] = cu, cd
        return up, dn

    for f in range(len(FOLDS)):
        t0 = time.time()
        train_ks = [k for k in ZH if fold_of[k] != f]
        own_ks = [k for k in ZH if fold_of[k] == f]
        rng = np.random.default_rng(10 + f)
        val_ks = set(rng.choice(train_ks, max(len(train_ks) // 20, 1), replace=False).tolist())
        fit_ks = [k for k in train_ks if k not in val_ks]
        L = NM_WIN + 1

        def _batch(ks, B):
            sel = rng.choice(len(ks), B)
            out = np.empty((B, L), dtype=np.float32)
            for i, j in enumerate(sel):
                z = ZH[ks[j]]
                a = int(rng.integers(0, z.size - L)) if z.size > L else 0
                w = z[a:a + L]
                out[i, :w.size] = w
                out[i, w.size:] = 0.0
            return torch.from_numpy(out).to(NM_DEV)

        model = _NullGRU().to(NM_DEV)
        opt = torch.optim.Adam(model.parameters(), lr=NM_LR)
        vb = _batch(list(val_ks), 512)
        best, best_state = float("inf"), None
        for step in range(NM_STEPS):
            xb = _batch(fit_ks, NM_BATCH)
            lw, mu, ls = model(xb[:, :-1])
            loss = _mdn_nll(lw, mu, ls, xb[:, 1:])[:, NM_BURN:].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 99 or step == NM_STEPS - 1:
                with torch.no_grad():
                    vl = float(_mdn_nll(*model(vb[:, :-1]), vb[:, 1:])[:, NM_BURN:].mean())
                if vl < best:
                    best, best_state = vl, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        torch.save(best_state, f"{CACHE}/nullmodel_fold{f}_{BASE_LABEL}.pt")
        print(f"  [null model] fold {f}: trained on {len(fit_ks):,} histories | best validation NLL {best:.4f} "
              f"(a standard normal would be 1.419) | {(time.time() - t0) / 60:.1f} min")

        # ---- features for this fold's own series (never seen by this model) ----
        t1 = time.time()
        own_ks.sort(key=lambda k: min(ZH[k].size, NM_WARM) + ZO[k].size)
        for a in range(0, len(own_ks), 64):
            ks = own_ks[a:a + 64]
            seqs = [np.r_[ZH[k][-NM_WARM:], ZO[k]] for k in ks]
            T = max(s.size for s in seqs)
            arr = np.zeros((len(ks), T), dtype=np.float32)
            for i, s in enumerate(seqs):
                arr[i, :s.size] = s
            xb = torch.from_numpy(arr).to(NM_DEV)
            with torch.no_grad():
                lw, mu, ls = model(xb[:, :-1])
                y = xb[:, 1:]
                nll = _mdn_nll(lw, mu, ls, y).cpu().numpy()
                cdf = _mdn_cdf(lw, mu, ls, y).clamp(1e-6, 1 - 1e-6).cpu().numpy()
            g = _nm_ndtri(cdf)
            for i, k in enumerate(ks):
                wh = min(ZH[k].size, NM_WARM)
                n_on = ZO[k].size
                # position j of nll/g predicts element j+1 of the sequence; history targets are 1..wh-1
                hs = slice(NM_BURN, wh - 1)
                os_ = slice(wh - 1, wh - 1 + n_on)
                streams = {"nll": nll[i], "g": g[i], "g2": g[i] ** 2 - 1.0}
                cols = []
                g2_std = None
                for nm_ in ("nll", "g", "g2"):
                    s_h, s_o = streams[nm_][hs], streams[nm_][os_]
                    mu_h, sd_h = float(s_h.mean()), float(s_h.std()) + 1e-6
                    so = (s_o - mu_h) / sd_h
                    cols += _aggregate(so)
                    if nm_ == "g2":
                        g2_std, g2_hist = so, (s_h - mu_h) / sd_h
                up, dn = _cusum(g2_std)
                hu, hd = _cusum(g2_hist)
                cols += [up / max(np.quantile(hu, 0.95), 1.0), dn / max(np.quantile(hd, 0.95), 1.0)]
                rows = np.flatnonzero(GI == k)
                assert rows.size == n_on and np.array_equal(t_all[rows], np.arange(n_on)), "cache rows not contiguous"
                NM_X[rows] = np.column_stack(cols).astype(np.float32)
        print(f"  [null model] fold {f}: features for {len(own_ks):,} series | {(time.time() - t1) / 60:.1f} min")
        del model
    NM_X = np.nan_to_num(NM_X, nan=0.0, posinf=0.0, neginf=0.0)
    NM_NAMES = NAMES
    np.savez(NM_PATH, X=NM_X, names=np.array(NM_NAMES))
    print(f"[null model] all features in {(time.time() - t_all0) / 60:.1f} min")

# ---------------- univariate check, then the paired CV ----------------
print("\n  univariate TS-AUC (all rows):  " + ", ".join(
    f"{n} {max(a, 1 - a):.4f}" for n, a in sorted(((n, ts_auc(y_all, t_all, NM_X[:, j].astype(np.float64)))
                                                  for j, n in enumerate(NM_NAMES)), key=lambda x: -max(x[1], 1 - x[1]))[:6]))
_nb = run_cv(BASE_LABEL)
_nc = run_cv(f"{BASE_LABEL}+nullmodel", add=[(NM_X, NM_NAMES)])
_c = compare(_nc, _nb)
print(f"\n=== LEARNED NULL MODEL vs {BASE_LABEL}")
for k_, p_ in _c.items():
    print(f"  {k_:9s}: {p_['delta']:+.5f} | se {p_['se']:.5f} | {p_['sigma']:+.1f} sigma | better {p_['better']}/{p_['n']}")
print("  VERDICT:", "ADOPT" if add_ok(_nc, _nb) else "REJECT")
