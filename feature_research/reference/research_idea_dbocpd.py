# ######################################################################
# #                          I D E A                                   #
# ######################################################################
#
# dbocpd -- the shipped BOCPD, but with its two priors FITTED TO THIS DATASET instead of generic guesses.
#
# The shipped bocpd asks "has the variance of e changed at some step k <= t?" with
#   * a CONSTANT hazard (1/500 per step), and
#   * a generic Inverse-Gamma(2, 1) prior on the post-break variance.
# What the data actually says (P1 of the data-structure probe): the online length N is independent of everything,
# breaks happen in ~50% of series, and tau is UNIFORM over the online segment. So for a series still running at
# step t, every break position k <= t is equally likely a priori, with probability
#       q_t = P(breaks) * E[1/N | N > t]            (and "no break yet" has probability 1 - (t+1) q_t)
# -- estimated from the training online lengths. It is not constant: it changes with t exactly as the design implies.
# The post-break variance prior becomes a MIXTURE over a grid of variance ratios r, weighted by how often each size of
# break actually occurs in the training series (post-break variance of e vs history, measured in hindsight).
#
# Same channel as the shipped family (e: AR-whitened, rank-normalized residual, N(0,1) with no break), exact Bayesian
# posterior over (break position k, break size r), O(t x |r grid|) per step, vectorized.
# Features (5):  dbocpd_logodds      posterior log-odds "broken by t" vs "not yet"
#                dbocpd_logodds_up   the same, counting only variance INCREASES
#                dbocpd_logodds_down the same, only DECREASES
#                dbocpd_age_log      log posterior mean age of the break
#                dbocpd_logratio     posterior mean log variance ratio
from scipy.special import ndtri as _db_ndtri, logsumexp as _db_lse

DB_GRID_R = (0.35, 0.5, 0.65, 0.8, 0.9, 1.1, 1.25, 1.5, 1.8, 2.2, 2.8, 3.6)
DB_MIN_POST = 32

# ---------------- fit the two priors from the training data ----------------
_Ns, _brk, _logratios = [], [], []
for did, xh, xo, tau in get_train_data():
    xo = np.asarray(xo, dtype=np.float64)
    _Ns.append(xo.size)
    _brk.append(tau is not None)
    if tau is None or xo.size - tau < DB_MIN_POST:
        continue
    xh = np.asarray(xh, dtype=np.float64)
    xh = xh[np.isfinite(xh)]
    xo[~np.isfinite(xo)] = xh.mean()
    mu = xh.mean()
    c = np.concatenate([xh, xo]) - mu
    x = c[:xh.size]
    d = float(x @ x)
    r_ = np.array([float(x[k:] @ x[:-k]) / d for k in (1, 2, 3)]) if d > 0 else np.zeros(3)
    R_ = np.array([[1, r_[0], r_[1]], [r_[0], 1, r_[0]], [r_[1], r_[0], 1]])
    try:
        a_ = np.linalg.solve(R_, r_)
    except np.linalg.LinAlgError:
        a_ = np.zeros(3)
    res = c[3:] - a_[0] * c[2:-1] - a_[1] * c[1:-2] - a_[2] * c[:-3]
    rh, rp = res[:xh.size - 3], res[xh.size - 3 + tau:]
    srt = np.sort(rh)
    e_post = _db_ndtri((np.searchsorted(srt, rp, side="left") + 0.5) / (srt.size + 1.0))
    _logratios.append(math.log(max(float(np.mean(e_post ** 2)), 1e-6)))
_Ns, _brk = np.asarray(_Ns), np.asarray(_brk)
_pi = float(_brk.mean())
DB_Q = np.array([_pi * float(np.mean(1.0 / _Ns[_Ns > t])) if (_Ns > t).any() else 0.0 for t in range(1000)])
DB_Q = np.minimum(DB_Q, 0.999 / np.arange(1, 1001))          # keep 1 - (t+1) q_t > 0
_lr = np.asarray(_logratios)
_gl = np.log(np.asarray(DB_GRID_R))
_idx = np.argmin(np.abs(_lr[:, None] - _gl[None, :]), axis=1)
_cnt = np.bincount(_idx, minlength=_gl.size) + 1.0            # +1: every size keeps some prior mass
DB_LOGW = np.log(_cnt / _cnt.sum())
print(f"[dbocpd] P(break) {_pi:.3f} | hazard per step q_t at t=0/100/300/600: "
      + " / ".join(f"{DB_Q[t]:.5f}" for t in (0, 100, 300, 600)) + "  (shipped: constant 0.00200)")
print("[dbocpd] break-size prior (variance ratio: weight): "
      + ", ".join(f"{r}: {w:.2f}" for r, w in zip(DB_GRID_R, np.exp(DB_LOGW))))


class DatasetBOCPDFamily(FeatureFamily):
    name = "dbocpd"
    version = 1
    cost = "O1"
    channels = ("e",)
    Q = DB_Q
    LOGR = np.log(np.asarray(DB_GRID_R))
    LOGW = DB_LOGW

    def names(self):
        return ["dbocpd_logodds", "dbocpd_logodds_up", "dbocpd_logodds_down", "dbocpd_age_log", "dbocpd_logratio"]

    def init(self, hist):
        self.cap = 1024
        self.S = np.zeros((self.cap, self.LOGR.size))           # cumulative log-likelihood ratio per (k, r)
        self.k = 0
        self.inv_r = np.exp(-self.LOGR)
        self.half_logr = 0.5 * self.LOGR
        self.up = self.LOGR > 0

    def update(self, ctx):
        x = float(ctx.vals["e"])
        x2 = x * x
        k = self.k
        if k + 1 >= self.cap:                                    # online segments are < 1000 steps; guard anyway
            self.S = np.vstack([self.S, np.zeros_like(self.S)])
            self.cap = self.S.shape[0]
        # log N(x; 0, r) - log N(x; 0, 1) for every break size r, added to every hypothesis k <= t
        ll = -self.half_logr - 0.5 * x2 * self.inv_r + 0.5 * x2
        self.S[:k + 1] += ll
        self.k = k = k + 1
        t = k - 1
        q = float(self.Q[min(t, self.Q.size - 1)])
        log_null = math.log(max(1.0 - (t + 1) * q, 1e-12))
        J = self.S[:k] + self.LOGW                               # (k, r): log w_r + LLR
        Ek = _db_lse(J, axis=1)                                  # evidence for "break at position k"
        lq = math.log(max(q, 1e-300))
        logodds = float(_db_lse(Ek)) + lq - log_null
        lo_up = float(_db_lse(J[:, self.up])) + lq - log_null
        lo_dn = float(_db_lse(J[:, ~self.up])) + lq - log_null
        pk = np.exp(Ek - Ek.max())
        pk /= pk.sum()
        age = float(pk @ np.arange(k - 1, -1, -1, dtype=np.float64))
        pr = np.exp(J - J.max())
        pr /= pr.sum()
        logratio = float((pr * self.LOGR).sum())
        return [_safe(logodds), _safe(lo_up), _safe(lo_dn), math.log1p(max(age, 0.0)), _safe(logratio)]


# ---------------- run: the CV test, then the head-to-head with the shipped bocpd ----------------
_BACKEND = "fork"
_l, _k, _cand, _base, _ok = test_family(DatasetBOCPDFamily)
_c = compare(_cand, _base)
for k_, p_ in _c.items():
    print(f"  +dbocpd {k_:9s}: {p_['delta']:+.5f} | se {p_['se']:.5f} | {p_['sigma']:+.1f} sigma | better {p_['better']}/{p_['n']}")
print("  VERDICT:", "ADOPT" if _ok else "REJECT")

_Zd = np.load(f"{CACHE}/new_dbocpd.npz")
_dn = [str(s) for s in _Zd["names"]]
_new = _Zd["X"][:, _dn.index("dbocpd_logodds")].astype(np.float64)
_old = base_column("bocpd_logodds").astype(np.float64)
print(f"\n  head-to-head, one feature alone, all rows: TS-AUC  shipped bocpd_logodds {ts_auc(y_all, t_all, _old):.4f}"
      f"  |  dbocpd_logodds {ts_auc(y_all, t_all, _new):.4f}")
print("  (the first check of the idea: if the fitted priors do not make the posterior itself a better ranker,")
print("   the specification was not the bottleneck)")
