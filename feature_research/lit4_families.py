# =====================================================================
#  Three more "model the history's dynamics, then measure departure" families, from literature
#  this project hasn't used yet. All need _CondBase, _history (volatility-families cell) and
#  HistoricalRef.x_series; _std_history is defined in the lit3 cell.
#
#    ssa     Singular Spectrum Analysis subspace distance
#            (Moskvina & Zhigljavsky 2003, "An algorithm based on singular spectrum analysis for
#            change-point detection"; Golyandina, Nekrutkin & Zhigljavsky 2001)
#    mchain  first-order Markov chain on the history's quantile states: transition surprise and
#            divergence (discrete-state change detection; Lai 1995 on sequential tests for Markov
#            processes) -- nonlinear, asymmetric dependence that AR and GARCH cannot express
#    xrec    cross-recurrence with the history in delay-embedding space (Marwan, Romano, Thiel &
#            Kurths 2007, "Recurrence plots for the analysis of complex systems"; Zbilut & Webber):
#            does the current JOINT state (z_t, z_t-1, z_t-2) revisit states the history visited?
# =====================================================================


# ------------------------------------------------------------------ 1. SSA subspace distance
class SSASubspaceFamily(_CondBase):
    """Embed the history as lagged vectors of length L (the trajectory matrix), take the top-r
    eigenvectors of their covariance: the 'signal subspace', which encodes the history's dominant
    oscillations and linear recurrences. Each new lagged vector is projected onto that subspace.
    Streams, standardized by the same quantities over the history: the squared distance of the
    new vector from the subspace, and the share of its energy the subspace captures. A change in
    the dynamics -- new periodicity, changed persistence, a structural shift -- moves vectors
    out of the history's subspace even when the marginal distribution is unchanged."""
    name = "ssa"
    version = 1
    STREAMS = ("dist", "frac")
    L = 16
    R = 3
    FIT_LAST = 1500

    def names(self):
        return self._agg_names()

    def init(self, hist):
        z = _std_history(hist)[-self.FIT_LAST:]
        L = self.L
        if z.size < 4 * L:
            z = np.concatenate([np.zeros(4 * L - z.size), z])
        X = np.lib.stride_tricks.sliding_window_view(z, L)
        C = X.T @ X / X.shape[0]
        w, V = np.linalg.eigh(C)
        self.U = V[:, np.argsort(w)[::-1][:self.R]]
        tot = np.einsum("ij,ij->i", X, X)
        proj = np.einsum("ij,ij->i", X @ self.U, X @ self.U)
        d = tot - proj
        fr = proj / np.maximum(tot, 1e-12)
        self.dm, self.ds = float(d.mean()), max(float(d.std()), 1e-9)
        self.fm, self.fs = float(fr.mean()), max(float(fr.std()), 1e-9)
        self.win = list(z[-L:])
        self._setup_buffers()

    def update(self, ctx):
        self.win = self.win[1:] + [float(ctx.zr)]
        v = np.asarray(self.win)
        tot = float(v @ v)
        p = self.U.T @ v
        proj = float(p @ p)
        d = tot - proj
        fr = proj / max(tot, 1e-12)
        self._push({"dist": min((d - self.dm) / self.ds, 50.0), "frac": (fr - self.fm) / self.fs})
        return self._aggregate()


# ------------------------------------------------------------------ 2. Markov chain on quantile states
class MarkovChainFamily(_CondBase):
    """Quantize the series into K states at the history's own quantiles, estimate the history's
    K x K transition matrix, and score the online path under it. Unlike an AR model this captures
    NONLINEAR and ASYMMETRIC dependence (e.g. large moves followed by reversals, low values being
    sticky); unlike GARCH it is about levels, not squared shocks. Streams, zero-mean under no break:
    transition surprise -log P[s_{t-1}, s_t] minus its historical average, marginal surprise
    -log pi[s_t] minus its historical average, and the stay-in-state rate minus its historical
    rate. Plus a KL divergence of the transitions in the last 256 steps from the history's."""
    name = "mchain"
    version = 1
    STREAMS = ("trans", "marg", "stay")
    K = 6
    KL_W = 256

    def names(self):
        return self._agg_names() + ["mchain_kl_w256"]

    def init(self, hist):
        z = _std_history(hist)
        self.edges = np.quantile(z, np.arange(1, self.K) / self.K)
        s = np.searchsorted(self.edges, z)
        N = np.full((self.K, self.K), 0.5)
        np.add.at(N, (s[:-1], s[1:]), 1.0)
        self.P = N / N.sum(axis=1, keepdims=True)
        self.lP = -np.log(self.P)
        pi = np.bincount(s, minlength=self.K) + 0.5
        self.pi = pi / pi.sum()
        self.lpi = -np.log(self.pi)
        self.h_trans = float(np.sum(self.pi[:, None] * self.P * self.lP))
        self.h_marg = float(self.pi @ self.lpi)
        self.p_stay = float(np.sum(self.pi * np.diag(self.P)))
        self.prev = int(s[-1])
        self.tbuf = np.zeros((self.KL_W, 2), dtype=np.int64)
        self.tcnt = np.zeros((self.K, self.K)); self.tn = 0
        self._setup_buffers()

    def update(self, ctx):
        st = int(np.searchsorted(self.edges, float(ctx.zr)))
        a = self.prev
        i = self.tn % self.KL_W
        if self.tn >= self.KL_W:
            o = self.tbuf[i]; self.tcnt[o[0], o[1]] -= 1
        self.tbuf[i] = (a, st); self.tcnt[a, st] += 1; self.tn += 1
        self._push({"trans": self.lP[a, st] - self.h_trans,
                    "marg": self.lpi[st] - self.h_marg,
                    "stay": (1.0 if st == a else 0.0) - self.p_stay})
        self.prev = st
        m = min(self.tn, self.KL_W)
        rows = self.tcnt.sum(axis=1)
        Q = (self.tcnt + 0.1) / (rows[:, None] + 0.1 * self.K)
        kl = float(np.sum((rows / m)[:, None] * Q * np.log(Q / self.P)))
        return self._aggregate() + [_safe(m * kl)]


# ------------------------------------------------------------------ 3. cross-recurrence with the history
class CrossRecurrenceFamily(_CondBase):
    """Delay-embed the series as v_t = (z_t, z_t-1, z_t-2) -- the joint state of the process.
    The history's embedded vectors are the reference 'attractor'. For each new state: the
    cross-recurrence rate (share of history states within radius eps, eps set so the history's own
    recurrence rate is ~5%) and the log distance to its k-th nearest history state, each
    standardized by the same quantities computed within the history (with a Theiler window so a
    point never counts its own neighbours in time). Every other family compares MARGINAL
    distributions or fits a parametric model; this asks, non-parametrically, whether the process is
    visiting joint states its history never visited."""
    name = "xrec"
    version = 1
    STREAMS = ("rr", "knn")
    M = 3
    NREF = 1000
    K = 10
    THEILER = 3

    def names(self):
        return self._agg_names()

    def init(self, hist):
        z = _std_history(hist)
        M = self.M
        E = np.lib.stride_tricks.sliding_window_view(z, M)[:, ::-1]      # (z_t, z_t-1, z_t-2)
        self.H = np.ascontiguousarray(E[-self.NREF:])
        n = self.H.shape[0]
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(300, n), replace=False)
        D = np.sqrt(((self.H[idx, None, :] - self.H[None, :, :]) ** 2).sum(-1))
        far = np.abs(idx[:, None] - np.arange(n)[None, :]) > self.THEILER
        self.eps = float(np.quantile(D[far], 0.05))
        rr = np.array([(D[i][far[i]] < self.eps).mean() for i in range(len(idx))])
        kk = min(self.K, int(far.sum(axis=1).min()) - 1)
        self.kk = max(kk, 1)
        kn = np.array([np.log(np.partition(D[i][far[i]], self.kk)[self.kk] + 1e-9) for i in range(len(idx))])
        self.rm, self.rs = float(rr.mean()), max(float(rr.std()), 1e-6)
        self.km, self.ks = float(kn.mean()), max(float(kn.std()), 1e-6)
        self.win = list(z[-M:])
        self._setup_buffers()

    def update(self, ctx):
        self.win = self.win[1:] + [float(ctx.zr)]
        v = np.asarray(self.win[::-1])
        d = np.sqrt(((self.H - v) ** 2).sum(axis=1))
        rr = float((d < self.eps).mean())
        kn = math.log(float(np.partition(d, self.kk)[self.kk]) + 1e-9)
        self._push({"rr": (rr - self.rm) / self.rs, "knn": (kn - self.km) / self.ks})
        return self._aggregate()
