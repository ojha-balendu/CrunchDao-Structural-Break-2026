# =====================================================================
#  Five families from the structural-break literature, all built on the lesson GARCH taught:
#  MODEL THE HISTORY, then measure how the online data departs from that model.
#  They need _CondBase, _history (from the volatility-families cell) and HistoricalRef.x_series.
#
#    ordpat  ordinal-pattern distribution vs the history's (Bandt & Pompe 2002; Sinn, Ghodsi &
#            Keller 2012; Unakafov & Keller 2018) -- dynamics, invariant to any monotone transform
#    wavvar  Haar wavelet variance per time scale, relative to the HISTORY's own wavelet spectrum
#            (Percival & Walden 2000; Whitcher, Guttorp & Percival 2000)
#    recres  recursive residuals of the history's AR model + CUSUM / CUSUM-of-squares
#            (Brown, Durbin & Evans 1975; Ploberger & Kraemer 1992) and a sequential Wald distance
#    hmm     2-state Gaussian hidden Markov model fit on the history (Hamilton 1989):
#            predictive surprise and regime occupancy
#    evt     peaks-over-threshold / generalized Pareto tail model of the history
#            (Pickands 1975; Balkema & de Haan 1974): exceedance rate, excess likelihood, excess PIT
# =====================================================================
import itertools
from scipy.special import ndtri as _ndtri_l3


def _std_history(hist):
    xs = _history(hist)
    sd = float(hist.sd) if hist.sd > 0 else 1.0
    return (xs - float(hist.mu)) / sd


# ------------------------------------------------------------------ 1. ordinal patterns
class OrdinalPatternFamily(_CondBase):
    """Each window of d consecutive values is reduced to the permutation that sorts it (its
    'ordinal pattern'). The pattern distribution encodes the dynamics -- trends, oscillation,
    dependence -- and is unchanged by any monotone transform, so it ignores scale entirely.
    The history's pattern probabilities are the reference; streams are the per-step surprise
    -log p_hist(pattern) - H_hist for d = 3 and 4 (mean zero under no break), the monotone-pattern
    rate minus its historical rate, plus a KL divergence of the last 128 patterns (d = 3)."""
    name = "ordpat"
    version = 1
    STREAMS = ("s3", "s4", "mono")
    KL_W = 128

    def names(self):
        return self._agg_names() + ["ordpat_kl3_w128"]

    @staticmethod
    def _table(d):
        return {p: i for i, p in enumerate(itertools.permutations(range(d)))}

    @staticmethod
    def _code(vals, tab):
        return tab[tuple(sorted(range(len(vals)), key=lambda i: vals[i]))]

    def init(self, hist):
        z = _std_history(hist)
        self.t3, self.t4 = self._table(3), self._table(4)
        c3 = np.ones(6); c4 = np.ones(24)
        for i in range(3, z.size):
            c4[self._code(z[i - 3:i + 1], self.t4)] += 1
            c3[self._code(z[i - 2:i + 1], self.t3)] += 1
        self.p3, self.p4 = c3 / c3.sum(), c4 / c4.sum()
        self.lp3, self.lp4 = -np.log(self.p3), -np.log(self.p4)
        self.h3, self.h4 = float(self.p3 @ self.lp3), float(self.p4 @ self.lp4)
        self.mono = {self.t3[(0, 1, 2)], self.t3[(2, 1, 0)]}
        self.pmono = float(sum(self.p3[i] for i in self.mono))
        self.last = list(z[-3:])
        self.kbuf = np.zeros(self.KL_W, dtype=np.int64); self.kcnt = np.zeros(6); self.kn = 0
        self._setup_buffers()

    def update(self, ctx):
        v = float(ctx.zr)
        w4 = self.last + [v]
        self.last = w4[1:]
        c3 = self._code(w4[1:], self.t3); c4 = self._code(w4, self.t4)
        i = self.kn % self.KL_W
        if self.kn >= self.KL_W:
            self.kcnt[self.kbuf[i]] -= 1
        self.kbuf[i] = c3; self.kcnt[c3] += 1; self.kn += 1
        self._push({"s3": self.lp3[c3] - self.h3, "s4": self.lp4[c4] - self.h4,
                    "mono": (1.0 if c3 in self.mono else 0.0) - self.pmono})
        m = min(self.kn, self.KL_W)
        q = (self.kcnt + 0.5) / (m + 3.0)
        return self._aggregate() + [_safe(float(m * np.sum(q * np.log(q / self.p3))))]


# ------------------------------------------------------------------ 2. wavelet variance by scale
class WaveletVarianceFamily(_CondBase):
    """Haar wavelet coefficients at scales L = 1, 2, 4, 8:
        d_L(t) = (sum of the last L values - sum of the L before) / sqrt(2L)
    Their variance at each scale is the series' energy in that frequency band. The HISTORY's
    wavelet variance per scale is the reference, so each stream d_L^2 / v_hist(L) - 1 is zero-mean
    under no break whatever the history's own spectrum looks like -- the 'relative to the history's
    own model' step the plain spectral features lacked."""
    name = "wavvar"
    version = 1
    SCALES = (1, 2, 4, 8)
    STREAMS = tuple(f"L{L}" for L in SCALES)

    def names(self):
        return self._agg_names()

    def init(self, hist):
        z = _std_history(hist)
        c = np.concatenate([[0.0], np.cumsum(z)])
        self.vh = {}
        for L in self.SCALES:
            if z.size < 4 * L:
                self.vh[L] = 1.0
                continue
            s_now = c[2 * L:] - c[L:-L]
            s_prev = c[L:-L] - c[:-2 * L]
            d = (s_now - s_prev) / math.sqrt(2 * L)
            self.vh[L] = max(float(np.mean(d * d)), 1e-8)
        self.win = list(z[-2 * max(self.SCALES):])
        self._setup_buffers()

    def update(self, ctx):
        self.win = self.win[1:] + [float(ctx.zr)]
        vals = {}
        for L in self.SCALES:
            a = sum(self.win[-L:]); b = sum(self.win[-2 * L:-L])
            d = (a - b) / math.sqrt(2 * L)
            vals[f"L{L}"] = d * d / self.vh[L] - 1.0
        self._push(vals)
        return self._aggregate()


# ------------------------------------------------------------------ 3. recursive residuals
class RecursiveResidualFamily(_CondBase):
    """The history's AR(3) regression (with intercept) is fit by OLS, then updated online by
    recursive least squares with forgetting factor LAM. The one-step recursive residual
        w_t = (z_t - x_t' beta_{t-1}) / sqrt(LAM + x_t' P_{t-1} x_t) / sigma_hist
    is iid N(0,1)-like while the regression holds (Brown-Durbin-Evans). Streams: w (its CUSUM
    detects drift in any coefficient, including the intercept) and w^2 - 1 (CUSUM of squares:
    variance). Plus a sequential Wald distance between the tracked coefficients and the
    history's, scaled by the history's design matrix -- how far the dynamics have moved."""
    name = "recres"
    version = 1
    STREAMS = ("w", "w2")
    P_AR = 3
    LAM = 0.995

    def names(self):
        return self._agg_names() + ["recres_wald"]

    def init(self, hist):
        z = _std_history(hist)
        p = self.P_AR
        X = np.column_stack([np.ones(z.size - p)] + [z[p - k:z.size - k] for k in range(1, p + 1)])
        y = z[p:]
        XtX = X.T @ X + 1e-6 * np.eye(p + 1)
        self.beta_h = np.linalg.solve(XtX, X.T @ y)
        r = y - X @ self.beta_h
        self.sig = max(float(np.std(r)), 1e-6)
        self.beta = self.beta_h.copy()
        self.P = np.linalg.inv(XtX) * (X.shape[0] * (1.0 - self.LAM))   # as if the history were
        self.M = XtX / X.shape[0] / (1.0 - self.LAM)                      # an LAM-weighted sample
        self.lags = list(z[-p:][::-1])
        self._setup_buffers()

    def update(self, ctx):
        z = float(ctx.zr)
        x = np.array([1.0] + self.lags)
        Px = self.P @ x
        f = self.LAM + float(x @ Px)
        e = z - float(x @ self.beta)
        w = e / math.sqrt(max(f, 1e-12)) / self.sig
        k = Px / f
        self.beta = self.beta + k * e
        self.P = (self.P - np.outer(k, Px)) / self.LAM
        self.lags = [z] + self.lags[:-1]
        w = max(min(w, 20.0), -20.0)
        self._push({"w": w, "w2": (w * w - 1.0) * 0.7071067811865476})
        db = self.beta - self.beta_h
        return self._aggregate() + [_safe(float(db @ self.M @ db) / (self.sig ** 2))]


# ------------------------------------------------------------------ 4. hidden Markov regimes
class HMMRegimeFamily(_CondBase):
    """A 2-state Gaussian HMM (Hamilton 1989) is fit on the history by EM (scaled forward-backward).
    Online, the forward filter gives the predictive density of each new point and the filtered
    probability of the high-variance regime. Streams: predictive surprise
    -log p(z_t | past) minus its historical average, and high-regime occupancy minus its
    historical average. Unlike GARCH, the volatility moves by SWITCHING between two levels, so a
    break that changes how often or how long the process visits each regime shows up here."""
    name = "hmm"
    version = 1
    STREAMS = ("surp", "hi")
    EM_ITERS = 12
    FIT_LAST = 800

    def names(self):
        return self._agg_names() + ["hmm_p_hi"]

    @staticmethod
    def _dens(x, mu, var):
        return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)

    def _fit(self, z):
        v = max(float(np.var(z)), 1e-6)
        mu = np.array([float(np.mean(z)), float(np.mean(z))])
        var = np.array([0.5 * v, 2.0 * v])
        A = np.array([[0.97, 0.03], [0.03, 0.97]])
        pi = np.array([0.5, 0.5])
        T = z.size
        for _ in range(self.EM_ITERS):
            B = np.stack([self._dens(z, mu[0], var[0]), self._dens(z, mu[1], var[1])], axis=1) + 1e-300
            al = np.empty((T, 2)); c = np.empty(T)
            a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
            for t in range(1, T):
                a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
            be = np.empty((T, 2)); be[-1] = 1.0
            for t in range(T - 2, -1, -1):
                be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
            g = al * be; g /= g.sum(axis=1, keepdims=True)
            xi = (al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :]) / c[1:, None, None]
            A = xi.sum(axis=0); A /= A.sum(axis=1, keepdims=True)
            pi = g[0]
            wsum = g.sum(axis=0) + 1e-12
            mu = (g * z[:, None]).sum(axis=0) / wsum
            var = np.maximum((g * (z[:, None] - mu) ** 2).sum(axis=0) / wsum, 1e-4 * v)
        if var[0] > var[1]:                                     # state 1 = high variance
            mu, var, A = mu[::-1], var[::-1], A[::-1, ::-1]
        return mu, var, A

    def init(self, hist):
        z = _std_history(hist)[-self.FIT_LAST:]
        self.mu, self.var, self.A = self._fit(z)
        al = np.array([0.5, 0.5]); surp, hi = [], []
        for x in z:                                             # historical averages of the streams
            pr = al @ self.A
            b = self._dens(x, self.mu, self.var) + 1e-300
            p = float(pr @ b); al = pr * b / p
            surp.append(-math.log(p)); hi.append(al[1])
        self.al = al
        self.surp0, self.hi0 = float(np.mean(surp)), float(np.mean(hi))
        self._setup_buffers()

    def update(self, ctx):
        x = float(ctx.zr)
        pr = self.al @ self.A
        b = self._dens(x, self.mu, self.var) + 1e-300
        p = float(pr @ b)
        self.al = pr * b / p
        self._push({"surp": -math.log(max(p, 1e-300)) - self.surp0, "hi": float(self.al[1]) - self.hi0})
        return self._aggregate() + [_safe(float(self.al[1]))]


# ------------------------------------------------------------------ 5. extreme-value tail model
class EVTTailFamily(_CondBase):
    """Peaks over threshold: u = 95th percentile of |z| in the history; the excesses |z| - u are
    fit with a generalized Pareto distribution (method of moments). Streams, all zero-mean under
    no break: the standardized exceedance rate; for exceedances, the GPD negative log-likelihood of
    the excess minus its historical average (how implausible the SIZE of the extreme is); and the
    normal score of the excess's GPD CDF (whether extremes land deeper in the tail). The existing
    tail family counts exceedances of fixed thresholds; this models the tail's SHAPE."""
    name = "evt"
    version = 1
    STREAMS = ("rate", "nll", "pit")
    Q = 0.95

    def names(self):
        return self._agg_names() + ["evt_xi"]

    def _gpd_nll(self, y):
        if abs(self.xi) < 1e-6:
            return math.log(self.sg) + y / self.sg
        a = 1.0 + self.xi * y / self.sg
        if a <= 1e-12:
            return 50.0
        return math.log(self.sg) + (1.0 / self.xi + 1.0) * math.log(a)

    def _gpd_cdf(self, y):
        if abs(self.xi) < 1e-6:
            return 1.0 - math.exp(-y / self.sg)
        a = 1.0 + self.xi * y / self.sg
        return 1.0 if a <= 0 else 1.0 - a ** (-1.0 / self.xi)

    def init(self, hist):
        az = np.abs(_std_history(hist))
        self.u = float(np.quantile(az, self.Q))
        ex = az[az > self.u] - self.u
        if ex.size >= 20:
            m, v = float(ex.mean()), float(ex.var())
            r = m * m / max(v, 1e-12)
            self.xi = float(np.clip(0.5 * (1.0 - r), -0.45, 0.8))
            self.sg = max(0.5 * m * (1.0 + r), 1e-6)
        else:
            self.xi, self.sg = 0.1, max(float(ex.mean()) if ex.size else 0.3, 1e-6)
        self.p = max(float((az > self.u).mean()), 1e-3)
        nll_h = [self._gpd_nll(y) for y in ex] if ex.size else [0.0]
        self.nll0 = float(np.mean(nll_h))
        self._setup_buffers()

    def update(self, ctx):
        a = abs(float(ctx.zr))
        hit = a > self.u
        vals = {"rate": ((1.0 if hit else 0.0) - self.p) / math.sqrt(self.p * (1.0 - self.p)),
                "nll": 0.0, "pit": 0.0}
        if hit:
            y = a - self.u
            vals["nll"] = min(self._gpd_nll(y) - self.nll0, 50.0)
            F = min(max(self._gpd_cdf(y), 1e-9), 1 - 1e-9)
            vals["pit"] = float(_ndtri_l3(F))
        self._push(vals)
        return self._aggregate() + [_safe(self.xi)]
