"""Three new families, each from a different line of the online change-detection literature.
All assumption-light: they test the online stream against the historical calibration only
through ranks / PIT values, or not at all.

  ctm     -- conformal test martingales + e-detectors (Vovk et al. 2021 "Retrain or not
             retrain"; Shin, Ramdas & Rinaldo 2023 "E-detectors"). O(1)/step.
  newma   -- NEWMA: two EWMAs of random Fourier features at different rates (Keriven et al.).
             Model-free, no window, no reference sample. O(d)/step.
  bocpd2  -- Bayesian single-change filter with UNKNOWN mean and variance (Normal-Inverse-Gamma),
             unlike `bocpd` which fixes the mean at 0. O(t)/step.
"""
from scipy.special import ndtr as _ndtr, gammaln as _gammaln2

_EPSG = (0.3, 0.5, 0.8)                 # power-betting exponents for the e-detectors


class ConformalMartingaleFamily(FeatureFamily):
    """Testing by betting on conformal p-values.

    Under "no break" the PIT value of each new point against the historical sample is
    Uniform(0,1). Two p-value streams are used:
        p_loc   = u                        (location / tail shifts)
        p_scale = 2*min(F(e), 1-F(e))      (two-sided: small when |e| is large)
    For each stream we keep three statistics, all O(1):
        jumper -- Vovk's Simple Jumper martingale, a mixture over betting slopes
                  eps in {-1,0,1} with jump rate J (bets adapt without tuning)
        sr     -- Shiryaev-Roberts e-detector:  R_t = (1 + R_{t-1}) * e_t
        cusum  -- CUSUM-style e-detector:       C_t = max(1, C_{t-1}) * e_t
    with e_t a mixture of two-sided power bets, mean exactly 1 under the null:
        e_t = mean over g of  (g/2) * (p^(g-1) + (1-p)^(g-1)),
    which is large whenever p lands near either end of [0,1].
    SR and CUSUM restart at every step, so they need no break-point search at all.
    """
    name = "ctm"
    channels = ("e",)
    cost = "O1"
    version = 1
    JUMP = 0.01

    def names(self):
        return ["ctm_jump_loc", "ctm_sr_loc", "ctm_cusum_loc",
                "ctm_jump_scale", "ctm_sr_scale", "ctm_cusum_scale"]

    def init(self, hist):
        self.w = [[1 / 3.0, 1 / 3.0, 1 / 3.0], [1 / 3.0, 1 / 3.0, 1 / 3.0]]   # jumper weights
        self.wlog = [0.0, 0.0]                        # log wealth (overflow guard)
        self.sr = [-math.inf, -math.inf]              # log Shiryaev-Roberts (R_0 = 0)
        self.cu = [0.0, 0.0]                          # log CUSUM (C_0 = 1)

    def update(self, ctx):
        out = []
        p_loc = min(max(ctx.u, 1e-6), 1 - 1e-6)
        pe = float(_ndtr(ctx.vals["e"]))
        p_scale = min(max(2.0 * min(pe, 1.0 - pe), 1e-6), 1 - 1e-6)
        for s, p in enumerate((p_loc, p_scale)):
            J = self.JUMP
            a, b, c = self.w[s]
            m = J * (a + b + c) / 3.0
            q = 1.0 - J
            d = p - 0.5
            a = (q * a + m) * (1.0 - d)                      # eps = -1
            b = (q * b + m)                                  # eps =  0
            c = (q * c + m) * (1.0 + d)                      # eps = +1
            tot = a + b + c
            if tot <= 0.0:
                tot = 1e-300
            self.w[s] = [a / tot, b / tot, c / tot]
            self.wlog[s] += math.log(tot)

            e = 0.0
            for g in _EPSG:
                e += 0.5 * g * (p ** (g - 1.0) + (1.0 - p) ** (g - 1.0))
            e /= len(_EPSG)
            le = math.log(max(e, 1e-300))
            r = self.sr[s]
            self.sr[s] = (le if r == -math.inf else math.log1p(math.exp(min(r, 700.0))) + le
                          if r < 700.0 else r + le)           # log((1 + R) * e)
            self.cu[s] = max(0.0, self.cu[s]) + le            # log(max(1, C) * e)
            out += [_safe(self.wlog[s]), _safe(self.sr[s]), _safe(self.cu[s])]
        return out


class NEWMAFamily(FeatureFamily):
    """NEWMA (Keriven, Garreau & Poli): compare two exponentially weighted moving averages
    of the same random-feature embedding, one fast and one slow. Their difference is small
    while the distribution is stationary and jumps when it changes -- a kernel two-sample
    statistic with no window, no stored sample and no reference set. O(d) per step.
    Random Fourier features are seeded, so the features are deterministic across runs."""
    name = "newma"
    channels = ("e",)
    cost = "O1"
    version = 1
    D = 32                      # random features
    SIGMA = 1.0                 # RBF bandwidth (channel is unit-scale by construction)
    RATES = ((0.05, 0.005), (0.15, 0.015), (0.40, 0.04))    # (fast, slow) pairs

    def names(self):
        return ["newma_%d" % i for i in range(len(self.RATES))]

    def init(self, hist):
        rng = np.random.default_rng(12345)                   # fixed -> deterministic
        self.om = rng.normal(0.0, 1.0 / self.SIGMA, self.D)
        self.b = rng.uniform(0.0, 2 * math.pi, self.D)
        self.scale = math.sqrt(2.0 / self.D)
        self.zf = [np.zeros(self.D) for _ in self.RATES]
        self.zs = [np.zeros(self.D) for _ in self.RATES]
        self.n = 0

    def update(self, ctx):
        x = ctx.vals["e"]
        phi = self.scale * np.cos(self.om * x + self.b)
        self.n += 1
        out = []
        for i, (lf, ls) in enumerate(self.RATES):
            self.zf[i] += lf * (phi - self.zf[i])
            self.zs[i] += ls * (phi - self.zs[i])
            d = self.zf[i] - self.zs[i]
            out.append(_safe(math.log(float(np.dot(d, d)) + 1e-12)))
        return out


class BOCPDMeanVarFamily(FeatureFamily):
    """Bayesian single-change filter with BOTH mean and variance unknown (Normal-Inverse-Gamma
    prior), vectorized over every candidate break point. `bocpd` fixes the mean at 0, so it is
    blind to a pure location shift; this one is not. Posterior predictive is Student-t."""
    name = "bocpd2"
    channels = ("e",)
    cost = "O1"
    version = 1
    HAZARD = 1.0 / 500.0
    MU0, KAPPA0, A0, B0 = 0.0, 1.0, 2.0, 1.0

    def names(self):
        return ["bocpd2_logodds", "bocpd2_runlen_log", "bocpd2_mushift"]

    def init(self, hist):
        cap = 1024
        self.lw = np.zeros(cap)
        self.mu = np.zeros(cap); self.ka = np.zeros(cap)
        self.al = np.zeros(cap); self.be = np.zeros(cap)
        self.k = 0
        h = self.HAZARD
        self.c_keep = -math.log1p(-h)
        self.c_new = math.log(h) - math.log1p(-h)

    def _grow(self):
        old = self.lw.size; cap = 2 * old
        for nm in ("lw", "mu", "ka", "al", "be"):
            arr = np.zeros(cap); arr[:old] = getattr(self, nm)
            setattr(self, nm, arr)

    def update(self, ctx):
        x = ctx.vals["e"]
        k = self.k
        if k + 1 >= self.lw.size:
            self._grow()
        lw = self.lw
        if k:
            lw[:k] += self.c_keep
        lw[k] = self.c_new
        self.mu[k], self.ka[k], self.al[k], self.be[k] = self.MU0, self.KAPPA0, self.A0, self.B0
        k += 1
        self.k = k
        mu, ka, al, be = self.mu[:k], self.ka[:k], self.al[:k], self.be[:k]
        # Student-t predictive: df = 2a, loc = mu, scale^2 = b(k+1)/(a k)
        s2 = be * (ka + 1.0) / (al * ka)
        df = 2.0 * al
        z2 = (x - mu) ** 2 / s2
        logpred = (_gammaln2(al + 0.5) - _gammaln2(al) - 0.5 * np.log(math.pi * df * s2)
                   - (al + 0.5) * np.log1p(z2 / df))
        lw[:k] += logpred - (-0.5 * math.log(2 * math.pi) - 0.5 * x * x)
        # NIG update
        d = x - mu
        self.be[:k] = be + 0.5 * ka * d * d / (ka + 1.0)
        self.mu[:k] = mu + d / (ka + 1.0)
        self.ka[:k] = ka + 1.0
        self.al[:k] = al + 0.5
        L = float(lw[:k].max())
        wts = np.exp(lw[:k] - L)
        tot = float(wts.sum())
        logodds = L + math.log(tot)
        wts /= tot
        runlen = float(np.dot(wts, np.arange(k, 0, -1, dtype=np.float64)))
        mushift = float(np.dot(wts, self.mu[:k]))
        return [_safe(logodds), math.log(runlen), _safe(mushift)]
