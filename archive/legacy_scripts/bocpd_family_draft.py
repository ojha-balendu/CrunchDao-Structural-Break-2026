class BOCPDVarianceFamily(FeatureFamily):
    """Bayesian online filter for ONE variance break in `e`: "no break yet" (N(0,1)) vs
    "break before step k" for every k, post-break variance ~ Inverse-Gamma(A0, B0),
    constant hazard. Closed-form Student-t updates, vectorized over k."""
    name = "bocpd"
    channels = ("e",)
    cost = "O1"          # stateful -> runs every step
    version = 1
    HAZARD = 1.0 / 500.0
    A0 = 2.0
    B0 = 1.0

    def names(self):
        return ["bocpd_logodds", "bocpd_runlen_log", "bocpd_logvar"]

    def init(self, hist):
        cap = 1024
        self.lw = np.zeros(cap)
        self.b = np.zeros(cap)
        self.k = 0
        h = self.HAZARD
        self.c_keep = -math.log1p(-h)
        self.c_new = math.log(h) - math.log1p(-h)
        self._tables(cap)

    def _tables(self, cap):
        j = np.arange(cap, dtype=np.float64)
        a = self.A0 + 0.5 * j
        self.G = _gammaln(a + 0.5) - _gammaln(a)
        self.Acur = a

    def _grow(self):
        cap = 2 * self.lw.size
        for nm in ("lw", "b"):
            arr = np.zeros(cap)
            arr[:self.k] = getattr(self, nm)[:self.k]
            setattr(self, nm, arr)
        self._tables(cap)

    def update(self, ctx):
        x = ctx.vals["e"]
        x2 = x * x
        k = self.k
        if k + 1 >= self.lw.size:
            self._grow()
        lw, b = self.lw, self.b
        if k:
            lw[:k] += self.c_keep
        lw[k] = self.c_new
        b[k] = self.B0
        k += 1
        self.k = k
        G = self.G[k - 1::-1]
        A = self.Acur[k - 1::-1]
        B = b[:k]
        logpred = G - 0.5 * (_LOG_2PI + np.log(B)) - (A + 0.5) * np.log1p(x2 / (2.0 * B))
        lw[:k] += logpred - (-0.5 * _LOG_2PI - 0.5 * x2)
        B += 0.5 * x2
        L = float(lw[:k].max())
        wts = np.exp(lw[:k] - L)
        tot = float(wts.sum())
        logodds = L + math.log(tot)
        wts /= tot
        runlen = float(np.dot(wts, np.arange(k, 0, -1, dtype=np.float64)))
        var = float(np.dot(wts, B / (A + 0.5 - 1.0)))
        return [_safe(logodds), math.log(runlen), _safe(math.log(max(var, 1e-12)))]
