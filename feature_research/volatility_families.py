# =====================================================================
#  VOLATILITY-MODEL FAMILIES -- final versions for submission.ipynb (section 1)
#  garch (9) + argarch (11) + tgarch (11) + gjr (5) = 36 features
#  All require HistoricalRef.x_series (the ordered history).
#  NOTE for the notebook: move the `from scipy.special import ...` line to the top-level
#  imports cell -- Crunch rejects imports that are not at module level.
# =====================================================================
class GarchFamily(FeatureFamily):
    """GARCH(1,1) volatility model fitted on the historical segment; features are the online
    surprise relative to the model's own conditional variance (nu^2 = eps^2 / h_t).

    Every other family compares the online window against the UNCONDITIONAL distribution of the
    history. GARCH conditions on the recent past: volatility clustering the process itself
    generates is expected, so only departures from the fitted volatility DYNAMICS register.
    Fit: MLE with variance targeting on the ordered history (HistoricalRef.x_series), then the
    filter is warmed through that history so the first online step starts at the conditional
    variance."""
    name = "garch"
    channels = ("z",)
    cost = "O1"
    version = 3
    WINS = (32, 128, 512)
    EW = (0.02,)

    @staticmethod
    def _nll(al, be, x2, v):
        om = v * (1.0 - al - be)
        if om <= 1e-12:
            return 1e18
        h, s = v, 0.0
        for e2 in x2:
            s += math.log(h) + e2 / h
            h = om + al * e2 + be * h
            if h <= 1e-12:
                return 1e18
        return s

    def _fit(self, x):
        v = max(float(np.var(x)) if x.size > 8 else 1.0, 1e-8)
        x2 = (np.asarray(x, dtype=np.float64) ** 2)[-600:]
        best, bp = None, (0.0, 0.0)
        for al in (0.03, 0.06, 0.10, 0.15, 0.22, 0.30):
            for be in (0.0, 0.40, 0.65, 0.80, 0.88, 0.93):
                if al + be > 0.99:
                    continue
                f = self._nll(al, be, x2, v)
                if np.isfinite(f) and (best is None or f < best):
                    best, bp = f, (al, be)
        al, be = bp
        for step in (0.04, 0.015):
            for da in (-step, 0.0, step):
                for db in (-step, 0.0, step):
                    a2, b2 = al + da, be + db
                    if a2 < 0 or b2 < 0 or a2 + b2 > 0.99:
                        continue
                    f = self._nll(a2, b2, x2, v)
                    if f < best:
                        best, al, be = f, a2, b2
        return float(v * (1.0 - al - be)), float(al), float(be), float(v)

    def names(self):
        out = []
        for W in self.WINS:
            out += ["garch_nu2_w%d" % W, "garch_gap_w%d" % W]
        out += ["garch_nu2_ew%g" % a for a in self.EW]
        return out + ["garch_logh", "garch_nu2_exp"]

    def init(self, hist):
        xs = getattr(hist, "x_series", None)
        assert xs is not None and xs.size >= 50, "HistoricalRef.x_series missing"
        self.mu = float(hist.mu)
        self.sd = float(hist.sd) if hist.sd > 0 else 1.0
        eps_h = np.asarray(xs, dtype=np.float64) - self.mu
        self.om, self.al, self.be, self.v0 = self._fit(eps_h)
        h = self.v0
        for e in eps_h:
            h = self.om + self.al * e * e + self.be * h
        self.h = max(h, 1e-12)
        self.cap = max(self.WINS)
        self.nu2 = np.zeros(self.cap, dtype=np.float64)
        self.gap = np.zeros(self.cap, dtype=np.float64)
        self.n = 0
        self.s1 = 0.0
        self.ew = [0.0] * len(self.EW)

    def _win(self, a, W):
        m = W if self.n > W else self.n
        i = (self.n - 1) % self.cap
        s = i - m + 1
        return a[s:i + 1] if s >= 0 else np.concatenate([a[s % self.cap:], a[:i + 1]])

    def update(self, ctx):
        eps = float(ctx.zr) * self.sd
        h = max(self.h, 1e-12)
        nu2 = min(eps * eps / h, 100.0)
        self.h = self.om + self.al * eps * eps + self.be * h
        i = self.n % self.cap
        self.n += 1
        self.nu2[i] = (nu2 - 1.0) * 0.7071067811865476
        self.gap[i] = math.log(max(eps * eps, 1e-12) / h) if eps != 0.0 else 0.0
        self.s1 += self.nu2[i]
        out = []
        for W in self.WINS:
            a = self._win(self.nu2, W); b = self._win(self.gap, W)
            m = a.size
            if m < 8:
                out += [0.0, 0.0]
                continue
            out.append(_safe(float(a.mean()) * math.sqrt(m)))
            out.append(_safe(float(b.mean()) * math.sqrt(m)))
        for j, al in enumerate(self.EW):
            self.ew[j] += al * (self.nu2[i] - self.ew[j])
            out.append(_safe(self.ew[j] * math.sqrt(min(self.n, 2.0 / al))))
        out.append(_safe(math.log(max(self.h, 1e-12) / max(self.v0, 1e-12))))
        out.append(_safe(self.s1 / self.n * math.sqrt(self.n)))
        return out


# =====================================================================
#  Four conditional models of the HISTORY, each asking "does the online data still follow
#  the dynamics this model learned?"  All need HistoricalRef.x_series (the ordered history).
#    argarch  GARCH on the AR(3) residuals      -> joint mean + variance dynamics
#    tgarch   GARCH with Student-t innovations  -> proper (constant) kurtosis
#    gjr      GJR-GARCH                          -> asymmetric (leverage) volatility response
#    caviar   CAViaR conditional quantiles       -> dynamic tails, tested by exceedance rates
# =====================================================================
from scipy.special import stdtr as _stdtr, ndtri as _ndtri, gammaln as _gln

_SQ2 = 0.7071067811865476


def _history(hist):
    xs = getattr(hist, "x_series", None)
    assert xs is not None and xs.size >= 50, "HistoricalRef.x_series missing -- run the patch cell"
    return np.asarray(xs, dtype=np.float64)


def _ar_resid(xs, mu, ar):
    """AR(p) residuals of the centred history, using the AR coefficients fitted in HistoricalRef."""
    c = xs - mu
    p = len(ar)
    r = c.copy()
    for k in range(1, p + 1):
        r[p:] -= ar[k - 1] * c[p - k:-k]
    return r[p:]


def _garch_grid(x2, v, extra=None, nll=None):
    """Grid + local refinement over (alpha, beta[, extra]) with variance targeting."""
    best, bp = None, None
    extras = extra if extra is not None else (None,)
    for ex in extras:
        for al in (0.03, 0.08, 0.15, 0.25):
            for be in (0.0, 0.50, 0.75, 0.88, 0.94):
                f = nll(al, be, ex, x2, v)
                if np.isfinite(f) and (best is None or f < best):
                    best, bp = f, (al, be, ex)
    al, be, ex = bp
    for step in (0.04, 0.015):
        for da in (-step, 0.0, step):
            for db in (-step, 0.0, step):
                a2, b2 = al + da, be + db
                if a2 < 0 or b2 < 0:
                    continue
                f = nll(a2, b2, ex, x2, v)
                if f < best:
                    best, al, be = f, a2, b2
    return al, be, ex


class _CondBase(FeatureFamily):
    """Shared plumbing: a ring buffer per derived stream, window/EWMA/expanding aggregates."""
    channels = ("z",)
    cost = "O1"
    version = 1
    WINS = (32, 128, 512)
    STREAMS = ()          # names of the derived per-step streams, each ~ mean 0 under the null

    def _setup_buffers(self):
        self.cap = max(self.WINS)
        self.buf = {s: np.zeros(self.cap) for s in self.STREAMS}
        self.acc = {s: 0.0 for s in self.STREAMS}
        self.ew = {s: 0.0 for s in self.STREAMS}
        self.n = 0

    def _push(self, vals):
        i = self.n % self.cap
        self.n += 1
        for s, v in vals.items():
            self.buf[s][i] = v
            self.acc[s] += v
            self.ew[s] += 0.02 * (v - self.ew[s])

    def _win(self, a, W):
        m = W if self.n > W else self.n
        i = (self.n - 1) % self.cap
        s = i - m + 1
        return a[s:i + 1] if s >= 0 else np.concatenate([a[s % self.cap:], a[:i + 1]])

    def _aggregate(self):
        out = []
        for s in self.STREAMS:
            for W in self.WINS:
                w = self._win(self.buf[s], W)
                out.append(_safe(float(w.mean()) * math.sqrt(w.size)) if w.size >= 8 else 0.0)
            out.append(_safe(self.ew[s] * math.sqrt(min(self.n, 100.0))))
            out.append(_safe(self.acc[s] / self.n * math.sqrt(self.n)))
        return out

    def _agg_names(self):
        return [f"{self.name}_{s}_{k}" for s in self.STREAMS
                for k in [f"w{W}" for W in self.WINS] + ["ew", "exp"]]


# ---------------------------------------------------------------- 1. AR-GARCH
class ARGarchFamily(_CondBase):
    """GARCH on the AR(3) residuals instead of the raw series: the mean dynamics are removed first,
    so the volatility model sees genuine innovations. Streams: nu^2 - 1 (scale surprise) and
    log(eps^2/h) (realized vs expected variance)."""
    name = "argarch"
    STREAMS = ("nu2", "gap")

    @staticmethod
    def _nll(al, be, _ex, x2, v):
        om = v * (1.0 - al - be)
        if om <= 1e-12 or al + be >= 0.99:
            return 1e18
        h, s = v, 0.0
        for e2 in x2:
            s += math.log(h) + e2 / h
            h = om + al * e2 + be * h
        return s

    def names(self):
        return self._agg_names() + [f"{self.name}_logh"]

    def init(self, hist):
        xs = _history(hist)
        self.mu, self.sd = float(hist.mu), (float(hist.sd) if hist.sd > 0 else 1.0)
        self.ar = [float(a) for a in hist.ar_l]
        r = _ar_resid(xs, self.mu, self.ar)
        v = max(float(np.var(r)), 1e-10)
        al, be, _ = _garch_grid((r ** 2)[-600:], v, nll=self._nll)
        self.om, self.al, self.be, self.v0 = v * (1 - al - be), al, be, v
        h = v
        for e in r:
            h = self.om + self.al * e * e + self.be * h
        self.h = max(h, 1e-12)
        self.lags = list((xs[-len(self.ar):] - self.mu)[::-1])    # most recent first
        self._setup_buffers()

    def update(self, ctx):
        c = float(ctx.zr) * self.sd
        eps = c - sum(a * l for a, l in zip(self.ar, self.lags))
        self.lags = [c] + self.lags[:-1]
        h = max(self.h, 1e-12)
        nu2 = min(eps * eps / h, 100.0)
        self.h = self.om + self.al * eps * eps + self.be * h
        self._push({"nu2": (nu2 - 1.0) * _SQ2,
                    "gap": math.log(max(eps * eps, 1e-12) / h)})
        return self._aggregate() + [_safe(math.log(self.h / self.v0))]


# ---------------------------------------------------------------- 2. Student-t GARCH
class TGarchFamily(_CondBase):
    """GARCH with Student-t innovations: the degrees of freedom are estimated with alpha/beta, so the
    innovations are standardized by the RIGHT distribution. Each standardized innovation is mapped
    through the t CDF and back to a normal score, so under the null the stream is exactly N(0,1)
    even for fat-tailed data. Streams: z^2 - 1 (scale) and 1[|z|>2] - P(|Z|>2) (tail rate)."""
    name = "tgarch"
    STREAMS = ("z2", "tail2")
    NUS = (4.0, 6.0, 10.0, 30.0)
    P2 = 0.04550026389635842

    @staticmethod
    def _nll(al, be, nu, x2, v):
        om = v * (1.0 - al - be)
        if om <= 1e-12 or al + be >= 0.99:
            return 1e18
        c = _gln((nu + 1) / 2) - _gln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
        h, s = v, 0.0
        for e2 in x2:
            s -= c - 0.5 * math.log(h) - (nu + 1) / 2 * math.log1p(e2 / ((nu - 2) * h))
            h = om + al * e2 + be * h
        return s

    def names(self):
        return self._agg_names() + [f"{self.name}_nu"]

    def init(self, hist):
        xs = _history(hist)
        self.mu, self.sd = float(hist.mu), (float(hist.sd) if hist.sd > 0 else 1.0)
        eps = xs - self.mu
        v = max(float(np.var(eps)), 1e-10)
        al, be, nu = _garch_grid((eps ** 2)[-600:], v, extra=self.NUS, nll=self._nll)
        self.om, self.al, self.be, self.nu, self.v0 = v * (1 - al - be), al, be, nu, v
        self.tscale = math.sqrt(nu / (nu - 2.0))
        h = v
        for e in eps:
            h = self.om + self.al * e * e + self.be * h
        self.h = max(h, 1e-12)
        self._setup_buffers()

    def update(self, ctx):
        eps = float(ctx.zr) * self.sd
        h = max(self.h, 1e-12)
        s = eps / math.sqrt(h) * self.tscale
        u = min(max(float(_stdtr(self.nu, s)), 1e-12), 1 - 1e-12)
        z = float(_ndtri(u))
        self.h = self.om + self.al * eps * eps + self.be * h
        self._push({"z2": (z * z - 1.0) * _SQ2,
                    "tail2": (1.0 if abs(z) > 2.0 else 0.0) - self.P2})
        return self._aggregate() + [_safe(math.log(self.nu))]




# ---------------------------------------------------------------- 3. GJR-GARCH (v2)
class GJRGarchFamily(_CondBase):
    """GJR-GARCH: negative shocks can raise volatility more than positive ones
    (h_t = omega + (alpha + gamma*1[eps<0]) eps^2 + beta h). gamma shapes the filter but is not
    emitted; the asymmetry stream carried no signal. Stream: nu^2 - 1, the scale surprise relative
    to the asymmetric volatility model."""
    name = "gjr"
    version = 2
    STREAMS = ("nu2",)
    GAMMAS = (0.0, 0.05, 0.12, 0.25)

    @staticmethod
    def _nll(al, be, ga, x, v):
        om = v * (1.0 - al - ga / 2.0 - be)
        if om <= 1e-12 or al + ga / 2.0 + be >= 0.99:
            return 1e18
        h, s = v, 0.0
        for e in x:
            e2 = e * e
            s += math.log(h) + e2 / h
            h = om + (al + (ga if e < 0 else 0.0)) * e2 + be * h
        return s

    def names(self):
        return self._agg_names()

    def init(self, hist):
        xs = _history(hist)
        self.mu, self.sd = float(hist.mu), (float(hist.sd) if hist.sd > 0 else 1.0)
        eps = xs - self.mu
        v = max(float(np.var(eps)), 1e-10)
        al, be, ga = _garch_grid(eps[-600:], v, extra=self.GAMMAS, nll=self._nll)
        self.al, self.be, self.ga, self.v0 = al, be, ga, v
        self.om = v * (1 - al - ga / 2 - be)
        h = v
        for e in eps:
            h = self.om + (al + (ga if e < 0 else 0.0)) * e * e + be * h
        self.h = max(h, 1e-12)
        self._setup_buffers()

    def update(self, ctx):
        eps = float(ctx.zr) * self.sd
        h = max(self.h, 1e-12)
        nu2 = min(eps * eps / h, 100.0)
        self.h = self.om + (self.al + (self.ga if eps < 0 else 0.0)) * eps * eps + self.be * h
        self._push({"nu2": (nu2 - 1.0) * _SQ2})
        return self._aggregate()
