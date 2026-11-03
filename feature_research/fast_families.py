"""
Optimized, output-equivalent replacements for the families that survive the prune,
plus a faster StreamFeatures driver. Paste AFTER the original family definitions
(i.e. after section 1i) -- it re-uses HistoricalRef, WINDOWS, LBUF, stride_for, _safe, etc.

Every optimisation here is exact (identical math, only float-rounding order may differ,
max abs diff ~1e-12). Verified against the original classes with check_fast_equivalence().
"""
import math
import bisect
import numpy as np

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
_COEF_CACHE = {}


def _sorted_median(s):
    m = s.size
    return 0.5 * (s[(m - 1) // 2] + s[m // 2])


def _sorted_percentile(s, q):
    """== np.percentile(s, q) (linear) for an already-sorted 1-D array."""
    pos = (s.size - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, s.size - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _avg_ranks_of(x_sorted_self, x, other_sorted):
    """Average (tie-corrected, 1-based) ranks of the values `x` inside the combined
    sample (self ++ other), without concatenating or re-sorting anything."""
    less = np.searchsorted(x_sorted_self, x, "left") + np.searchsorted(other_sorted, x, "left")
    leq = np.searchsorted(x_sorted_self, x, "right") + np.searchsorted(other_sorted, x, "right")
    return 0.5 * (less + 1 + leq)


def _lmom_coefs(n):
    c = _COEF_CACHE.get(("lmom", n))
    if c is None:
        i = np.arange(1, n + 1, dtype=np.float64)
        c1 = (i - 1) / (n - 1) / n
        c2 = (i - 1) * (i - 2) / ((n - 1) * (n - 2)) / n
        c3 = (i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) / n
        c = _COEF_CACHE[("lmom", n)] = (c1, c2, c3)
    return c


def _l_moments_fast(x):
    n = x.size
    if n < 4:
        return (0.0, 1e-6, 0.0, 0.0)
    c1, c2, c3 = _lmom_coefs(n)
    b0 = x.mean()
    b1 = float(np.dot(c1, x))
    b2 = float(np.dot(c2, x))
    b3 = float(np.dot(c3, x))
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return (float(b0), float(max(l2, 1e-12)), float(l3), float(l4))


def _vasicek_fast(sorted_x):
    m = sorted_x.size
    if m < 8:
        return 0.0
    key = ("vas", m)
    c = _COEF_CACHE.get(key)
    if c is None:
        q = max(1, int(round(math.sqrt(m))))
        idx = np.arange(m)
        c = _COEF_CACHE[key] = (np.clip(idx - q, 0, m - 1), np.clip(idx + q, 0, m - 1), m / (2.0 * q))
    lo, hi, scale = c
    diffs = np.maximum(sorted_x[hi] - sorted_x[lo], 1e-9)
    return float(np.mean(np.log(scale * diffs)))


class _FastTwoSample(TwoSampleFamily):
    """Shared: sorted reference + cached per-reference constants."""

    def init(self, hist):
        super().init(hist)
        self.ref_sorted = {}
        for ch in ("z", "e"):
            r = self.ref[ch]
            if np.any(np.diff(r) < 0):
                r = np.sort(r)
            self.ref_sorted[ch] = r

    def update(self, ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                out.append(_safe(self.stat_fast(ctx, ch, W)))
        return out


# ---------------------------------------------------------------------------
# Tier 1 scale tests
# ---------------------------------------------------------------------------
class FTestFast(_FastTwoSample):
    name = "ftest"

    def init(self, hist):
        super().init(hist)
        self.v2 = {ch: _mean_var(self.ref[ch])[1] for ch in ("z", "e")}

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        m, nr = w.size, self.ref[ch].size
        if not _guard(m, nr):
            return 0.0
        v1 = ctx.window_var(ch, W)
        se = math.sqrt(2.0 / (m - 1) + 2.0 / (nr - 1))
        return math.log(v1 / self.v2[ch]) / se


class BartlettFast(FTestFast):
    name = "bartlett"

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        m, nr = w.size, self.ref[ch].size
        if not _guard(m, nr):
            return 0.0
        v1, v2 = ctx.window_var(ch, W), self.v2[ch]
        sp2 = ((m - 1) * v1 + (nr - 1) * v2) / (m + nr - 2)
        b = (m + nr - 2) * math.log(sp2) - (m - 1) * math.log(v1) - (nr - 1) * math.log(v2)
        c = 1.0 + (1.0 / 3.0) * (1.0 / (m - 1) + 1.0 / (nr - 1) - 1.0 / (m + nr - 2))
        return ((b / c if c > 0 else 0.0) - 1.0) / math.sqrt(2.0)


class LeveneFast(_FastTwoSample):
    name = "levene"

    def init(self, hist):
        super().init(hist)
        self.rc = {}
        for ch in ("z", "e"):
            r = self.ref[ch]
            zr = np.abs(r - np.median(r))
            zb = zr.mean()
            self.rc[ch] = (zr.sum(), zb, float(((zr - zb) ** 2).sum()))

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        m, nr = w.size, self.ref[ch].size
        if not _guard(m, nr):
            return 0.0
        zr_sum, zbar_r, ss_r = self.rc[ch]
        zw = np.abs(w - _sorted_median(w))
        zbar_w = zw.mean()
        zbar = (zw.sum() + zr_sum) / (m + nr)
        ssb = m * (zbar_w - zbar) ** 2 + nr * (zbar_r - zbar) ** 2
        ssw = float(((zw - zbar_w) ** 2).sum() + ss_r)
        df2 = m + nr - 2
        if ssw <= 0 or df2 <= 0:
            return 0.0
        f = ssb / (ssw / df2)
        return (1.0 if zbar_w >= zbar_r else -1.0) * math.sqrt(max(f, 0.0))


class FlignerFast(_FastTwoSample):
    name = "fligner"

    def init(self, hist):
        super().init(hist)
        self.ar = {}
        for ch in ("z", "e"):
            r = self.ref[ch]
            ar = np.sort(np.abs(r - np.median(r)))
            self.ar[ch] = (ar, np.searchsorted(ar, ar, "left"), np.searchsorted(ar, ar, "right"))

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        m = w.size
        ar, ar_l, ar_r = self.ar[ch]
        nr = ar.size
        if not _guard(m, nr):
            return 0.0
        aw = np.sort(np.abs(w - _sorted_median(w)))
        n = m + nr
        rk_w = _avg_ranks_of(aw, aw, ar)
        rk_r = 0.5 * (ar_l + np.searchsorted(aw, ar, "left") + 1 + ar_r + np.searchsorted(aw, ar, "right"))
        sw = ndtri(0.5 + rk_w / (2.0 * (n + 1)))
        sr = ndtri(0.5 + rk_r / (2.0 * (n + 1)))
        tot = sw.sum() + sr.sum()
        mean_score = tot / n
        ss = ((sw - mean_score) ** 2).sum() + ((sr - mean_score) ** 2).sum()
        var_score = ss / (n - 1)
        var = m * nr / n * var_score
        return (sw.sum() - m * mean_score) / math.sqrt(max(var, 1e-9))


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------
class LMomentsFast(LMomentsFamily):
    def update(self, ctx):
        out = []
        for ch in self.channels:
            l1r, l2r, l3r, l4r = self.ref_lmom[ch]
            t3r, t4r = l3r / l2r, l4r / l2r
            for W in WINDOWS:
                w = ctx.sorted_window(ch, W)
                m = w.size
                if m < 8:
                    out += [0.0, 0.0, 0.0]
                    continue
                l1, l2, l3, l4 = _l_moments_fast(w)
                out.append(_safe(math.log(max(l2, 1e-9) / l2r) * math.sqrt(m - 1)))
                out.append(_safe((l3 / l2 - t3r) * math.sqrt(m)))
                out.append(_safe((l4 / l2 - t4r) * math.sqrt(m)))
        return out


class IQRFast(IQRRatioFamily):
    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        iqr_w = float(_sorted_percentile(w, 75) - _sorted_percentile(w, 25))
        iqr_r = self.ref_iqr[self._cur_ch]
        if iqr_w <= 0 or iqr_r <= 0:
            return 0.0
        return math.log(iqr_w / iqr_r) * math.sqrt(max(m - 1, 1))


class _RankScaleFast(_FastTwoSample):
    def w_ranks(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        return w, _avg_ranks_of(w, w, self.ref_sorted[ch])


class AnsariFast(_RankScaleFast):
    name = "ansari"

    def stat_fast(self, ctx, ch, W):
        w, rk = self.w_ranks(ctx, ch, W)
        m, nr = w.size, self.ref[ch].size
        if not _guard(m, nr):
            return 0.0
        n = m + nr
        ab = np.minimum(rk, n + 1 - rk).sum()
        if n % 2 == 0:
            e = m * (n + 2) / 4.0
            var = m * nr * (n + 2) * (n - 2) / (48.0 * (n - 1))
        else:
            e = m * (n + 1) ** 2 / (4.0 * n)
            var = m * nr * (n + 1) * (3 + n * n) / (48.0 * n * n)
        return (ab - e) / math.sqrt(var) if var > 0 else 0.0


class MoodFast(_RankScaleFast):
    name = "mood"

    def stat_fast(self, ctx, ch, W):
        w, rk = self.w_ranks(ctx, ch, W)
        m, nr = w.size, self.ref[ch].size
        if not _guard(m, nr):
            return 0.0
        n = m + nr
        t = ((rk - (n + 1) / 2.0) ** 2).sum()
        e = m * (n * n - 1) / 12.0
        var = m * nr * (n + 1) * (n * n - 4) / 180.0
        return (t - e) / math.sqrt(var) if var > 0 else 0.0


class SiegelTukeyFast(_FastTwoSample):
    name = "sigtuk"

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        r = self.ref_sorted[ch]
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        n = m + nr
        n_lo = (n + 1) // 2
        # stable-sort position of each (sorted) window element in concat([w, r])
        pos = np.arange(m) + np.searchsorted(r, w, "left")
        st = np.where(pos < n_lo, 2.0 * pos + 1.0, 2.0 * (n - 1 - pos) + 2.0)
        mean = m * (n + 1) / 2.0
        var = m * nr * (n + 1) / 12.0
        return (st.sum() - mean) / math.sqrt(var) if var > 0 else 0.0


# ---------------------------------------------------------------------------
# Tier 3 / 4
# ---------------------------------------------------------------------------
class WassersteinFast(_FastTwoSample):
    name = "wass"
    cost = "heavy"

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        r = self.ref[ch]
        m, nr = w.size, r.size
        if not _guard(m, nr, 8):
            return 0.0
        key = ("wq", ch, m)
        rq = self._qcache.get(key)
        if rq is None:
            q = (np.arange(1, m + 1) - 0.5) / m
            rq = self._qcache[key] = np.quantile(r, q)
        return float(np.abs(w - rq).mean()) * math.sqrt(m)

    def init(self, hist):
        super().init(hist)
        self._qcache = {}


class EnergyFast(EnergyDistanceFamily):
    def init(self, hist):
        super().init(hist)
        self.pre = {}
        for ch in ("z", "e"):
            rs = np.sort(self.refsub[ch])
            nr = rs.size
            rr = np.abs(self.refsub[ch][:, None] - self.refsub[ch][None, :]).mean()
            self.pre[ch] = (rs, np.concatenate([[0.0], np.cumsum(rs)]), rr)
        self._coef = {}

    def update(self, ctx):
        out = []
        for ch in self.channels:
            rs, cs, rr = self.pre[ch]
            nr, tot = rs.size, cs[-1]
            for W in WINDOWS:
                w = ctx.sorted_window(ch, W)
                m = w.size
                if m < 8:
                    out.append(0.0)
                    continue
                c = self._coef.get(m)
                if c is None:
                    c = self._coef[m] = (2.0 * np.arange(m) - m + 1.0) * 2.0 / (m * m)
                ww = float(np.dot(c, w))
                k = np.searchsorted(rs, w, "right")
                wr = float((w * k - cs[k] + (tot - cs[k]) - w * (nr - k)).sum()) / (m * nr)
                e = 2 * wr - ww - rr
                out.append(_safe(max(e, 0.0) * math.sqrt(m)))
        return out


class MMDFast(MMDRbfFamily):
    def init(self, hist):
        super().init(hist)
        s2 = self.sigma2
        self.krr = {ch: np.exp(-(self.refsub[ch][:, None] - self.refsub[ch][None, :]) ** 2 / (2 * s2)).mean()
                    for ch in ("z", "e")}

    def update(self, ctx):
        out = []
        s2 = self.sigma2
        for ch in self.channels:
            rs = self.refsub[ch]
            for W in WINDOWS:
                w = ctx.sorted_window(ch, W)
                m = w.size
                if m < 8:
                    out.append(0.0)
                    continue
                kww = np.exp(-(w[:, None] - w[None, :]) ** 2 / (2 * s2)).mean()
                kwr = np.exp(-(w[:, None] - rs[None, :]) ** 2 / (2 * s2)).mean()
                out.append(_safe(max(kww + self.krr[ch] - 2 * kwr, 0.0) * m))
        return out


class EntropyFast(DiffEntropyFamily):
    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        return (_vasicek_fast(w) - self.ref_h[self._cur_ch]) * math.sqrt(m)


FAST_OVERRIDES = {
    "ftest": FTestFast, "bartlett": BartlettFast, "levene": LeveneFast, "fligner": FlignerFast,
    "lmom": LMomentsFast, "iqr": IQRFast, "ansari": AnsariFast, "mood": MoodFast,
    "sigtuk": SiegelTukeyFast, "wass": WassersteinFast, "energy": EnergyFast, "mmd": MMDFast,
    "entropy": EntropyFast,
}


# ---------------------------------------------------------------------------
# Faster driver: numpy ring buffer, only the channels in use, no per-step names()
# ---------------------------------------------------------------------------
class ChannelStateFast(ChannelState):
    def __init__(self):
        super().__init__()
        self.nbuf = np.zeros(LBUF, dtype=np.float64)

    def push(self, v, t):
        super().push(v, t)
        self.nbuf[t % LBUF] = v

    def window(self, W, t):
        n = t + 1
        m = W if n > W else n
        i = t % LBUF
        s = i - m + 1
        if s >= 0:
            return self.nbuf[s:i + 1].copy()
        return np.concatenate([self.nbuf[s % LBUF:], self.nbuf[:i + 1]])


class CtxFast(Ctx):
    __slots__ = ("_var_cache",)

    def __init__(self, *a):
        super().__init__(*a)
        self._var_cache = {}

    def window_var(self, ch, W):
        key = (ch, W)
        v = self._var_cache.get(key)
        if v is None:
            w = self.sorted_window(ch, W)
            v = self._var_cache[key] = max(float(w.var(ddof=1)), 1e-12) if w.size >= 2 else 1e-6
        return v


# which ChannelState each family reads (window families read "e" via TWOSAMPLE_CHANNELS)
_FAMILY_CHANNELS = {"e2": {"e2"}, "z2": {"z2"}, "e": {"e"}, "z": {"z"},
                    "contrast": {"z", "e"}, "slope": {"z"}}


class StreamFeaturesFast:
    def __init__(self, x_hist, active_families=None, registry=None):
        reg = registry if registry is not None else REGISTRY
        h = self.h = HistoricalRef(x_hist)
        self.families = []
        for f in reg:
            if active_families is not None and f.name not in active_families:
                continue
            cls = FAST_OVERRIDES.get(f.name)
            fam = cls() if cls is not None else f.clone()
            fam.init(h)
            self.families.append(fam)
        self.names = [nm for f in self.families for nm in f.names()]
        self._nout = [len(f.names()) for f in self.families]
        need = set()
        for f in self.families:
            if f.name in _FAMILY_CHANNELS:
                need |= _FAMILY_CHANNELS[f.name]
            elif f.cost != "O1" or isinstance(f, TwoSampleFamily):
                need |= set(getattr(f, "channels", ()) or ())
        self.channels = {c: ChannelStateFast() for c in CH if c in need}
        self.tx = list(h.tail_x)
        self.t = -1

    def update(self, x, emit=True):
        h = self.h
        self.t += 1
        t = self.t
        n = t + 1
        xv = float(x)
        if xv != xv or xv in (float("inf"), float("-inf")):
            xv = h.mu
        j = bisect.bisect_left(h.xs_l, xv)
        z = h.ztab[j]
        u = (j + 0.5) / (h.n + 1.0)
        zr = (xv - h.mu) / h.sd
        tx = self.tx
        rres = xv - h.mu
        for k in range(AR_P):
            rres -= h.ar_l[k] * (tx[AR_P - 1 - k] - h.mu)
        tx.pop(0)
        tx.append(xv)
        ze = h.etab[bisect.bisect_left(h.es_l, rres)]
        vals = {"z": z, "z2": (z * z - 1.0) * INV_SQRT2, "e": ze, "e2": (ze * ze - 1.0) * INV_SQRT2}
        chs = self.channels
        for name, cs in chs.items():
            cs.push(vals[name], t)
        ctx = CtxFast(t, n, vals, zr, u, chs, h)
        out = []
        for f in self.families:
            stride = stride_for(f.cost, t)
            if stride == 1 or t % stride == 0 or not hasattr(f, "_last"):
                f._last = f.update(ctx)
            out.extend(f._last)
        return out


def check_fast_equivalence(datasets, active_families, n_series=20, atol=1e-8, rtol=1e-8):
    names_ref = None
    worst = 0.0
    for did, xh, xo, tau in list(datasets)[:n_series]:
        a = StreamFeatures(xh, active_families=active_families)
        b = StreamFeaturesFast(xh, active_families=active_families)
        assert a.names == b.names, "name/order mismatch"
        names_ref = a.names
        for step, v in enumerate(xo):
            ra = np.asarray(a.update(v)); rb = np.asarray(b.update(v))
            d = np.abs(ra - rb)
            worst = max(worst, float(d.max()))
            if not np.allclose(ra, rb, atol=atol, rtol=rtol):
                k = int(np.argmax(d))
                raise AssertionError("series %s step %d %s: orig=%.12g fast=%.12g"
                                     % (did, step, names_ref[k], ra[k], rb[k]))
    print("FAST EQUIVALENCE OK over %d series, max abs diff %.2e" % (n_series, worst))
