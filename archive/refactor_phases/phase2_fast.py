"""PHASE 2 -- fold fast_families.py into the notebook.

Run AFTER tools/make_reference_impl.py (which freezes the arithmetic being replaced).

Each fast class takes over the NAME of the family it replaces, so there is exactly one
class per family in the notebook and `REGISTRY` / `names()` / feature order are all
unchanged. `StreamFeatures`, `ChannelState` and `Ctx` keep their public names too, with
the fast bodies folded in -- `extract_series`, `infer()` and the parity check are
untouched.

Two deliberate changes vs. fast_families.py as delivered:

  * channel selection. The delivered driver picked channels from a hand-written
    `_FAMILY_CHANNELS` dict plus an `elif f.cost != "O1" or isinstance(f, TwoSampleFamily)`
    fallback. That silently skips the declared `channels` of any O(1) non-two-sample
    family, so a new O(1) family reading `ctx.channels[...]` would KeyError. Since
    `ChannelBankFamily` already sets `self.channels = (channel,)`, the union of every
    active family's declared `channels` gives the same answer for today's registry and
    stays correct for tomorrow's.

  * ansari/mood share their tie-averaged window ranks via `Ctx.window_ranks` (task 2.3).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell


CHANNEL_STATE = '''class ChannelState:
    """Ring buffer + exp/EWMA/window/CUSUM incremental trackers for one channel.
    Owned by the driver, pushed once per step; any family needing this channel
    reads it off `ctx` instead of recomputing it.

    Two copies of the ring buffer are kept on purpose. `buf` is a Python list: the
    running-sum update reads one scalar per window per step, and a list index is
    cheaper than a numpy 0-d unbox. `abuf` is the numpy mirror that `window()`
    slices, which replaces a 256-element list comprehension per window per step."""

    def __init__(self):
        L = LBUF
        self.buf = [0.0] * L
        self.abuf = np.zeros(L, dtype=np.float64)
        self.rs = [0.0] * len(WINDOWS)
        self.exp_s = 0.0
        self.ew_m = [0.0] * len(ALPHAS)
        self.ew_w2 = [0.0] * len(ALPHAS)
        self.gp = [0.0] * len(DRIFTS)
        self.gn = [0.0] * len(DRIFTS)

    def push(self, v: float, t: int) -> None:
        L = LBUF
        i = t % L
        b = self.buf
        rs = self.rs
        for k in range(len(WINDOWS)):
            rs[k] += v - b[(i - WINDOWS[k]) % L]
        b[i] = v
        self.abuf[i] = v
        self.exp_s += v
        m, w2 = self.ew_m, self.ew_w2
        for k in range(len(ALPHAS)):
            a = ALPHAS[k]
            m[k] = (1.0 - a) * m[k] + a * v
            w2[k] = (1.0 - a) * (1.0 - a) * w2[k] + a * a
        gp, gn = self.gp, self.gn
        for k in range(len(DRIFTS)):
            d = DRIFTS[k]
            p = gp[k] + v - d
            gp[k] = p if p > 0.0 else 0.0
            q = gn[k] - v - d
            gn[k] = q if q > 0.0 else 0.0

    def window(self, W: int, t: int) -> np.ndarray:
        """Last min(W, t+1) raw pushed values, chronological order."""
        n = t + 1
        m = W if n > W else n
        L = LBUF
        i = t % L
        s = i - m + 1
        if s >= 0:
            return self.abuf[s:i + 1].copy()
        return np.concatenate([self.abuf[s % L:], self.abuf[:i + 1]])
'''

CTX = '''class Ctx:
    __slots__ = ("t", "n", "vals", "zr", "u", "channels", "h",
                 "_sort_cache", "_var_cache", "_rank_cache")

    def __init__(self, t, n, vals, zr, u, channels, h):
        self.t = t
        self.n = n
        self.vals = vals              # dict: "z"->float, "z2"->..., "e"->..., "e2"->...
        self.zr = zr
        self.u = u
        self.channels = channels      # dict: "z"->ChannelState, ...
        self.h = h
        self._sort_cache = {}
        self._var_cache = {}
        self._rank_cache = {}

    def sorted_window(self, channel: str, W: int) -> np.ndarray:
        key = (channel, W)
        c = self._sort_cache.get(key)
        if c is None:
            w = self.channels[channel].window(W, self.t)
            c = self._sort_cache[key] = np.sort(w)
        return c

    def window_var(self, channel: str, W: int) -> float:
        """Sample variance (ddof=1) of the trailing window, memoized for the step.
        ftest and bartlett both want it; it matches _mean_var()'s guards exactly."""
        key = (channel, W)
        v = self._var_cache.get(key)
        if v is None:
            w = self.sorted_window(channel, W)
            v = self._var_cache[key] = (max(float(w.var(ddof=1)), 1e-12)
                                        if w.size >= 2 else 1e-6)
        return v

    def window_ranks(self, channel: str, W: int, ref_sorted: np.ndarray) -> np.ndarray:
        """Tie-averaged ranks of the sorted trailing window inside (window ++ reference),
        memoized for the step. `ansari` and `mood` compute exactly this, off the same
        reference grid, so they share one computation.

        Keyed on (channel, W) only, which is safe because every caller is a
        RefCachedTwoSample and they all read the SAME reference (h.z_ref / h.e_ref via
        TwoSampleFamily.init). A family with its own reference -- `fligner` ranks
        |x - median| instead -- must not use this cache, and does not."""
        key = (channel, W)
        r = self._rank_cache.get(key)
        if r is None:
            w = self.sorted_window(channel, W)
            r = self._rank_cache[key] = _avg_ranks_of(w, w, ref_sorted)
        return r
'''

FAST_HELPERS = '''# =====================================================================
#  Exact fast paths shared by the two-sample families (Phase 2)
#
#  Every one of these is an algebraic rewrite, not an approximation: the only
#  difference from the code they replace is floating-point summation order.
#  `tools/reference_impl.py` holds the arithmetic they replaced and
#  tools/test_phase2.py diffs them at atol=rtol=1e-8.
# =====================================================================
_COEF_CACHE = {}


def _sorted_median(s: np.ndarray) -> float:
    """np.median(x) for an already-sorted 1-D array, without the internal partition."""
    m = s.size
    return 0.5 * (s[(m - 1) // 2] + s[m // 2])


def _sorted_percentile(s: np.ndarray, q: float) -> float:
    """np.percentile(s, q) (linear interpolation) for an already-sorted 1-D array."""
    pos = (s.size - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, s.size - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _avg_ranks_of(x_sorted_self: np.ndarray, x: np.ndarray,
                  other_sorted: np.ndarray) -> np.ndarray:
    """Average (tie-corrected, 1-based) ranks of `x` within the combined sample
    (x_sorted_self ++ other_sorted), by searchsorted instead of concatenate+argsort.

    rank = (#strictly-less + 1 + #less-or-equal) / 2, which is the midpoint of the
    tie block -- identical to _avg_ranks() on the concatenation."""
    less = np.searchsorted(x_sorted_self, x, "left") + np.searchsorted(other_sorted, x, "left")
    leq = np.searchsorted(x_sorted_self, x, "right") + np.searchsorted(other_sorted, x, "right")
    return 0.5 * (less + 1 + leq)


def _lmom_coefs(n: int):
    """Hosking b1/b2/b3 weight vectors for length n, built once per n."""
    c = _COEF_CACHE.get(("lmom", n))
    if c is None:
        i = np.arange(1, n + 1, dtype=np.float64)
        c1 = (i - 1) / (n - 1) / n
        c2 = (i - 1) * (i - 2) / ((n - 1) * (n - 2)) / n
        c3 = (i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) / n
        c = _COEF_CACHE[("lmom", n)] = (c1, c2, c3)
    return c


def _l_moments_fast(x: np.ndarray):
    """_l_moments() with the three weight vectors precomputed (x must be SORTED)."""
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


def _vasicek_fast(sorted_x: np.ndarray) -> float:
    """Vasicek spacings entropy with the clipped index arrays built once per length."""
    m = sorted_x.size
    if m < 8:
        return 0.0
    key = ("vas", m)
    c = _COEF_CACHE.get(key)
    if c is None:
        q = max(1, int(round(math.sqrt(m))))
        idx = np.arange(m)
        c = _COEF_CACHE[key] = (np.clip(idx - q, 0, m - 1),
                                np.clip(idx + q, 0, m - 1), m / (2.0 * q))
    lo, hi, scale = c
    diffs = np.maximum(sorted_x[hi] - sorted_x[lo], 1e-9)
    return float(np.mean(np.log(scale * diffs)))
'''

REF_CACHED_BASE = '''class RefCachedTwoSample(TwoSampleFamily):
    """TwoSampleFamily whose reference-side work is done once in init().

    Subclasses implement `stat_fast(ctx, ch, W)` and pull the window off `ctx`
    themselves, so they can use the memoized sorted window / variance / ranks
    instead of being handed a bare array."""

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_sorted = {}
        for ch in ("z", "e"):
            r = self.ref[ch]
            if np.any(np.diff(r) < 0):
                r = np.sort(r)
            self.ref_sorted[ch] = r

    def stat_fast(self, ctx: Ctx, ch: str, W: int) -> float:
        raise NotImplementedError

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                out.append(_safe(self.stat_fast(ctx, ch, W)))
        return out
'''

TIER1 = '''class FTestFamily(RefCachedTwoSample):
    name = "ftest"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.v2 = {ch: _mean_var(self.ref[ch])[1] for ch in ("z", "e")}

    def stat_fast(self, ctx, ch, W):
        m, nr = ctx.sorted_window(ch, W).size, self.ref[ch].size
        if not _guard(m, nr):
            return 0.0
        v1 = ctx.window_var(ch, W)
        se = math.sqrt(2.0 / (m - 1) + 2.0 / (nr - 1))
        return math.log(v1 / self.v2[ch]) / se if se > 0 else 0.0


class BartlettFamily(FTestFamily):
    name = "bartlett"

    def stat_fast(self, ctx, ch, W):
        m, nr = ctx.sorted_window(ch, W).size, self.ref[ch].size
        if not _guard(m, nr):
            return 0.0
        v1, v2 = ctx.window_var(ch, W), self.v2[ch]
        sp2 = ((m - 1) * v1 + (nr - 1) * v2) / (m + nr - 2)
        b = (m + nr - 2) * math.log(sp2) - (m - 1) * math.log(v1) - (nr - 1) * math.log(v2)
        c = 1.0 + (1.0 / 3.0) * (1.0 / (m - 1) + 1.0 / (nr - 1) - 1.0 / (m + nr - 2))
        return ((b / c if c > 0 else 0.0) - 1.0) / math.sqrt(2.0)


class LeveneFamily(RefCachedTwoSample):
    """Brown-Forsythe variant (median-centered) -- robust to non-normality.
    The reference half of the statistic is constant, so it is precomputed."""
    name = "levene"

    def init(self, hist: HistoricalRef):
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


class FlignerKilleenFamily(RefCachedTwoSample):
    """Rank-based (normal-scores) scale test -- distribution-free, robust to tails.

    Ranks |x - median| over both samples. That is a DIFFERENT ranking from the one
    ansari/mood share, so this family must not use Ctx.window_ranks."""
    name = "fligner"

    def init(self, hist: HistoricalRef):
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
        rk_r = 0.5 * (ar_l + np.searchsorted(aw, ar, "left") + 1
                      + ar_r + np.searchsorted(aw, ar, "right"))
        sw = ndtri(0.5 + rk_w / (2.0 * (n + 1)))
        sr = ndtri(0.5 + rk_r / (2.0 * (n + 1)))
        mean_score = (sw.sum() + sr.sum()) / n
        ss = ((sw - mean_score) ** 2).sum() + ((sr - mean_score) ** 2).sum()
        var = m * nr / n * (ss / (n - 1))
        return (sw.sum() - m * mean_score) / math.sqrt(max(var, 1e-9))
'''

TIER2_LMOM = '''class LMomentsFamily(FeatureFamily):
    """L-scale, L-skewness, L-kurtosis vs. the reference grid's L-moments.

    NOT a TwoSampleFamily: it inherits channels = ("z", "e") from FeatureFamily, so it
    emits 2 channels x 3 statistics x 3 windows = 18 features, and it is why the "z"
    ChannelState is still maintained after the Phase 1 bank cull."""
    name = "lmom"
    cost = "sorted"

    def init(self, hist: HistoricalRef):
        self.ref_lmom = {"z": hist.z_ref_lmom, "e": hist.e_ref_lmom}

    def names(self):
        out = []
        for ch in self.channels:
            for stat in ("lscale", "lskew", "lkurt"):
                for W in WINDOWS:
                    out.append("%s_%s_%s_w%d" % (self.name, stat, ch, W))
        return out

    def update(self, ctx: Ctx):
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
'''

TIER2_IQR = '''class IQRRatioFamily(TwoSampleFamily):
    name = "iqr"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_iqr = {ch: float(np.subtract(*np.percentile(self.ref[ch], [75, 25])))
                         for ch in ("z", "e")}

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        # w arrives sorted (needs_sorted), so the quartiles are two interpolated reads
        iqr_w = float(_sorted_percentile(w, 75) - _sorted_percentile(w, 25))
        iqr_r = self.ref_iqr[self._cur_ch]
        if iqr_w <= 0 or iqr_r <= 0:
            return 0.0
        return math.log(iqr_w / iqr_r) * math.sqrt(max(m - 1, 1))

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                w = self._window(ctx, ch, W)
                out.append(_safe(self.stat(w, self.ref[ch])))
        return out
'''

TIER2_RANK = '''class _RankScaleFamily(RefCachedTwoSample):
    """Shared base for the rank-based scale tests that rank the window against the
    reference grid. `ansari` and `mood` differ only in the score applied to the ranks,
    so the ranks themselves are computed once per (channel, W) on the Ctx."""

    def w_ranks(self, ctx, ch, W):
        return ctx.sorted_window(ch, W), ctx.window_ranks(ch, W, self.ref_sorted[ch])


class AnsariBradleyFamily(_RankScaleFamily):
    """Rank-based scale test: score = min(rank, N+1-rank) (ends-inward ranking)."""
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


class MoodFamily(_RankScaleFamily):
    """Rank-based scale test: score = (rank - (N+1)/2)^2."""
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


class SiegelTukeyFamily(RefCachedTwoSample):
    """Ends-alternating ('zig-zag') ranking, then a Wilcoxon rank-sum on it.

    The zig-zag rank depends on the element's EXACT position under a stable sort of
    concat([window, reference]), not on its tie-averaged rank. For a sorted window,
    element i sits at position i + (number of reference values strictly below it):
    stability puts window values ahead of equal reference values, and ahead of each
    other in their existing order."""
    name = "sigtuk"

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        r = self.ref_sorted[ch]
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        n = m + nr
        n_lo = (n + 1) // 2
        pos = np.arange(m) + np.searchsorted(r, w, "left")
        st = np.where(pos < n_lo, 2.0 * pos + 1.0, 2.0 * (n - 1 - pos) + 2.0)
        mean = m * (n + 1) / 2.0
        var = m * nr * (n + 1) / 12.0
        return (st.sum() - mean) / math.sqrt(var) if var > 0 else 0.0
'''

TIER3 = '''class WassersteinFamily(RefCachedTwoSample):
    name = "wass"
    cost = "heavy"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self._qcache = {}

    def stat_fast(self, ctx, ch, W):
        w = ctx.sorted_window(ch, W)
        m, nr = w.size, self.ref[ch].size
        if not _guard(m, nr, 8):
            return 0.0
        # the reference quantiles depend only on (channel, window length), and there
        # are at most len(WINDOWS) distinct lengths per series
        key = (ch, m)
        rq = self._qcache.get(key)
        if rq is None:
            q = (np.arange(1, m + 1) - 0.5) / m
            rq = self._qcache[key] = np.quantile(self.ref[ch], q)
        return float(np.abs(w - rq).mean()) * math.sqrt(m)


class EnergyDistanceFamily(TwoSampleFamily):
    """Two-sample energy distance, O(m log m) instead of O(m * n_ref).

    For SORTED x, sum_ij |x_i - x_j| = 2 * sum_i (2i - m + 1) x_i, so the within-window
    term is one dot product. The cross term uses the reference prefix sums: for each
    w_i with k reference values <= it,
        sum_j |w_i - r_j| = w_i*k - cs[k] + (tot - cs[k]) - w_i*(nr - k).
    The reference-vs-reference term is constant and computed once in init()."""
    name = "energy"
    cost = "heavy"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.refsub = {ch: RefSub(hist.z_ref if ch == "z" else hist.e_ref).arr
                       for ch in ("z", "e")}
        self.pre = {}
        for ch in ("z", "e"):
            sub = self.refsub[ch]
            rs = np.sort(sub)
            rr = np.abs(sub[:, None] - sub[None, :]).mean()
            self.pre[ch] = (rs, np.concatenate([[0.0], np.cumsum(rs)]), rr)
        self._coef = {}

    def update(self, ctx: Ctx):
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
                out.append(_safe(max(2 * wr - ww - rr, 0.0) * math.sqrt(m)))
        return out


class MMDRbfFamily(EnergyDistanceFamily):
    """MMD^2 with an RBF kernel. The reference-vs-reference kernel mean is constant,
    so only the window-window and window-reference blocks are built per step.

    The bandwidth is the median pairwise distance over BOTH channels' reference
    subsamples concatenated, so the "z" RefSub is load-bearing even when
    TWOSAMPLE_CHANNELS is ("e",) only."""
    name = "mmd"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        allref = np.concatenate([self.refsub["z"], self.refsub["e"]])
        d = np.abs(allref[:, None] - allref[None, :])
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        self.sigma2 = max(float(med) ** 2, 1e-6)
        s2 = self.sigma2
        self.krr = {ch: float(np.exp(-(self.refsub[ch][:, None] - self.refsub[ch][None, :]) ** 2
                                      / (2 * s2)).mean())
                    for ch in ("z", "e")}

    def update(self, ctx: Ctx):
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
'''

TIER4 = '''class DiffEntropyFamily(TwoSampleFamily):
    """Vasicek spacings estimator of differential entropy."""
    name = "entropy"
    cost = "heavy"
    needs_sorted = True

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_h = {ch: _vasicek_fast(np.sort(self.ref[ch])) for ch in ("z", "e")}

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        return (_vasicek_fast(w) - self.ref_h[self._cur_ch]) * math.sqrt(m)

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                w = ctx.sorted_window(ch, W)
                out.append(_safe(self.stat(w, self.ref[ch])))
        return out
'''

DRIVER = '''class StreamFeatures:
    """Registry driver. `active_families=None` (default) runs every family in
    REGISTRY; pass a set of family `.name`s to restrict to a subset.

    Each family is `.clone()`-d from its REGISTRY prototype (see the note on
    FeatureFamily.clone()) so this instance's per-series state is fully its own --
    required for build_dataset()'s process/thread parallelism to be safe.

    Phase 2 changes, all of them exact:
      * only the ChannelStates some active family declares are built and pushed;
      * per-family output width is captured once in __init__ instead of calling
        f.names() and asserting on every step (the check moved to __init__);
      * every reference-side quantity is precomputed in each family's init().
    """

    def __init__(self, x_hist, active_families=None, registry=None):
        reg = REGISTRY if registry is None else registry
        h = self.h = HistoricalRef(x_hist)
        self.families = [f.clone() for f in reg
                          if active_families is None or f.name in active_families]
        for f in self.families:
            f.init(h)
        self.names = [nm for f in self.families for nm in f.names()]
        self._nout = [len(f.names()) for f in self.families]

        # Build only the channels some active family declares it reads. The union of
        # declared `channels` is the safe rule: ChannelBankFamily sets its own
        # .channels in __init__, TwoSampleFamily inherits TWOSAMPLE_CHANNELS, and
        # LMomentsFamily keeps the FeatureFamily default ("z", "e") -- which is what
        # keeps the "z" state alive after the Phase 1 bank cull.
        need = set()
        for f in self.families:
            need |= set(getattr(f, "channels", ()) or ())
        self.channels = {name: ChannelState() for name in CH if name in need}

        self.tx = list(h.tail_x)
        self.t = -1

    def update(self, x) -> List[float]:
        h = self.h
        self.t += 1
        t = self.t
        n = t + 1

        xv = float(x)
        if xv != xv or xv in (float("inf"), float("-inf")):
            xv = h.mu

        # ---- channel 0/1 : PIT of the raw value ----------------------
        j = bisect.bisect_left(h.xs_l, xv)
        z = h.ztab[j]
        u = (j + 0.5) / (h.n + 1.0)
        zr = (xv - h.mu) / h.sd

        # ---- channel 2/3 : PIT of the AR residual --------------------
        tx = self.tx
        rres = xv - h.mu
        for k in range(AR_P):
            rres -= h.ar_l[k] * (tx[AR_P - 1 - k] - h.mu)
        tx.pop(0)
        tx.append(xv)
        je = bisect.bisect_left(h.es_l, rres)
        ze = h.etab[je]

        vals = {"z": z, "z2": (z * z - 1.0) * INV_SQRT2,
                "e": ze, "e2": (ze * ze - 1.0) * INV_SQRT2}
        for name, cs in self.channels.items():
            cs.push(vals[name], t)

        ctx = Ctx(t, n, vals, zr, u, self.channels, h)

        out = []
        for f in self.families:
            stride = stride_for(f.cost, t)
            if stride == 1 or t % stride == 0 or not hasattr(f, "_last"):
                f._last = f.update(ctx)
            out.extend(f._last)
        return out
'''


def main():
    nb = nbedit.load()

    # ---------------------------------------------------------------- 1c cell
    i = find_cell(nb, "class ChannelState:", "code")
    text = src(nb["cells"][i])
    text = nbedit.replace_block(text, "class ChannelState:", CHANNEL_STATE)
    text = nbedit.replace_block(text, "class Ctx:", CTX)
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1f cell
    i = find_cell(nb, "class TwoSampleFamily(FeatureFamily):", "code")
    text = src(nb["cells"][i])
    text = nbedit.insert_before(text, "class TwoSampleFamily(", FAST_HELPERS)
    # ORDER MATTERS: drop the originals BEFORE inserting the replacements. TIER1
    # defines Bartlett/Levene/Fligner too, and block lookup takes the first match --
    # inserting first would make remove_blocks delete the NEW ones and keep the old.
    for hp in ("class BartlettFamily(", "class LeveneFamily(", "class FlignerKilleenFamily("):
        text, _ = nbedit.remove_blocks(text, [hp])
    text = nbedit.replace_block(text, "class FTestFamily(", REF_CACHED_BASE + "\n\n" + TIER1)
    old_head = 'def _avg_ranks(x: np.ndarray) -> np.ndarray:\n'
    assert old_head in text
    text = text.replace(old_head, old_head + '''    """Tie-averaged (1-based) ranks of one array, via argsort over the whole array.

    No family in the pruned registry calls this any more. Every rank-based family
    (fligner, ansari, mood) now ranks with _avg_ranks_of(), which searchsorts into a
    reference that was sorted once in init() instead of concatenating and re-sorting
    window ++ reference on every evaluation; sigtuk needs stable-sort POSITIONS rather
    than averaged ranks and derives them the same way.

    It stays for two reasons: it is what a new family will reach for before there is a
    fast path for it, and it is the definition _avg_ranks_of() is checked against in
    tools/test_phase2.py."""\n''')
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1g cell
    i = find_cell(nb, "class LMomentsFamily(FeatureFamily):", "code")
    text = src(nb["cells"][i])
    text = nbedit.replace_block(text, "class LMomentsFamily(", TIER2_LMOM)
    text = nbedit.replace_block(text, "class IQRRatioFamily(", TIER2_IQR)
    for hp in ("class MoodFamily(", "class SiegelTukeyFamily("):   # originals first
        text, _ = nbedit.remove_blocks(text, [hp])
    text = nbedit.replace_block(text, "class AnsariBradleyFamily(", TIER2_RANK)
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1h cell
    i = find_cell(nb, "class WassersteinFamily(", "code")
    text = src(nb["cells"][i])
    for hp in ("class EnergyDistanceFamily(", "class MMDRbfFamily("):   # originals first
        text, _ = nbedit.remove_blocks(text, [hp])
    text = nbedit.replace_block(text, "class WassersteinFamily(", TIER3)
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1i cell
    i = find_cell(nb, "class DiffEntropyFamily(", "code")
    text = src(nb["cells"][i])
    text = nbedit.replace_block(text, "class DiffEntropyFamily(", TIER4)
    set_src(nb["cells"][i], text)

    # ------------------------------------------------------------ driver cell
    i = find_cell(nb, "class StreamFeatures:", "code")
    text = src(nb["cells"][i])
    text = nbedit.replace_block(text, "class StreamFeatures:", DRIVER)
    set_src(nb["cells"][i], text)

    nbedit.save(nb)
    print("phase 2 applied")


if __name__ == "__main__":
    main()
