"""Frozen reference implementations -- EQUIVALENCE TESTING ONLY.

Snapshot of the notebook's own code as it stood after the Phase 1 registry cull and
BEFORE the Phase 2 (fast families) / Phase 3 (fast metrics) rewrites. Every block
below is byte-identical to what the notebook contained at that point.

Nothing on the crunch submission path imports this. It exists so
tools/test_phase2.py and tools/test_phase3.py can diff the optimized notebook code
against the arithmetic it replaced.

    from nb_exec import run_notebook
    from reference_impl import build_reference
    ns  = run_notebook()          # the live (optimized) notebook
    ref = build_reference(ns)     # the frozen reference, sharing ns's config
    ref["StreamFeatures"]         # the pre-Phase-2 driver
    ref["ts_auc"]                 # the pre-Phase-3 sklearn metric

build_reference() layers the snapshot over a COPY of the notebook namespace, so the
reference families reuse the notebook's HistoricalRef, config constants (WINDOWS,
ALPHAS, ...), FeatureFamily base, _safe, _l_moments and stride_for -- the things
Phases 2-3 do not touch -- while overriding everything they do touch.
"""

# flake8: noqa
import bisect
import copy
import math
from typing import List

import numpy as np
import pandas as pd
from scipy import special as _sp
from sklearn.metrics import roc_auc_score


def ndtri(x):
    return _sp.ndtri(x)


REFERENCE_SOURCE = r'''
class ChannelState:
    """Ring buffer + exp/EWMA/window/CUSUM incremental trackers for one channel.
    Owned by the driver, pushed once per step; any family needing this channel
    reads it off `ctx` instead of recomputing it, rather than each family keeping its
    own ring buffer over the same values."""

    def __init__(self):
        L = LBUF
        self.buf = [0.0] * L
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
        idx = [(i - m + 1 + k) % L for k in range(m)]
        return np.asarray([self.buf[k] for k in idx], dtype=np.float64)


class Ctx:
    __slots__ = ("t", "n", "vals", "zr", "u", "channels", "h", "_sort_cache")

    def __init__(self, t, n, vals, zr, u, channels, h):
        self.t = t
        self.n = n
        self.vals = vals              # dict: "z"->float, "z2"->..., "e"->..., "e2"->...
        self.zr = zr
        self.u = u
        self.channels = channels      # dict: "z"->ChannelState, ...
        self.h = h
        self._sort_cache = {}

    def sorted_window(self, channel: str, W: int) -> np.ndarray:
        key = (channel, W)
        c = self._sort_cache.get(key)
        if c is None:
            w = self.channels[channel].window(W, self.t)
            c = self._sort_cache[key] = np.sort(w)
        return c


class ChannelBankFamily(FeatureFamily):
    """exp / EWMA(3) / window(4) / CUSUM(3) bank for one channel."""

    def __init__(self, channel: str):
        self.channel = channel
        self.name = channel
        self.channels = (channel,)

    def names(self):
        p = self.channel
        n = [p + "_exp"]
        n += ["%s_ew%g" % (p, a) for a in ALPHAS]
        n += ["%s_w%d" % (p, W) for W in WINDOWS]
        n += ["%s_cusum%g" % (p, d) for d in DRIFTS]
        return n

    def update(self, ctx: Ctx):
        cs = ctx.channels[self.channel]
        n = ctx.n
        out = [cs.exp_s / math.sqrt(n)]
        for k in range(len(ALPHAS)):
            out.append(cs.ew_m[k] / math.sqrt(cs.ew_w2[k] + 1e-18))
        for k in range(len(WINDOWS)):
            nw = WINDOWS[k] if n > WINDOWS[k] else n
            out.append(cs.rs[k] / math.sqrt(nw))
        for k in range(len(DRIFTS)):
            out.append(cs.gp[k] if cs.gp[k] > cs.gn[k] else cs.gn[k])
        return out


class TailFamily(FeatureFamily):
    name = "tail"
    channels = ("z",)

    def init(self, hist: HistoricalRef):
        L = LBUF
        self.t2buf = [0.0] * L
        self.t3buf = [0.0] * L
        self.t2w = self.t3w = self.t2e = self.t3e = 0.0
        self.beyond = 0.0
        self.max_absz = 0.0
        self.p2, self.p3 = hist.p2, hist.p3
        self.xmin, self.xmax, self.hn = hist.xmin, hist.xmax, hist.n
        self.mu, self.sd = hist.mu, hist.sd

    def names(self):
        out = []
        for thr in (2, 3):
            out += ["tail%d_w%d" % (thr, TAIL_W), "tail%d_exp" % thr]
        out += ["beyond_range_exp", "max_absz_exp"]
        return out

    def update(self, ctx: Ctx):
        zr = ctx.zr
        t, n = ctx.t, ctx.n
        L = LBUF
        i = t % L
        az = zr if zr >= 0.0 else -zr
        e2 = 1.0 if az > 2.0 else 0.0
        e3 = 1.0 if az > 3.0 else 0.0
        self.t2w += e2 - self.t2buf[(i - TAIL_W) % L]
        self.t3w += e3 - self.t3buf[(i - TAIL_W) % L]
        self.t2buf[i] = e2
        self.t3buf[i] = e3
        self.t2e += e2
        self.t3e += e3
        nw = TAIL_W if n > TAIL_W else n
        out = []
        for cnt_w, cnt_e, p in ((self.t2w, self.t2e, self.p2), (self.t3w, self.t3e, self.p3)):
            out.append((cnt_w - nw * p) / math.sqrt(nw * p * (1.0 - p) + 1e-12))
            out.append((cnt_e - n * p) / math.sqrt(n * p * (1.0 - p) + 1e-12))
        xv = self.mu + zr * self.sd
        if xv < self.xmin or xv > self.xmax:
            self.beyond += 1.0
        pb_rate = 2.0 / (self.hn + 1.0)
        out.append((self.beyond - n * pb_rate) / math.sqrt(n * pb_rate * (1.0 - pb_rate) + 1e-12))
        if az > self.max_absz:
            self.max_absz = az
        out.append(self.max_absz)
        return out


class TimeFamily(FeatureFamily):
    """Pure functions of t. Structurally 0.5 TS-AUC alone -- kept always, exempt
    from every prune/ablate rule (only useful through interactions)."""
    name = "time"
    channels = ()

    def names(self):
        return ["t_step", "t_log"]

    def update(self, ctx: Ctx):
        return [float(ctx.t), math.log1p(ctx.t)]


def _mean_var(x: np.ndarray):
    n = x.size
    if n < 2:
        return 0.0, 1e-6
    mu = float(x.mean())
    v = float(x.var(ddof=1))
    return mu, max(v, 1e-12)


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    neq = np.empty(n, dtype=bool)
    neq[0] = True
    if n > 1:
        np.not_equal(s[1:], s[:-1], out=neq[1:])
    starts = np.flatnonzero(neq)
    grp = np.cumsum(neq) - 1
    counts = np.diff(np.append(starts, n))
    avg = starts.astype(np.float64) + (counts.astype(np.float64) + 1.0) / 2.0
    out = np.empty(n, dtype=np.float64)
    out[order] = avg[grp]
    return out


def _guard(m: int, nr: int, min_m: int = 4) -> bool:
    return m >= min_m and nr >= min_m


class RefSub:
    """Fixed-size random subsample of a reference array, used by O(m*n_ref)
    pairwise-cost families (Tier 3 energy distance / MMD) so a large n_H
    doesn't blow up the per-step cost."""
    CAP = 192

    def __init__(self, ref: np.ndarray, seed: int = 0):
        if ref.size <= self.CAP:
            self.arr = ref
        else:
            rng = np.random.default_rng(seed)
            self.arr = rng.choice(ref, size=self.CAP, replace=False)


class TwoSampleFamily(FeatureFamily):
    cost = "sorted"
    needs_sorted = True
    channels = TWOSAMPLE_CHANNELS      # <-- add this line
    def names(self):
        return ["%s_%s_w%d" % (self.name, ch, W) for ch in self.channels for W in WINDOWS]

    def init(self, hist: HistoricalRef):
        self.ref = {"z": hist.z_ref, "e": hist.e_ref}

    def _window(self, ctx: Ctx, ch: str, W: int) -> np.ndarray:
        if self.needs_sorted:
            return ctx.sorted_window(ch, W)
        return ctx.channels[ch].window(W, ctx.t)

    def stat(self, w: np.ndarray, r: np.ndarray) -> float:
        raise NotImplementedError

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            ref = self.ref[ch]
            for W in WINDOWS:
                w = self._window(ctx, ch, W)
                out.append(_safe(self.stat(w, ref)))
        return out


class FTestFamily(TwoSampleFamily):
    name = "ftest"
    needs_sorted = False

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        _, v1 = _mean_var(w)
        _, v2 = _mean_var(r)
        logf = math.log(v1 / v2)
        se = math.sqrt(2.0 / (m - 1) + 2.0 / (nr - 1))
        return logf / se if se > 0 else 0.0


class BartlettFamily(TwoSampleFamily):
    name = "bartlett"
    needs_sorted = False

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        _, v1 = _mean_var(w)
        _, v2 = _mean_var(r)
        sp2 = ((m - 1) * v1 + (nr - 1) * v2) / (m + nr - 2)
        b = (m + nr - 2) * math.log(sp2) - (m - 1) * math.log(v1) - (nr - 1) * math.log(v2)
        c = 1.0 + (1.0 / 3.0) * (1.0 / (m - 1) + 1.0 / (nr - 1) - 1.0 / (m + nr - 2))
        chi2_1 = b / c if c > 0 else 0.0
        return (chi2_1 - 1.0) / math.sqrt(2.0)


class LeveneFamily(TwoSampleFamily):
    """Brown-Forsythe variant (median-centered) -- robust to non-normality."""
    name = "levene"
    needs_sorted = False

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        zw = np.abs(w - np.median(w))
        zr = np.abs(r - np.median(r))
        zbar_w, zbar_r = zw.mean(), zr.mean()
        zbar = (zw.sum() + zr.sum()) / (m + nr)
        ssb = m * (zbar_w - zbar) ** 2 + nr * (zbar_r - zbar) ** 2
        ssw = float(((zw - zbar_w) ** 2).sum() + ((zr - zbar_r) ** 2).sum())
        df2 = m + nr - 2
        if ssw <= 0 or df2 <= 0:
            return 0.0
        f = ssb / (ssw / df2)
        sign = 1.0 if zbar_w >= zbar_r else -1.0
        return sign * math.sqrt(max(f, 0.0))


class FlignerKilleenFamily(TwoSampleFamily):
    """Rank-based (normal-scores) scale test -- distribution-free, robust to tails."""
    name = "fligner"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        aw = np.abs(w - np.median(w))
        ar = np.abs(r - np.median(r))
        combined = np.concatenate([aw, ar])
        n = combined.size
        ranks = _avg_ranks(combined)
        scores = ndtri(0.5 + ranks / (2.0 * (n + 1)))
        sum_w = scores[:m].sum()
        mean_score = scores.mean()
        var_score = scores.var(ddof=1) if n > 1 else 1e-6
        mean = m * mean_score
        var = m * nr / n * var_score if n > 0 else 1e-6
        return (sum_w - mean) / math.sqrt(max(var, 1e-9))


class LMomentsFamily(FeatureFamily):
    """L-scale, L-skewness, L-kurtosis vs. the reference grid's L-moments."""
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
            t3r = l3r / l2r
            t4r = l4r / l2r
            for W in WINDOWS:
                w = ctx.sorted_window(ch, W)
                m = w.size
                if m < 8:
                    out += [0.0, 0.0, 0.0]
                    continue
                l1, l2, l3, l4 = _l_moments(w)
                out.append(_safe(math.log(max(l2, 1e-9) / l2r) * math.sqrt(m - 1)))
                out.append(_safe((l3 / l2 - t3r) * math.sqrt(m)))
                out.append(_safe((l4 / l2 - t4r) * math.sqrt(m)))
        return out


class IQRRatioFamily(TwoSampleFamily):
    name = "iqr"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_iqr = {ch: float(np.subtract(*np.percentile(self.ref[ch], [75, 25])))
                         for ch in ("z", "e")}

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        ch = self._cur_ch
        q75, q25 = np.percentile(w, [75, 25])
        iqr_w = float(q75 - q25)
        iqr_r = self.ref_iqr[ch]
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


class AnsariBradleyFamily(TwoSampleFamily):
    """Rank-based scale test: score = min(rank, N+1-rank) (ends-inward ranking)."""
    name = "ansari"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        n = m + nr
        combined = np.concatenate([w, r])
        ranks = _avg_ranks(combined)
        scores = np.minimum(ranks, n + 1 - ranks)
        ab = scores[:m].sum()
        if n % 2 == 0:
            e = m * (n + 2) / 4.0
            var = m * nr * (n + 2) * (n - 2) / (48.0 * (n - 1))
        else:
            e = m * (n + 1) ** 2 / (4.0 * n)
            var = m * nr * (n + 1) * (3 + n * n) / (48.0 * n * n)
        return (ab - e) / math.sqrt(var) if var > 0 else 0.0


class MoodFamily(TwoSampleFamily):
    """Rank-based scale test: score = (rank - (N+1)/2)^2."""
    name = "mood"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        n = m + nr
        combined = np.concatenate([w, r])
        ranks = _avg_ranks(combined)
        scores = (ranks - (n + 1) / 2.0) ** 2
        t = scores[:m].sum()
        e = m * (n * n - 1) / 12.0
        var = m * nr * (n + 1) * (n * n - 4) / 180.0
        return (t - e) / math.sqrt(var) if var > 0 else 0.0


class SiegelTukeyFamily(TwoSampleFamily):
    """Ends-alternating ('zig-zag') ranking, then a Wilcoxon rank-sum on it."""
    name = "sigtuk"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        n = m + nr
        combined = np.concatenate([w, r])
        order = np.argsort(combined, kind="mergesort")
        st_rank = np.empty(n, dtype=np.float64)
        n_lo = (n + 1) // 2
        n_hi = n // 2
        st_rank[order[:n_lo]] = 2.0 * np.arange(n_lo) + 1.0          # 1,3,5,... from the low end
        if n_hi:
            st_rank[order[n - 1:n - 1 - n_hi:-1]] = 2.0 * np.arange(n_hi) + 2.0   # 2,4,6,... from the high end
        rank_sum_w = st_rank[:m].sum()
        mean = m * (n + 1) / 2.0
        var = m * nr * (n + 1) / 12.0
        return (rank_sum_w - mean) / math.sqrt(var) if var > 0 else 0.0


class WassersteinFamily(TwoSampleFamily):
    name = "wass"
    cost = "heavy"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr, 8):
            return 0.0
        q = (np.arange(1, m + 1) - 0.5) / m
        rq = np.quantile(r, q)
        w1 = float(np.abs(w - rq).mean())
        return w1 * math.sqrt(m)


class EnergyDistanceFamily(TwoSampleFamily):
    name = "energy"
    cost = "heavy"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.refsub = {ch: RefSub(hist.z_ref if ch == "z" else hist.e_ref).arr for ch in ("z", "e")}

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        rs = r
        wr = np.abs(w[:, None] - rs[None, :]).mean()
        ww = np.abs(w[:, None] - w[None, :]).mean()
        rr = np.abs(rs[:, None] - rs[None, :]).mean()
        e = 2 * wr - ww - rr
        return max(e, 0.0) * math.sqrt(m)

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            for W in WINDOWS:
                w = self._window(ctx, ch, W)
                out.append(_safe(self.stat(w, self.refsub[ch])))
        return out


class MMDRbfFamily(EnergyDistanceFamily):
    name = "mmd"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        allref = np.concatenate([self.refsub["z"], self.refsub["e"]])
        d = np.abs(allref[:, None] - allref[None, :])
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        self.sigma2 = max(float(med) ** 2, 1e-6)

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        rs = r
        s2 = self.sigma2
        kww = np.exp(-(w[:, None] - w[None, :]) ** 2 / (2 * s2)).mean()
        krr = np.exp(-(rs[:, None] - rs[None, :]) ** 2 / (2 * s2)).mean()
        kwr = np.exp(-(w[:, None] - rs[None, :]) ** 2 / (2 * s2)).mean()
        mmd2 = kww + krr - 2 * kwr
        return max(mmd2, 0.0) * m


class DiffEntropyFamily(TwoSampleFamily):
    """Vasicek spacings estimator of differential entropy."""
    name = "entropy"
    cost = "heavy"
    needs_sorted = True

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_h = {ch: self._vasicek(self.ref[ch]) for ch in ("z", "e")}

    @staticmethod
    def _vasicek(sorted_x):
        m = sorted_x.size
        if m < 8:
            return 0.0
        q = max(1, int(round(math.sqrt(m))))
        idx = np.arange(m)
        lo = np.clip(idx - q, 0, m - 1)
        hi = np.clip(idx + q, 0, m - 1)
        diffs = sorted_x[hi] - sorted_x[lo]
        diffs = np.maximum(diffs, 1e-9)
        return float(np.mean(np.log(m / (2.0 * q) * diffs)))

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        ch = self._cur_ch
        h_w = self._vasicek(w)
        h_r = self.ref_h[ch]
        return (h_w - h_r) * math.sqrt(m)

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                w = ctx.sorted_window(ch, W)
                out.append(_safe(self.stat(w, self.ref[ch])))
        return out


class StreamFeatures:
    """Same per-step cost model as before (O(1) for the ported bank; strided
    O(W) / O(W log W) for the new Tier 1-4 families). `active_families=None`
    (default) runs every family in REGISTRY; pass a set of family `.name`s to
    restrict to a subset -- this is how `infer()` and the crunch-submission
    `train()` stay fast (compute ACTIVE_FAMILIES only), while the offline
    iteration loop always extracts with active_families=None (full registry).

    Each family is `.clone()`-d from its REGISTRY prototype (see the note on
    FeatureFamily.clone()) so this instance's per-series state is fully its
    own -- required for build_dataset()'s threaded parallelism to be safe."""

    def __init__(self, x_hist, active_families=None):
        h = self.h = HistoricalRef(x_hist)
        self.families = [f.clone() for f in REGISTRY
                          if active_families is None or f.name in active_families]
        for f in self.families:
            f.init(h)
        self.names = [nm for f in self.families for nm in f.names()]
        self.channels = {name: ChannelState() for name in CH}
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

        vals = {"z": z, "z2": (z * z - 1.0) * INV_SQRT2, "e": ze, "e2": (ze * ze - 1.0) * INV_SQRT2}
        for name, v in vals.items():
            self.channels[name].push(v, t)

        ctx = Ctx(t, n, vals, zr, u, self.channels, h)

        out = []
        for f in self.families:
            stride = stride_for(f.cost, t)
            if stride == 1 or t % stride == 0 or not hasattr(f, "_last"):
                r = f.update(ctx)
                assert len(r) == len(f.names()), (
                    "%s.update() returned %d values, names() has %d"
                    % (f.name, len(r), len(f.names())))
                f._last = r
            else:
                r = f._last
            out.extend(r)
        return out

# ---------------------------------------------------------------------------
#  Reference REGISTRY -- same 17 families, same order as the notebook's
# ---------------------------------------------------------------------------
for _ch in ("z2", "e2"):
    REGISTRY.append(ChannelBankFamily(_ch))
REGISTRY.append(TailFamily())
REGISTRY.append(TimeFamily())
REGISTRY.extend([FTestFamily(), BartlettFamily(), LeveneFamily(), FlignerKilleenFamily()])
REGISTRY.extend([LMomentsFamily(), IQRRatioFamily(),
                 AnsariBradleyFamily(), MoodFamily(), SiegelTukeyFamily()])
REGISTRY.extend([WassersteinFamily(), EnergyDistanceFamily(), MMDRbfFamily()])
REGISTRY.extend([DiffEntropyFamily()])
FEATURES = [nm for f in REGISTRY for nm in f.names()]


def ts_auc(y, t, score):
    y = np.asarray(y); t = np.asarray(t); score = np.asarray(score, dtype=np.float64)
    order = np.argsort(t, kind="mergesort")
    y, t, score = y[order], t[order], score[order]
    bounds = np.flatnonzero(np.diff(t)) + 1
    num = den = 0.0
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(t)]):
        yy = y[lo:hi]
        npos = int(yy.sum()); nneg = len(yy) - npos
        if npos == 0 or nneg == 0:
            continue
        w = float(npos * nneg)
        num += w * roc_auc_score(yy, score[lo:hi])
        den += w
    return num / den if den > 0 else 0.5


def _folded_ts_auc_table(X, y, t, feature_names):
    rows = []
    for k, nm in enumerate(feature_names):
        s = np.nan_to_num(X[:, k].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        a = ts_auc(y, t, s)
        rows.append((nm, max(a, 1 - a), a))
    return pd.DataFrame(rows, columns=["feature", "ts_auc_abs", "ts_auc_raw"])


def permutation_null_ts_auc(X, y, t, g, feature_idx, feature_names, n_perm=100, seed=0):
    """Series-level label permutation: shuffle which SERIES are treated as
    breaking, keep each series' own row/t structure intact. A relabeled
    "breaker" reuses its own original first-break t as a synthetic tau (so the
    permuted series still has a well-defined post-break region); a real
    breaker relabeled non-breaking has all its rows set to y=0. Never shuffle
    rows independently -- that destroys the within-series correlation that is
    exactly why the naive (row-count) null is too tight."""
    rng = np.random.default_rng(seed)
    uniq_series = np.unique(g)
    series_breaks = {}
    for sid in uniq_series:
        series_breaks[sid] = int(y[g == sid].max())
    sids = list(series_breaks.keys())
    breaks_arr = np.array([series_breaks[s] for s in sids])

    null_aucs = {nm: [] for nm in feature_names}
    for _ in range(n_perm):
        perm_breaks = rng.permutation(breaks_arr)
        perm_map = dict(zip(sids, perm_breaks))
        y_perm = np.empty_like(y)
        for sid in uniq_series:
            mask = g == sid
            if perm_map[sid] == 1 and series_breaks[sid] == 1:
                y_perm[mask] = y[mask]
            elif perm_map[sid] == 1 and series_breaks[sid] == 0:
                tt = t[mask]
                cut = tt[len(tt) // 2] if len(tt) else 0
                y_perm[mask] = (tt >= cut).astype(y.dtype)
            else:
                y_perm[mask] = 0
        for j, nm in zip(feature_idx, feature_names):
            s = np.nan_to_num(X[:, j].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            a = ts_auc(y_perm, t, s)
            null_aucs[nm].append(max(a, 1 - a))

    rows = [(nm, float(np.mean(v)), float(np.percentile(v, 95))) for nm, v in null_aucs.items()]
    return pd.DataFrame(rows, columns=["feature", "null_mean", "null_p95"])

'''


def build_reference(ns):
    """Exec the frozen snapshot over a copy of the notebook namespace `ns`."""
    refns = dict(ns)
    refns["REGISTRY"] = []
    refns["ndtri"] = ndtri
    refns["roc_auc_score"] = roc_auc_score
    refns["np"] = np
    refns["pd"] = pd
    refns["math"] = math
    refns["bisect"] = bisect
    refns["copy"] = copy
    refns["List"] = List
    exec(compile(REFERENCE_SOURCE, "reference_impl.REFERENCE_SOURCE", "exec"), refns)
    return refns
