"""Families removed from the notebook's REGISTRY in the Phase 1 cull.

These are NOT imported by anything. The notebook is what crunch submits, so nothing
on the train()/infer() path may import a local module -- this file is a parts bin, not
a dependency.

TO RESTORE ONE
  1. Paste the class body back into the notebook cell for its tier
     (1d = ported bank, 1f = Tier 1 moments, 1g = Tier 2 robust,
      1h = Tier 3 distances, 1i = Tier 4 dynamics).
  2. Append an instance to REGISTRY in that cell's registration block, e.g.
        REGISTRY.extend([..., HurstFamily()])
     Register it in the cell for its own tier so FEATURES keeps a stable order.
  3. Re-run from the config cell down; FEATURES / N_FEATURES re-derive themselves
     and the Phase 4 per-family cache will extract only the restored family.
  4. `z` / `e` are the two ChannelBankFamily instances that were unregistered
     (the class itself is still in the notebook, used by `z2` / `e2`). To bring
     one back, just add it to BANK_CHANNELS in the 1d cell.

Each class below is byte-identical to what was deleted from the notebook, so it
depends on the same notebook-level names (FeatureFamily, TwoSampleFamily, Ctx,
HistoricalRef, _safe, _avg_ranks, _guard, _mean_var, WINDOWS, ...). JSFamily
subclasses HellingerFamily, so restoring `js` means restoring `hell` too.

Removed families (25): acf, ad, archlm, bowley, chi2, contrast, cvm, dak2, e, h,
hell, hill, hurst, jb, js, kurt, ljungbox, moors, mwu, perm_ent, sign, skew,
slope, welch, z
"""

# flake8: noqa


# =====================================================================
# 1d -- ported bank
# =====================================================================


# --- family 'contrast' ---
class ContrastFamily(FeatureFamily):
    name = "contrast"
    channels = ("z", "e")

    def names(self):
        return ["z_contrast", "e_contrast"]

    def update(self, ctx: Ctx):
        out = []
        n = ctx.n
        for ch in ("z", "e"):
            rs = ctx.channels[ch].rs
            n1 = WINDOWS[0] if n > WINDOWS[0] else n
            n2 = (WINDOWS[1] if n > WINDOWS[1] else n) - n1
            if n2 <= 0:
                out.append(0.0)
            else:
                d = rs[0] / n1 - (rs[1] - rs[0]) / n2
                out.append(d * math.sqrt(n1 * n2 / float(n1 + n2)))
        return out

# --- family 'acf' ---
class AutocorrFamily(FeatureFamily):
    name = "acf"
    channels = ("z",)

    def init(self, hist: HistoricalRef):
        self.z1, self.z2 = hist.z_seed1, hist.z_seed2
        L = LBUF
        self.pbuf = [[0.0] * L, [0.0] * L]
        self.ps = [0.0, 0.0]
        self.pexp = [0.0, 0.0]
        self.rho = (hist.zacf1, hist.zacf2)

    def names(self):
        out = []
        for lag in (1, 2):
            out += ["z_acf%d_w%d" % (lag, ACF_W), "z_acf%d_exp" % lag]
        return out

    def update(self, ctx: Ctx):
        z = ctx.vals["z"]
        t, n = ctx.t, ctx.n
        L = LBUF
        i = t % L
        prods = (z * self.z1, z * self.z2)
        self.z2 = self.z1
        self.z1 = z
        out = []
        for k in (0, 1):
            pb = self.pbuf[k]
            self.ps[k] += prods[k] - pb[(i - ACF_W) % L]
            pb[i] = prods[k]
            self.pexp[k] += prods[k]
            nw = ACF_W if n > ACF_W else n
            out.append((self.ps[k] / nw - self.rho[k]) * math.sqrt(nw))
            out.append((self.pexp[k] / n - self.rho[k]) * math.sqrt(n))
        return out

# --- family 'sign' ---
class SignRunsFamily(FeatureFamily):
    name = "sign"
    channels = ("z",)

    def init(self, hist: HistoricalRef):
        L = LBUF
        self.sbuf = [0.0] * L
        self.ssum = [0.0] * len(SIGN_W)
        self.sexp = 0.0
        self.run_len = 0.0
        self.run_max = 0.0
        self.last_sign = 0

    def names(self):
        out = ["sign_w%d" % W for W in SIGN_W]
        out += ["sign_exp", "run_len", "run_max"]
        return out

    def update(self, ctx: Ctx):
        z = ctx.vals["z"]
        t, n = ctx.t, ctx.n
        L = LBUF
        i = t % L
        s = 1.0 if z > 0.0 else 0.0
        for k in range(len(SIGN_W)):
            self.ssum[k] += s - self.sbuf[(i - SIGN_W[k]) % L]
        self.sbuf[i] = s
        self.sexp += s
        out = []
        for k in range(len(SIGN_W)):
            nw = SIGN_W[k] if n > SIGN_W[k] else n
            out.append((self.ssum[k] - 0.5 * nw) / math.sqrt(0.25 * nw))
        out.append((self.sexp - 0.5 * n) / math.sqrt(0.25 * n))
        sg = 1 if z > 0.0 else -1
        if sg == self.last_sign:
            self.run_len += 1.0
        else:
            self.run_len = 1.0
            self.last_sign = sg
        if self.run_len > self.run_max:
            self.run_max = self.run_len
        out.append(self.run_len)
        out.append(self.run_max)
        return out

# --- family 'chi2' ---
class Chi2Family(FeatureFamily):
    name = "chi2"
    channels = ()

    def init(self, hist: HistoricalRef):
        L = LBUF
        self.bbuf = [0.0] * L
        self.cnt_w = [0.0] * NBINS
        self.cnt_e = [0.0] * NBINS

    def names(self):
        return ["chi2_w%d" % BIN_W, "chi2_exp"]

    def update(self, ctx: Ctx):
        u, t, n = ctx.u, ctx.t, ctx.n
        L = LBUF
        i = t % L
        bi = int(u * NBINS)
        if bi >= NBINS:
            bi = NBINS - 1
        if t >= BIN_W:
            self.cnt_w[int(self.bbuf[(i - BIN_W) % L])] -= 1.0
        self.bbuf[i] = float(bi)
        self.cnt_w[bi] += 1.0
        self.cnt_e[bi] += 1.0
        nw = BIN_W if n > BIN_W else n
        ew = nw / NBINS
        chi = sum((self.cnt_w[k] - ew) ** 2 for k in range(NBINS)) / ew
        v1 = (chi - (NBINS - 1)) / math.sqrt(2.0 * (NBINS - 1))
        ee = n / NBINS
        chi = sum((self.cnt_e[k] - ee) ** 2 for k in range(NBINS)) / ee
        v2 = (chi - (NBINS - 1)) / math.sqrt(2.0 * (NBINS - 1))
        return [v1, v2]

# --- family 'slope' ---
class SlopeFamily(FeatureFamily):
    name = "slope"
    channels = ("z",)

    def init(self, hist: HistoricalRef):
        self.slope_B = 0.0

    def names(self):
        return ["slope_w%d" % SLOPE_W]

    def update(self, ctx: Ctx):
        z = ctx.vals["z"]
        t, n = ctx.t, ctx.n
        L = LBUF
        i = t % L
        cs = ctx.channels["z"]
        zb = cs.buf
        self.slope_B += t * z - (t - SLOPE_W) * zb[(i - SLOPE_W) % L]
        nw = SLOPE_W if n > SLOPE_W else n
        A = cs.rs[WINDOWS.index(SLOPE_W)] if SLOPE_W in WINDOWS else 0.0
        jbar = t - (nw - 1) / 2.0
        sxx = nw * (nw * nw - 1.0) / 12.0
        return [(self.slope_B - jbar * A) / math.sqrt(sxx + 1e-12)]

# --- family 'h' ---
class ContextFamily(FeatureFamily):
    """Constant per series -- carries no break evidence by itself (design
    principle #5); lets the model learn how much to trust other families for
    this kind of process. Judged by ablation, not univariate AUC."""
    name = "h"
    channels = ()

    def init(self, hist: HistoricalRef):
        self.context = hist.context

    def names(self):
        return ["h_n", "h_logn", "h_skew", "h_kurt", "h_acf1", "h_acf2",
                 "h_ar1", "h_ar2", "h_ar3", "h_rsd_ratio", "h_sd_mad",
                 "h_p2", "h_p3", "h_zacf1"]

    def update(self, ctx: Ctx):
        return list(self.context)


# =====================================================================
# 1f -- Tier 1 moment ladder
# =====================================================================


# --- family 'welch' ---
class WelchTFamily(TwoSampleFamily):
    name = "welch"
    needs_sorted = False

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        m1, v1 = _mean_var(w)
        m2, v2 = _mean_var(r)
        se = math.sqrt(v1 / m + v2 / nr)
        return (m1 - m2) / se if se > 0 else 0.0

# --- family 'mwu' ---
class MannWhitneyFamily(TwoSampleFamily):
    name = "mwu"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr):
            return 0.0
        combined = np.concatenate([w, r])
        ranks = _avg_ranks(combined)
        rank_sum_w = ranks[:m].sum()
        u = rank_sum_w - m * (m + 1) / 2.0
        mean = m * nr / 2.0
        var = m * nr * (m + nr + 1) / 12.0
        return (u - mean) / math.sqrt(var) if var > 0 else 0.0

# --- family 'skew' ---
class SkewFamily(TwoSampleFamily):
    """D'Agostino skewness g1, standardized ~N(0, 6/n)."""
    name = "skew"
    needs_sorted = False

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        mu, v = _mean_var(w)
        s = math.sqrt(v)
        if s <= 0:
            return 0.0
        g1 = float(((w - mu) ** 3).mean()) / s ** 3
        return g1 / math.sqrt(6.0 / m)

# --- family 'kurt' ---
class KurtFamily(TwoSampleFamily):
    """Anscombe-Glynn excess kurtosis g2, standardized ~N(0, 24/n)."""
    name = "kurt"
    needs_sorted = False

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        mu, v = _mean_var(w)
        s2 = v
        if s2 <= 0:
            return 0.0
        g2 = float(((w - mu) ** 4).mean()) / s2 ** 2 - 3.0
        return g2 / math.sqrt(24.0 / m)

# --- family 'jb' ---
class JarqueBeraFamily(TwoSampleFamily):
    """JB ~ chi2(2), standardized like the existing chi2_* features."""
    name = "jb"
    needs_sorted = False

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        mu, v = _mean_var(w)
        s = math.sqrt(v)
        if s <= 0:
            return 0.0
        g1 = float(((w - mu) ** 3).mean()) / s ** 3
        g2 = float(((w - mu) ** 4).mean()) / s ** 4 - 3.0
        jb = m / 6.0 * (g1 ** 2 + g2 ** 2 / 4.0)
        return (jb - 2.0) / 2.0

# --- family 'dak2' ---
class DAgostinoK2Family(TwoSampleFamily):
    """D'Agostino-Pearson omnibus K^2 ~ chi2(2) -- the scipy.stats.normaltest
    algorithm, reimplemented directly so the online path stays dependency-free."""
    name = "dak2"
    needs_sorted = False

    def stat(self, w, r):
        n = w.size
        if n < 20:
            return 0.0
        mu, v = _mean_var(w)
        s = math.sqrt(v)
        if s <= 0:
            return 0.0
        g1 = float(((w - mu) ** 3).mean()) / s ** 3
        g2 = float(((w - mu) ** 4).mean()) / s ** 4 - 3.0

        y = g1 * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
        beta2 = (3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) /
                 ((n - 2) * (n + 5) * (n + 7) * (n + 9)))
        w2m1 = 2 * (beta2 - 1.0)
        if w2m1 <= 0:
            return 0.0
        w2 = -1.0 + math.sqrt(w2m1)
        if w2 <= 1.0:
            return 0.0
        delta = 1.0 / math.sqrt(0.5 * math.log(w2))
        alpha = math.sqrt(2.0 / (w2 - 1.0))
        arg = y / alpha
        z1 = delta * math.log(arg + math.sqrt(arg * arg + 1.0))

        e = 3.0 * (n - 1) / (n + 1)
        var = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
        if var <= 0:
            return 0.0
        x = (g2 - e) / math.sqrt(var)
        sqrtb1 = (6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) *
                  math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3))))
        if sqrtb1 <= 0:
            return 0.0
        a = 6.0 + 8.0 / sqrtb1 * (2.0 / sqrtb1 + math.sqrt(1.0 + 4.0 / sqrtb1 ** 2))
        term = (1.0 - 2.0 / a) / (1.0 + x * math.sqrt(2.0 / (a - 4.0)))
        if term <= 0 or a <= 0:
            return 0.0
        z2 = ((1.0 - 2.0 / (9.0 * a)) - term ** (1.0 / 3.0)) / math.sqrt(2.0 / (9.0 * a))

        k2 = z1 * z1 + z2 * z2
        return (k2 - 2.0) / 2.0


# =====================================================================
# 1g -- Tier 2 robust analogues
# =====================================================================


# --- family 'bowley' ---
class BowleySkewFamily(TwoSampleFamily):
    """(Q3 + Q1 - 2*Q2) / (Q3 - Q1) -- quartile-based skewness."""
    name = "bowley"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_b = {ch: self._bowley(self.ref[ch]) for ch in ("z", "e")}

    @staticmethod
    def _bowley(x):
        if x.size < 4:
            return 0.0
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        d = q3 - q1
        return float((q3 + q1 - 2 * q2) / d) if d > 0 else 0.0

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            ref_b = self.ref_b[ch]
            for W in WINDOWS:
                w = self._window(ctx, ch, W)
                m = w.size
                if m < 8:
                    out.append(0.0)
                    continue
                out.append(_safe((self._bowley(w) - ref_b) * math.sqrt(m)))
        return out

# --- family 'moors' ---
class MoorsKurtosisFamily(TwoSampleFamily):
    """Octile-based kurtosis: ((E7-E5)+(E3-E1)) / (E6-E2)."""
    name = "moors"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_t = {ch: self._moors(self.ref[ch]) for ch in ("z", "e")}

    @staticmethod
    def _moors(x):
        if x.size < 8:
            return 0.0
        e1, e2, e3, e5, e6, e7 = np.percentile(x, [12.5, 25, 37.5, 62.5, 75, 87.5])
        den = e6 - e2
        return float(((e7 - e5) + (e3 - e1)) / den) if den > 0 else 0.0

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            ref_t = self.ref_t[ch]
            for W in WINDOWS:
                w = self._window(ctx, ch, W)
                m = w.size
                if m < 16:
                    out.append(0.0)
                    continue
                out.append(_safe((self._moors(w) - ref_t) * math.sqrt(m)))
        return out


# =====================================================================
# 1h -- Tier 3 distribution distances
# =====================================================================


# --- family 'cvm' ---
class CvMFamily(TwoSampleFamily):
    """Two-sample Cramer-von Mises (Anderson 1962 form), standardized against
    the known asymptotic mean/variance of the omega^2 statistic."""
    name = "cvm"
    cost = "heavy"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr, 8):
            return 0.0
        n = m + nr
        combined = np.concatenate([w, r])
        ranks = _avg_ranks(combined)
        rw = np.sort(ranks[:m])
        rr = np.sort(ranks[m:])
        i1 = np.arange(1, m + 1)
        i2 = np.arange(1, nr + 1)
        u = (m * np.sum((rw - i1) ** 2) + nr * np.sum((rr - i2) ** 2)) / (m * nr * n) - \
            (4.0 * m * nr - 1.0) / (6.0 * n)
        return (u - 1.0 / 6.0) / math.sqrt(1.0 / 45.0)

# --- family 'ad' ---
class ADFamily(TwoSampleFamily):
    """Discretized two-sample Anderson-Darling (tail-weighted EDF distance)."""
    name = "ad"
    cost = "heavy"

    def stat(self, w, r):
        m, nr = w.size, r.size
        if not _guard(m, nr, 8):
            return 0.0
        n = m + nr
        combined = np.sort(np.concatenate([w, r]))
        z_sorted = np.sort(w)
        r_sorted = np.sort(r)
        fw = np.searchsorted(z_sorted, combined[:-1], side="right") / m
        fr = np.searchsorted(r_sorted, combined[:-1], side="right") / nr
        i = np.arange(1, n)
        frac = i / n
        weight = frac * (1.0 - frac)
        weight = np.where(weight > 1e-6, weight, 1e-6)
        a2 = float(np.sum((fw - fr) ** 2 / weight)) * (m * nr) / (n * n)
        return (a2 - 1.0) * math.sqrt(max(m, 1))

# --- family 'hell' ---
class HellingerFamily(TwoSampleFamily):
    """Binned Hellinger distance against the reference's own decile grid."""
    name = "hell"
    cost = "heavy"
    NB = 10

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.edges = {ch: np.quantile(self.ref[ch], np.linspace(0, 1, self.NB + 1))
                      for ch in ("z", "e")}

    def _hist(self, x, edges):
        edges = edges.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        c, _ = np.histogram(x, bins=edges)
        p = c.astype(np.float64) + 0.5
        return p / p.sum()

    def stat(self, w, r):
        if w.size < 8:
            return 0.0
        ch = self._cur_ch
        edges = self.edges[ch]
        pw = self._hist(w, edges)
        pr = self._hist(r, edges)
        h = math.sqrt(0.5 * float(np.sum((np.sqrt(pw) - np.sqrt(pr)) ** 2)))
        return h * math.sqrt(w.size)

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                w = self._window(ctx, ch, W)
                out.append(_safe(self.stat(w, self.ref[ch])))
        return out

# --- family 'js' ---
class JSFamily(HellingerFamily):
    """Binned Jensen-Shannon divergence, same binning as HellingerFamily."""
    name = "js"

    def stat(self, w, r):
        if w.size < 8:
            return 0.0
        ch = self._cur_ch
        edges = self.edges[ch]
        pw = self._hist(w, edges)
        pr = self._hist(r, edges)
        mmix = 0.5 * (pw + pr)
        kl_w = float(np.sum(pw * np.log(pw / mmix)))
        kl_r = float(np.sum(pr * np.log(pr / mmix)))
        js = 0.5 * kl_w + 0.5 * kl_r
        return js * w.size


# =====================================================================
# 1i -- Tier 4 dynamics / tails
# =====================================================================


# --- family 'hill' ---
class HillEstimatorFamily(TwoSampleFamily):
    """Hill tail-index estimator on |window| vs. |reference|, using k = m//4
    upper order statistics. Standardized via the estimator's known asymptotic
    SE = alpha/sqrt(k)."""
    name = "hill"
    cost = "heavy"

    def init(self, hist: HistoricalRef):
        super().init(hist)
        self.ref_alpha = {ch: self._hill(np.abs(self.ref[ch])) for ch in ("z", "e")}

    @staticmethod
    def _hill(absx, k=None):
        n = absx.size
        if n < 20:
            return 4.0
        k = k or max(4, n // 4)
        k = min(k, n - 1)
        s = np.sort(absx)[::-1]
        top = s[:k]
        xk = s[k]
        if xk <= 0:
            return 4.0
        logs = np.log(np.maximum(top / xk, 1e-12))
        m = float(logs.mean())
        return (k / m) if m > 1e-9 else 4.0

    def stat(self, w, r):
        m = w.size
        if m < 20:
            return 0.0
        ch = self._cur_ch
        alpha_ref = self.ref_alpha[ch]
        k = max(4, m // 4)
        alpha_w = self._hill(np.abs(w), k)
        se = alpha_ref / math.sqrt(k)
        return (alpha_w - alpha_ref) / se if se > 0 else 0.0

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                w = self._window(ctx, ch, W)
                out.append(_safe(self.stat(w, self.ref[ch])))
        return out

# --- family 'perm_ent' ---
class PermutationEntropyFamily(FeatureFamily):
    """Ordinal-pattern entropy (order 3) of the RAW chronological window --
    invisible to the marginal, sensitive to dynamics/dependence changes. Under
    iid continuous data normalized PE -> 1 exactly, so 1.0 is the reference
    constant (no need to store a separate reference sample)."""
    name = "perm_ent"
    cost = "sorted"
    D = 3

    def names(self):
        return ["perm_ent_%s_w%d" % (ch, W) for ch in self.channels for W in WINDOWS]

    def _pe(self, x: np.ndarray) -> float:
        d = self.D
        m = x.size - d + 1
        if m < 8:
            return 1.0
        win = np.lib.stride_tricks.sliding_window_view(x, d)
        ordi = np.argsort(win, axis=1, kind="stable")
        code = (ordi * (d ** np.arange(d))).sum(axis=1)
        cnt = np.bincount(code)
        cnt = cnt[cnt > 0].astype(np.float64)
        p = cnt / cnt.sum()
        h = -(p * np.log(p)).sum()
        return h / math.log(math.factorial(d))

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            for W in WINDOWS:
                w = ctx.channels[ch].window(W, ctx.t)
                pe = self._pe(w)
                out.append(_safe((pe - 1.0) * math.sqrt(max(w.size, 1))))
        return out

# --- family 'archlm' ---
class ArchLMFamily(FeatureFamily):
    """Order-1 ARCH-LM: does squared-e autocorrelate (volatility clustering
    changed), not just its level. Direct target given e2 > z2 in the diagnostic."""
    name = "archlm"
    cost = "sorted"
    channels = ("e",)

    def names(self):
        return ["archlm_%s_w%d" % (ch, W) for ch in self.channels for W in WINDOWS]

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            for W in WINDOWS:
                w = ctx.channels[ch].window(W, ctx.t)
                m = w.size
                if m < 10:
                    out.append(0.0)
                    continue
                sq = w * w
                x, y = sq[:-1], sq[1:]
                if x.std() < 1e-9 or y.std() < 1e-9:
                    out.append(0.0)
                    continue
                r = float(np.corrcoef(x, y)[0, 1])
                lm = (m - 1) * r * r
                out.append(_safe((lm - 1.0) / math.sqrt(2.0)))
        return out

# --- family 'ljungbox' ---
class LjungBoxFamily(FeatureFamily):
    """Reuses the autocorrelation pattern from AutocorrFamily (1d), but tests
    joint significance of lags 1..H rather than one lag at a time."""
    name = "ljungbox"
    cost = "sorted"
    H = 5

    def names(self):
        return ["ljungbox_%s_w%d" % (ch, W) for ch in self.channels for W in WINDOWS]

    def update(self, ctx: Ctx):
        out = []
        for ch in self.channels:
            for W in WINDOWS:
                w = ctx.channels[ch].window(W, ctx.t)
                m = w.size
                h = min(self.H, max(m // 5, 1))
                if m < 12 or h < 1:
                    out.append(0.0)
                    continue
                x = w - w.mean()
                d = float(np.dot(x, x))
                if d <= 0:
                    out.append(0.0)
                    continue
                q = 0.0
                for k in range(1, h + 1):
                    if m - k <= 0:
                        break
                    rho_k = float(np.dot(x[k:], x[:m - k])) / d
                    q += rho_k * rho_k / (m - k)
                q *= m * (m + 2)
                out.append(_safe((q - h) / math.sqrt(2.0 * h)))
        return out

# --- family 'hurst' ---
class HurstFamily(TwoSampleFamily):
    """Single-scale R/S estimate of the Hurst exponent (long-memory shift).
    Reference value under iid is exactly 0.5."""
    name = "hurst"
    cost = "heavy"
    needs_sorted = False

    def stat(self, w, r):
        m = w.size
        if m < 16:
            return 0.0
        x = w - w.mean()
        cum = np.cumsum(x)
        rng = float(cum.max() - cum.min())
        s = float(w.std(ddof=1))
        if s <= 0 or rng <= 0:
            return 0.0
        h = math.log(rng / s) / math.log(m)
        return (h - 0.5) * math.sqrt(m)


# ============================================================================
#  2026-09-24 -- families tested in research.ipynb and REJECTED
#
#  Appended when research.ipynb was rebuilt as the lean idea tester. The full
#  pre-rebuild notebook, with every experiment's cells, is kept at
#  archive/research_2026-09-24.ipynb.
#
#  Verdict numbers appear below only where the experiment's cell output or a
#  comment still recorded them; most outputs had been cleared by later runs, so
#  those families carry the rejection with no numbers rather than invented ones.
#
#  These depend on notebook/engine names that are NOT defined here: FeatureFamily,
#  Ctx, HistoricalRef, _CondBase, _safe, _history, _ar_resid, _garch_grid,
#  _std_history, _SQ2, WINDOWS, ALPHAS, DRIFTS. All of them live in submission.ipynb
#  (and so in the converted _engine.py); import them from there to revive one.
# ============================================================================

# ---- helpers the rejected families need and that live ONLY here ----

# _EPSG: constant used by ConformalMartingaleFamily (from research.ipynb cell 23)
_EPSG = (0.3, 0.5, 0.8)                 # power-betting exponents for the e-detectors

# _BV, _ZCAP: constants used by the lit5 families (from research.ipynb cell 167)
_BV = math.pi / 2.0        # (pi/2) E|Z||Z'| = 1 for iid N(0,1): bipower is unbiased for variance
_ZCAP = 20.0               # standardized stream values are clipped here (rare-event streams)

# _zr_hist: helper used by _Lit5Base (from research.ipynb cell 167)
def _zr_hist(hist):
    xs = _history(hist)
    return (xs - float(hist.mu)) / (float(hist.sd) if hist.sd > 0 else 1.0)

# _Lit5Base: shared base for the eight lit5 families (from research.ipynb cell 167)
class _Lit5Base(_CondBase):
    """History-calibrated streams + an AR(3) innovation in raw (not rank) units."""
    channels = ("z",)
    cost = "O1"
    version = 1
    SD_FLOOR = 0.05
    WARM = 50               # history steps skipped before calibration (recursive states warming up)

    def names(self):
        return self._agg_names()

    # ---- raw-unit AR innovation, same recursion as HistoricalRef's residual ----
    def _setup_innov(self, hist):
        zh = _zr_hist(hist)
        self.ar = [float(a) for a in hist.ar_l]
        p = len(self.ar)
        r = zh[p:].copy()
        for k in range(1, p + 1):
            r -= self.ar[k - 1] * zh[p - k:zh.size - k]
        s = float(r.std(ddof=1)) if r.size > 2 else 1.0
        self.rs = s if s > 1e-12 else 1.0
        self.lagz = [float(v) for v in zh[-p:][::-1]]           # most recent first
        return zh, r / self.rs                                   # r[j] belongs to time p + j

    def _innov(self, zr):
        e = zr
        for a, l in zip(self.ar, self.lagz):
            e -= a * l
        self.lagz = [zr] + self.lagz[:-1]
        return e / self.rs

    # ---- calibration ----
    def _calibrate(self, S):
        self.mu_s, self.sd_s = {}, {}
        for k in self.STREAMS:
            v = np.asarray(S.get(k, ()), dtype=np.float64)
            v = v[np.isfinite(v)]
            m, s = (float(v.mean()), float(v.std())) if v.size >= 20 else (0.0, 1.0)
            self.mu_s[k] = m
            self.sd_s[k] = max(s, self.SD_FLOOR)

    def _emit(self, raw):
        vals = {}
        for k in self.STREAMS:
            z = (raw[k] - self.mu_s[k]) / self.sd_s[k]
            vals[k] = _safe(min(max(z, -_ZCAP), _ZCAP))
        self._push(vals)
        return self._aggregate()


# ---- the rejected families ----

# GLRAggFamily (glr2): rejected
class GLRAggFamily(FeatureFamily):
    """Aggregations over the unknown break point k, from two prefix sums of `e`.

    No break: e ~ N(0,1). Break at k (length m, S1 = sum e, S2 = sum e^2):
        variance LLR_k = m/2 * (s2 - 1 - log s2),  s2 = S2/m
        mean     LLR_k = S1^2 / (2m)
    sup over k is the classic test; Andrews & Ploberger (1994) showed mean/exponential
    aggregation has better weighted-average power when k is unknown. bocpd_logodds is
    already the exponential one, so this adds the mean, a top-decile trimmed mean, and
    the mean-shift test (the old z/e families used FIXED windows, not a maximized k)."""
    name = "glr2"
    channels = ("e",)
    cost = "O1"          # stateful (prefix sums) -> runs every step
    version = 1
    MIN_SEG = 8
    TRIM_Q = 0.10

    def names(self):
        return ["glr_var_mean", "glr_var_trimmean", "glr_mean_max", "glr_mean_argmax"]

    def init(self, hist):
        cap = 1025
        self.P2 = np.zeros(cap, dtype=np.float64)
        self.P1 = np.zeros(cap, dtype=np.float64)
        self.K = np.arange(cap, dtype=np.float64)
        self.n = 0

    def _grow(self):
        old = self.P2.size; cap = 2 * old
        for nm in ("P2", "P1"):
            arr = np.zeros(cap, dtype=np.float64); arr[:old] = getattr(self, nm)
            setattr(self, nm, arr)
        self.K = np.arange(cap, dtype=np.float64)

    def update(self, ctx):
        x = ctx.vals["e"]
        n = self.n + 1
        if n >= self.P2.size:
            self._grow()
        self.P2[n] = self.P2[n - 1] + x * x
        self.P1[n] = self.P1[n - 1] + x
        self.n = n
        M = self.MIN_SEG
        if n < M:
            return [0.0, 0.0, 0.0, 0.0]
        nk = n - M + 1
        m = n - self.K[:nk]
        r = np.maximum((self.P2[n] - self.P2[:nk]) / m, 1e-12)
        llr_var = 0.5 * m * (r - 1.0 - np.log(r))
        s1 = self.P1[n] - self.P1[:nk]
        llr_mean = (s1 * s1) / (2.0 * m)
        ntop = max(1, int(round(self.TRIM_Q * nk)))
        top = np.partition(llr_var, nk - ntop)[nk - ntop:] if ntop < nk else llr_var
        j = int(np.argmax(llr_mean))
        return [_safe(float(llr_var.mean())), _safe(float(top.mean())),
                _safe(float(llr_mean[j])), math.log(m[j])]


# ConformalMartingaleFamily (ctm): rejected
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


# NEWMAFamily (newma): rejected
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


# BOCPDMeanVarFamily (bocpd2): rejected
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


# DependenceFamily (dep): rejected
class DependenceFamily(FeatureFamily):
    """Detects a change in the DEPENDENCE structure, which nothing else in the bank sees.

    The `e` channel is the residual after whitening with the AR(p) fitted on the historical
    segment, so under "no break" it is serially uncorrelated: rho_k ~ N(0, 1/m). If the break
    changes the autocorrelation (white noise -> AR(1)) while leaving the marginal variance
    alone, every variance/distribution feature stays flat and only this one moves. Equivalently:
    var(e_t - e_{t-1}) = 2*var(e)*(1 - rho_1), so this is the differenced-channel signal.

    Per window: rho_1, rho_2 as z-scores (rho*sqrt(m)), and Ljung-Box style Q = m*(rho_1^2+rho_2^2),
    which fires for either sign. Plus two EWMA rho_1 for faster reaction after a break."""
    name = "dep"
    channels = ("e",)
    cost = "O1"
    version = 1
    EW_ALPHAS = (0.02, 0.08)

    def names(self):
        out = []
        for W in WINDOWS:
            out += ["dep_rho1_w%d" % W, "dep_rho2_w%d" % W, "dep_q_w%d" % W]
        out += ["dep_rho1_ew%g" % a for a in self.EW_ALPHAS]
        return out

    def init(self, hist):
        self.s11 = [0.0] * len(self.EW_ALPHAS)     # EWMA of e_t * e_{t-1}
        self.s00 = [0.0] * len(self.EW_ALPHAS)     # EWMA of e_t^2
        self.prev = 0.0
        self.n = 0
        self.cap = max(WINDOWS)                    # own ring buffer: no channel plumbing needed
        self.buf = np.zeros(self.cap, dtype=np.float64)

    def _window(self, W):
        m = W if self.n > W else self.n
        i = (self.n - 1) % self.cap
        s = i - m + 1
        if s >= 0:
            return self.buf[s:i + 1]
        return np.concatenate([self.buf[s % self.cap:], self.buf[:i + 1]])

    def update(self, ctx):
        x = ctx.vals["e"]
        self.n += 1
        self.buf[(self.n - 1) % self.cap] = x
        out = []
        for W in WINDOWS:
            w = self._window(W)
            m = w.size
            if m < 12:
                out += [0.0, 0.0, 0.0]
                continue
            s0 = float(np.dot(w, w))
            if s0 <= 0.0:
                out += [0.0, 0.0, 0.0]
                continue
            r1 = float(np.dot(w[1:], w[:-1])) / s0
            r2 = float(np.dot(w[2:], w[:-2])) / s0
            rm = math.sqrt(m)
            out += [_safe(r1 * rm), _safe(r2 * rm), _safe(m * (r1 * r1 + r2 * r2))]

        p = self.prev
        self.prev = x
        for i, a in enumerate(self.EW_ALPHAS):
            self.s11[i] += a * (x * p - self.s11[i])
            self.s00[i] += a * (x * x - self.s00[i])
            r = self.s11[i] / self.s00[i] if self.s00[i] > 1e-12 else 0.0
            out.append(_safe(r * math.sqrt(min(self.n, 2.0 / a))))
        return out


# LongWindowFamily (long): rejected
class LongWindowFamily(FeatureFamily):
    """The scale statistics that win at W=256, extended to 512 and 1024.

    Scale features improve monotonically with window (e2_w32 .550 < w64 .556 < w256 .560),
    while SHAPE features go the other way (lmom_lkurt_e_w32 .559 > w64 > w256), so this
    extends only the scale side. Both statistics are exact O(1) from running sums of e and
    e^2, so the longer windows cost nothing. Also adds a slower EWMA, since ew0.02 beat ew0.08."""
    name = "long"
    channels = ("e",)
    cost = "O1"
    version = 1
    WINS = (512, 1024)
    EW = (0.005,)

    def names(self):
        out = []
        for W in self.WINS:
            out += ["e2_w%d" % W, "ftest_e_w%d" % W]
        out += ["e2_ew%g" % a for a in self.EW]
        return out

    def init(self, hist):
        self.cap = max(self.WINS)
        self.buf = np.zeros(self.cap, dtype=np.float64)
        self.n = 0
        self.ew = [0.0] * len(self.EW)
        r = hist.etab if hasattr(hist, "etab") else None
        self.vref = float(np.var(np.asarray(r)[1:-1], ddof=1)) if r is not None else 1.0
        self.nref = (len(r) - 2) if r is not None else 1000

    def _sums(self, W):
        m = W if self.n > W else self.n
        i = (self.n - 1) % self.cap
        s = i - m + 1
        w = self.buf[s:i + 1] if s >= 0 else np.concatenate([self.buf[s % self.cap:], self.buf[:i + 1]])
        return w.sum(), float(np.dot(w, w)), m

    def update(self, ctx):
        x = ctx.vals["e"]
        self.n += 1
        self.buf[(self.n - 1) % self.cap] = x
        out = []
        for W in self.WINS:
            s1, s2, m = self._sums(W)
            if m < 16:
                out += [0.0, 0.0]
                continue
            out.append(_safe((s2 / m - 1.0) / math.sqrt(2.0) * math.sqrt(m)))
            v = max((s2 - s1 * s1 / m) / (m - 1), 1e-12)
            se = math.sqrt(2.0 / (m - 1) + 2.0 / max(self.nref - 1, 2))
            out.append(_safe(math.log(v / max(self.vref, 1e-12)) / se))
        e2 = (x * x - 1.0) * 0.7071067811865476
        for i, a in enumerate(self.EW):
            self.ew[i] += a * (e2 - self.ew[i])
            out.append(_safe(self.ew[i] * math.sqrt(min(self.n, 2.0 / a))))
        return out


# DensityRatioFamily (dratio): rejected
class DensityRatioFamily(FeatureFamily):
    """uLSIF: fit w(x) = sum_j a_j K(x, c_j) to approximate p_test/p_ref by least squares.

    With a FIXED basis and the historical sample as the reference, the Gram matrix
    H_ij = E_ref[K_i K_j] is constant, so a = (H + lam I)^-1 h where h_i = E_test[K_i] is
    just a mean of basis functions over the test window. The Pearson divergence estimate
    PE = h.a/2 - 1/2 is then O(k) per step given running sums of the basis vector.
    Windows are exact (add the new phi, drop the one leaving); the EWMA versions need no
    buffer at all and react faster.
    """
    name = "dratio"
    channels = ("e",)
    cost = "O1"
    version = 1
    CENTERS = (-2.5, -1.75, -1.0, -0.5, 0.0, 0.5, 1.0, 1.75, 2.5)
    SIGMA = 0.7
    LAM = 0.05
    EW = (0.02, 0.005)

    def names(self):
        return (["dratio_pe_w%d" % W for W in WINDOWS]
                + ["dratio_pe_ew%g" % a for a in self.EW])

    def init(self, hist):
        c = np.asarray(self.CENTERS, dtype=np.float64)
        self.c = c
        self.k = c.size
        s2 = 2.0 * self.SIGMA ** 2
        ref = np.asarray(hist.etab, dtype=np.float64)[1:-1]         # historical PIT-normal grid
        P = np.exp(-(ref[:, None] - c[None, :]) ** 2 / s2)          # basis on the reference
        H = (P.T @ P) / max(ref.size, 1)
        self.Hinv = np.linalg.inv(H + self.LAM * np.eye(self.k))
        self.s2 = s2
        self.cap = max(WINDOWS)
        self.buf = np.zeros((self.cap, self.k), dtype=np.float64)
        self.sums = {W: np.zeros(self.k) for W in WINDOWS}
        self.ew = [np.zeros(self.k) for _ in self.EW]
        self.n = 0

    def _pe(self, h):
        a = self.Hinv @ h
        return float(h @ a) * 0.5 - 0.5

    def update(self, ctx):
        x = ctx.vals["e"]
        phi = np.exp(-(x - self.c) ** 2 / self.s2)
        i = self.n % self.cap
        self.n += 1
        out = []
        for W in WINDOWS:
            s = self.sums[W]
            s += phi
            if self.n > W:                                   # drop the one leaving the window
                s -= self.buf[(self.n - 1 - W) % self.cap]
            m = W if self.n > W else self.n
            out.append(_safe(self._pe(s / m) * math.sqrt(m)) if m >= 16 else 0.0)
        self.buf[i] = phi
        for j, a in enumerate(self.EW):
            self.ew[j] += a * (phi - self.ew[j])
            out.append(_safe(self._pe(self.ew[j]) * math.sqrt(min(self.n, 2.0 / a))))
        return out


# RunLengthFamily (rlen): rejected
class RunLengthFamily(FeatureFamily):
    """Adams & MacKay (2007) BOCPD: posterior over the run length since the last change.

    Unlike `bocpd` (one change, ever), this allows repeated changes via a constant hazard,
    so the filter tracks "how long the process has looked stationary". Under no break the
    run length grows with t; a break collapses it. The features read the posterior itself:
    expected and MAP run length (log), its entropy, and the mass on short runs -- quantities
    that are not monotone functions of any variance statistic.
    """
    name = "rlen"
    channels = ("e",)
    cost = "O1"
    version = 1
    HAZARD = 1.0 / 250.0
    MU0, KAPPA0, A0, B0 = 0.0, 1.0, 2.0, 1.0
    SHORT = 32

    def names(self):
        return ["rlen_mean_log", "rlen_map_log", "rlen_entropy", "rlen_p_short"]

    def init(self, hist):
        cap = 1024
        self.w = np.zeros(cap); self.w[0] = 1.0            # run-length posterior
        self.mu = np.zeros(cap); self.ka = np.zeros(cap)
        self.al = np.zeros(cap); self.be = np.zeros(cap)
        self.mu[0], self.ka[0], self.al[0], self.be[0] = self.MU0, self.KAPPA0, self.A0, self.B0
        self.k = 1

    def _grow(self):
        old = self.w.size; cap = 2 * old
        for nm in ("w", "mu", "ka", "al", "be"):
            arr = np.zeros(cap); arr[:old] = getattr(self, nm)
            setattr(self, nm, arr)

    def update(self, ctx):
        x = ctx.vals["e"]
        k = self.k
        if k + 1 >= self.w.size:
            self._grow()
        mu, ka, al, be = self.mu[:k], self.ka[:k], self.al[:k], self.be[:k]
        s2 = be * (ka + 1.0) / (al * ka)
        df = 2.0 * al
        pred = np.exp(_gl3(al + 0.5) - _gl3(al) - 0.5 * np.log(math.pi * df * s2)
                      - (al + 0.5) * np.log1p((x - mu) ** 2 / (s2 * df)))
        h = self.HAZARD
        w = self.w[:k]
        growth = w * pred * (1.0 - h)
        cp = float((w * pred * h).sum())
        # posterior update of each surviving hypothesis, then shift by one
        d = x - mu
        nbe = be + 0.5 * ka * d * d / (ka + 1.0)
        nmu = mu + d / (ka + 1.0)
        nka = ka + 1.0
        nal = al + 0.5
        self.w[1:k + 1] = growth
        self.mu[1:k + 1] = nmu; self.ka[1:k + 1] = nka
        self.al[1:k + 1] = nal; self.be[1:k + 1] = nbe
        self.w[0] = cp
        self.mu[0], self.ka[0], self.al[0], self.be[0] = self.MU0, self.KAPPA0, self.A0, self.B0
        k += 1
        self.k = k
        w = self.w[:k]
        tot = float(w.sum())
        if tot <= 0.0:
            w[:] = 0.0; w[0] = 1.0; tot = 1.0
        w /= tot
        r = np.arange(k, dtype=np.float64)
        mean_r = float(w @ r)
        map_r = float(np.argmax(w))
        ent = float(-(w[w > 0] * np.log(w[w > 0])).sum())
        p_short = float(w[:min(self.SHORT, k)].sum())
        return [math.log1p(mean_r), math.log1p(map_r), _safe(ent), _safe(p_short)]


# DistanceFamily (dist): rejected
class DistanceFamily(FeatureFamily):
    """AD / CvM / KS / JS / Hellinger / chi-square between the recent window and the historical
    reference. v2: cost "O1" -- v1 was declared "sorted", so the driver fed its private buffer only
    every 4th/16th point after t=128 and every window silently spanned 4-16x more real time.
    The O(W log W) computation is throttled internally on the same schedule instead."""
    name = "dist"
    channels = ("e",)
    cost = "O1"
    version = 2
    WINS = (64, 256)
    NBIN = 12

    def names(self):
        return ["dist_%s_w%d" % (s, W) for W in self.WINS
                for s in ("ad", "cvm", "ks", "js", "hell", "chi2")]

    def init(self, hist):
        ref = np.sort(np.asarray(hist.etab, dtype=np.float64)[1:-1])
        self.ref, self.nref = ref, ref.size
        self.edges = np.quantile(ref, np.arange(1, self.NBIN) / self.NBIN)
        self.p0 = 1.0 / self.NBIN
        self.cap = max(self.WINS)
        self.buf = np.zeros(self.cap, dtype=np.float64)
        self.n = 0
        self._last = [0.0] * (6 * len(self.WINS))

    def _window(self, W):
        m = W if self.n > W else self.n
        i = (self.n - 1) % self.cap
        s = i - m + 1
        return self.buf[s:i + 1] if s >= 0 else np.concatenate([self.buf[s % self.cap:], self.buf[:i + 1]])

    def update(self, ctx):
        self.buf[self.n % self.cap] = ctx.vals["e"]           # EVERY point goes into the buffer
        self.n += 1
        t = ctx.t
        stride = 1 if t < 128 else (4 if t < 512 else 16)
        if t % stride:
            return self._last                                  # hold, but the buffer stays exact
        out = []
        for W in self.WINS:
            w = self._window(W)
            m = w.size
            if m < 16:
                out += [0.0] * 6
                continue
            ws = np.sort(w)
            F = np.clip((np.searchsorted(self.ref, ws, "right") + 0.5) / (self.nref + 1.0), 1e-6, 1 - 1e-6)
            i1 = 2.0 * np.arange(1, m + 1) - 1.0
            ad = -m - float(np.sum(i1 * (np.log(F) + np.log(1.0 - F[::-1])))) / m
            cvm = 1.0 / (12.0 * m) + float(np.sum((F - i1 / (2.0 * m)) ** 2))
            k = np.arange(1, m + 1) / m
            ks = float(max(np.max(k - F), np.max(F - (k - 1.0 / m)))) * math.sqrt(m)
            cnt = np.bincount(np.searchsorted(self.edges, w, "right"), minlength=self.NBIN).astype(np.float64)
            p = cnt / m
            mmix = 0.5 * (p + self.p0)
            nz = p > 0
            js = float(0.5 * (np.sum(p[nz] * np.log(p[nz] / mmix[nz]))
                              + np.sum(self.p0 * np.log(self.p0 / mmix))))
            hell = float(np.sqrt(max(1.0 - np.sum(np.sqrt(p * self.p0)), 0.0)))
            chi2 = float(np.sum((p - self.p0) ** 2) / self.p0) * m
            out += [_safe(ad), _safe(cvm * m), _safe(ks), _safe(js * m), _safe(hell * math.sqrt(m)), _safe(chi2)]
        self._last = out
        return out


# RawMeanFamily (rawmu): rejected
class RawMeanFamily(FeatureFamily):
    """Location tests on the RAW series, before PIT and before AR whitening.

    Every location family in the original bank (z, e, welch, mwu, contrast, slope) worked on
    the PIT-transformed value or the AR(3) residual, where a level shift is partly absorbed by
    the AR fit and compressed by the rank transform. This reads ctx.zr = (x - mu_hist)/sd_hist
    directly, so a mean break arrives undiminished.

    Per window: the standardized mean (z-test vs the historical mean) and a Welch-style t that
    also uses the window's own variance, so it is not fooled when the variance moves at the same
    time. Plus one-sided CUSUM detectors for persistent drift, and an expanding-mean z-test."""
    name = "rawmu"
    channels = ("z",)
    cost = "O1"
    version = 1
    WINS = (32, 128, 512)
    DRIFTS = (0.25, 0.5)

    def names(self):
        out = []
        for W in self.WINS:
            out += ["rawmu_z_w%d" % W, "rawmu_t_w%d" % W]
        out += ["rawmu_z_exp", "rawmu_ew0.02"]
        out += ["rawmu_cusum_up%g" % d for d in self.DRIFTS]
        out += ["rawmu_cusum_dn%g" % d for d in self.DRIFTS]
        return out

    def init(self, hist):
        self.cap = max(self.WINS)
        self.buf = np.zeros(self.cap, dtype=np.float64)
        self.n = 0
        self.s1 = 0.0
        self.ew = 0.0
        self.up = [0.0] * len(self.DRIFTS)
        self.dn = [0.0] * len(self.DRIFTS)

    def _window(self, W):
        m = W if self.n > W else self.n
        i = (self.n - 1) % self.cap
        s = i - m + 1
        return self.buf[s:i + 1] if s >= 0 else np.concatenate([self.buf[s % self.cap:], self.buf[:i + 1]])

    def update(self, ctx):
        r = float(ctx.zr)                      # (x - mu_hist) / sd_hist
        self.n += 1
        self.buf[(self.n - 1) % self.cap] = r
        self.s1 += r
        self.ew += 0.02 * (r - self.ew)

        out = []
        for W in self.WINS:
            w = self._window(W)
            m = w.size
            if m < 8:
                out += [0.0, 0.0]
                continue
            mean = float(w.mean())
            out.append(_safe(mean * math.sqrt(m)))
            se = math.sqrt(max(float(w.var(ddof=1)), 1e-12) / m)
            out.append(_safe(mean / se))
        out.append(_safe(self.s1 / self.n * math.sqrt(self.n)))
        out.append(_safe(self.ew * math.sqrt(min(self.n, 100.0))))
        for j, d in enumerate(self.DRIFTS):
            self.up[j] = max(0.0, self.up[j] + r - d)
            self.dn[j] = max(0.0, self.dn[j] - r - d)
        out += [_safe(v) for v in self.up] + [_safe(v) for v in self.dn]
        return out


# TemporalFamily (temporal): rejected
class TemporalFamily(FeatureFamily):
    """Everything about TIME structure in one family, so one CV run answers all three questions.

    A. dependence & spectrum (on e, the AR-whitened residual, serially uncorrelated under no
       break): ACF lags 1-10, PACF 1-3 (Durbin-Levinson), Ljung-Box Q(10), and the share of
       spectral power in low / mid / high bands.
    B. volatility clustering: ACF of e^2 and |e| at lags 1-3, plus an ARCH-LM-style Q on e^2.
    C. persistence (on raw standardized ctx.zr): variance ratios VR(k) at k = 2, 4, 8, and an
       aggregated-variance Hurst estimate.
    Called every step (cost "O1") since it keeps its own buffers; the O(W log W) work runs
    every STRIDE steps and is held in between."""
    name = "temporal"
    channels = ("e",)
    cost = "O1"
    version = 1
    W = 256
    STRIDE = 4
    LAGS = 10

    def names(self):
        out = ["tmp_acf%d" % k for k in range(1, self.LAGS + 1)]
        out += ["tmp_pacf%d" % k for k in (1, 2, 3)]
        out += ["tmp_lbq10", "tmp_band_lo", "tmp_band_mid", "tmp_band_hi"]
        out += ["tmp_acf_sq%d" % k for k in (1, 2, 3)] + ["tmp_acf_abs%d" % k for k in (1, 2, 3)]
        out += ["tmp_archq5"]
        out += ["tmp_vr%d" % k for k in (2, 4, 8)] + ["tmp_hurst"]
        return out

    def init(self, hist):
        self.e = np.zeros(self.W); self.r = np.zeros(self.W)
        self.n = 0
        self._last = [0.0] * len(self.names())

    def _ordered(self, a):
        m = min(self.n, self.W)
        i = (self.n - 1) % self.W
        s = i - m + 1
        return a[s:i + 1] if s >= 0 else np.concatenate([a[s % self.W:], a[:i + 1]])

    @staticmethod
    def _acf(x, K):
        x = x - x.mean()
        d = float(np.dot(x, x))
        if d <= 0:
            return np.zeros(K)
        return np.array([float(np.dot(x[k:], x[:-k])) / d for k in range(1, K + 1)])

    @staticmethod
    def _pacf(rho, K):
        phi = np.zeros((K + 1, K + 1)); out = []
        for k in range(1, K + 1):
            num = rho[k - 1] - sum(phi[k - 1, j] * rho[k - 1 - j] for j in range(1, k))
            den = 1.0 - sum(phi[k - 1, j] * rho[j - 1] for j in range(1, k))
            phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
            out.append(phi[k, k])
        return out

    def update(self, ctx):
        i = self.n % self.W
        self.e[i] = ctx.vals["e"]; self.r[i] = float(ctx.zr)
        self.n += 1
        if self.n < 48 or (self.n % self.STRIDE):
            return self._last

        e = self._ordered(self.e); r = self._ordered(self.r); m = e.size
        rm = math.sqrt(m)
        acf = self._acf(e, self.LAGS)
        out = [_safe(v * rm) for v in acf]
        out += [_safe(v * rm) for v in self._pacf(acf, 3)]
        out.append(_safe(m * (m + 2) * float(np.sum(acf ** 2 / (m - np.arange(1, self.LAGS + 1))))))

        P = (np.abs(np.fft.rfft(e - e.mean())) ** 2)[1:]
        tot = float(P.sum()) or 1.0
        b = len(P) // 3
        out += [_safe(float(P[:b].sum()) / tot - 1/3), _safe(float(P[b:2*b].sum()) / tot - 1/3),
                _safe(float(P[2*b:].sum()) / tot - 1/3)]

        a2 = self._acf(e * e, 5); a1 = self._acf(np.abs(e), 3)
        out += [_safe(v * rm) for v in a2[:3]] + [_safe(v * rm) for v in a1]
        out.append(_safe(m * float(np.sum(a2 ** 2))))

        v1 = float(np.var(r)) or 1e-12
        for k in (2, 4, 8):
            s = np.convolve(r, np.ones(k), "valid")[::k]
            out.append(_safe((float(np.var(s)) / (k * v1) - 1.0) * math.sqrt(len(s))))
        ks, vs = [], []
        for k in (2, 4, 8, 16):
            nb = m // k
            if nb >= 4:
                blk = r[:nb * k].reshape(nb, k).mean(axis=1)
                vk = float(np.var(blk))
                if vk > 0:
                    ks.append(math.log(k)); vs.append(math.log(vk))
        out.append(_safe(1.0 + 0.5 * np.polyfit(ks, vs, 1)[0]) if len(ks) >= 3 else 0.5)
        self._last = out
        return out


# ContextFamily (h): rejected
class ContextFamily(FeatureFamily):
    """What the HISTORICAL segment looks like -- constant for the whole series.

    Carries no break evidence by itself (so univariate TS-AUC sits near 0.5 by design). Its job is
    to let the trees learn how much to trust every other column for this kind of series: a variance
    ratio of 1.5 against a long, clean, well-fitted history is strong evidence; the same 1.5 against
    a short, heavy-tailed, badly-fitted one is noise. Culled on univariate AUC originally and never
    CV-tested -- the same mistake that nearly lost bocpd_runlen_log.

    Reads hist.context, computed once in HistoricalRef.__init__: length, skew, kurtosis, ACF(1,2),
    AR(3) coefficients, residual-sd / sd (how much the AR fit explains), sd / MAD (tail weight),
    P(|z|>2), P(|z|>3), and the ACF of the PIT-normal series."""
    name = "h"
    channels = ()
    cost = "O1"
    version = 2

    def names(self):
        return ["h_n", "h_logn", "h_skew", "h_kurt", "h_acf1", "h_acf2",
                "h_ar1", "h_ar2", "h_ar3", "h_rsd_ratio", "h_sd_mad",
                "h_p2", "h_p3", "h_zacf1"]

    def init(self, hist):
        ctx = getattr(hist, "context", None)
        assert ctx is not None and len(ctx) == 14, \
            "HistoricalRef.context missing or changed -- check HistoricalRef.__init__"
        self.ctx = [_safe(float(v)) for v in ctx]

    def update(self, ctx):
        return self.ctx


# CaviarFamily (caviar): rejected
class CaviarFamily(_CondBase):
    """CAViaR (Engle & Manganelli 2004), symmetric absolute-value form, on |eps|:
        q_t = b0 + b1*q_{t-1} + b2*|eps_{t-1}|
    one model per quantile level theta, fit on the history by minimizing the pinball (quantile) loss.
    Under the null, P(|eps_t| > q_t) = 1 - theta at EVERY step regardless of volatility clustering,
    so the exceedance rate is a model-based test of the tails. Streams: standardized hit indicator
    per theta, plus log(|eps|/q) at the higher level (how far past the quantile)."""
    name = "caviar"
    THETAS = (0.90, 0.975)
    STREAMS = ("hit90", "hit975", "excess")

    @staticmethod
    def _loss(b1, b2, a, theta, qbar, mabs):
        b0 = qbar * (1.0 - b1) - b2 * mabs
        if b0 <= 0:
            return 1e18, b0
        q, s = qbar, 0.0
        for x in a:
            u = x - q
            s += (theta - (1.0 if u < 0 else 0.0)) * u
            q = b0 + b1 * q + b2 * x
        return s, b0

    def _fit(self, a, theta):
        qbar = float(np.quantile(a, theta)); mabs = float(a.mean())
        best, bp = None, None
        for b1 in (0.5, 0.7, 0.8, 0.9, 0.95):
            for b2 in (0.03, 0.08, 0.15, 0.25, 0.4):
                f, b0 = self._loss(b1, b2, a[-600:], theta, qbar, mabs)
                if f < 1e17 and (best is None or f < best):
                    best, bp = f, (b0, b1, b2)
        if bp is None:                                   # static quantile fallback
            bp = (qbar, 0.0, 0.0)
        return bp, qbar

    def names(self):
        return self._agg_names()

    def init(self, hist):
        xs = _history(hist)
        self.mu, self.sd = float(hist.mu), (float(hist.sd) if hist.sd > 0 else 1.0)
        a = np.abs(xs - self.mu)
        self.m = []
        for th in self.THETAS:
            (b0, b1, b2), qbar = self._fit(a, th)
            q = qbar
            for x in a:
                q = b0 + b1 * q + b2 * x
            self.m.append([b0, b1, b2, max(q, 1e-12), th])
        self._setup_buffers()

    def update(self, ctx):
        x = abs(float(ctx.zr) * self.sd)
        vals = {}
        for key, mdl in zip(("hit90", "hit975"), self.m):
            b0, b1, b2, q, th = mdl
            p = 1.0 - th
            vals[key] = ((1.0 if x > q else 0.0) - p) / math.sqrt(p * (1.0 - p))
            mdl[3] = max(b0 + b1 * q + b2 * x, 1e-12)
        vals["excess"] = math.log(max(x, 1e-12) / self.m[1][3]) if x > 0 else 0.0
        self._push(vals)
        return self._aggregate()


# GarchScoreFamily (gscore): rejected
class GarchScoreFamily(_CondBase):
    """'GARCH the other way round': instead of asking whether the online data is surprising under
    the history's parameters, ask which way the online data wants the parameters to MOVE.

    A Student-t GARCH(1,1) is fit on the ordered history (same fit as `tgarch`). For every new
    observation we evaluate the SCORE -- the gradient of its log-likelihood w.r.t. each parameter --
    at theta_hist = (omega, alpha, beta, nu). Under no break the scores average to zero; after a
    break their running sums drift, and each component points at a specific parameter:
        omega  volatility level    alpha  shock propagation    beta  persistence    nu  tail weight
    The Lagrange-multiplier / Nyblom-Hansen stability test, run sequentially. Scores are standardized
    by their mean and covariance over the HISTORY, plus a joint chi-square. O(1) per step."""
    name = "gscore"
    version = 1
    WINS = (128, 512)
    COMP = ("om", "al", "be", "nu")
    STREAMS = COMP
    NUS = (4.0, 6.0, 10.0, 30.0)

    def names(self):
        return self._agg_names() + [f"{self.name}_chi2_w128", f"{self.name}_chi2_exp"]

    def _score(self, eps):
        h = max(self.h, 1e-12)
        nu = self.nu
        q = eps * eps / ((nu - 2.0) * h)
        dl_dh = -0.5 / h + 0.5 * (nu + 1.0) * q / ((1.0 + q) * h)
        g = np.array([dl_dh * self.dh[0], dl_dh * self.dh[1], dl_dh * self.dh[2],
                      0.5 * _digamma((nu + 1) / 2) - 0.5 * _digamma(nu / 2) - 0.5 / (nu - 2.0)
                      - 0.5 * math.log1p(q) + 0.5 * (nu + 1.0) * q / ((1.0 + q) * (nu - 2.0))])
        e2 = eps * eps
        self.dh = np.array([1.0 + self.be * self.dh[0],
                            e2 + self.be * self.dh[1],
                            h + self.be * self.dh[2]])
        self.h = self.om + self.al * e2 + self.be * h
        return g

    def init(self, hist):
        xs = _history(hist)
        self.mu, self.sd = float(hist.mu), (float(hist.sd) if hist.sd > 0 else 1.0)
        eps = xs - self.mu
        v = max(float(np.var(eps)), 1e-10)
        al, be, nu = _garch_grid((eps ** 2)[-600:], v, extra=self.NUS, nll=TGarchFamily._nll)
        self.al, self.be, self.nu = al, be, nu
        self.om = v * (1 - al - be)
        self.h, self.dh = v, np.zeros(3)
        G = np.empty((eps.size, 4))
        for i, e in enumerate(eps):
            G[i] = self._score(e)
        G = G[min(50, eps.size // 4):]
        self.gm = G.mean(axis=0)
        self.gs = np.maximum(G.std(axis=0), 1e-12)
        Z = (G - self.gm) / self.gs
        self.Cinv = np.linalg.inv(np.cov(Z.T) + 1e-3 * np.eye(4))
        self._setup_buffers()

    def _joint(self, W):
        zs = np.stack([self._win(self.buf[c], W) for c in self.COMP], axis=1)
        m = zs.shape[0]
        if m < 8:
            return 0.0
        zbar = zs.mean(axis=0)
        return float(m * zbar @ self.Cinv @ zbar)

    def update(self, ctx):
        eps = float(ctx.zr) * self.sd
        z = np.clip((self._score(eps) - self.gm) / self.gs, -20.0, 20.0)
        self._push(dict(zip(self.COMP, z.tolist())))
        acc = np.array([self.acc[c] for c in self.COMP]) / self.n
        chi_exp = float(self.n * acc @ self.Cinv @ acc)
        return self._aggregate() + [_safe(self._joint(128)), _safe(chi_exp)]


# OrdinalPatternFamily (ordpat): rejected
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


# EVTTailFamily (evt): rejected
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


# IrreversibilityFamily (irrev): rejected
class IrreversibilityFamily(_Lit5Base):
    """Is the process the same played backwards? (Ramsey & Rothman 1996; Lacasa et al. 2012)
    d1, d4 : cubed k-step change of the (clipped) standardized level, in history-sd units --
             skewness of increments, i.e. sharp rises vs slow falls. Not lskew (skew of LEVELS).
    rr1    : Ramsey-Rothman bicovariance  z_t^2 z_{t-1} - z_t z_{t-1}^2  (zero for reversible series)
    ord    : 1[up-up] - 1[down-down] over consecutive triples (ordpat's `mono` pools the two)."""
    name = "irrev"
    STREAMS = ("d1", "d4", "rr1", "ord")
    C = 4.0

    def init(self, hist):
        z = np.clip(_zr_hist(hist), -self.C, self.C)
        d1 = z[1:] - z[:-1]
        d4 = z[4:] - z[:-4]
        self.s1 = max(float(d1.std()), 1e-6)
        self.s4 = max(float(d4.std()), 1e-6)
        a, b, c = z[:-2], z[1:-1], z[2:]
        self._calibrate({
            "d1": (d1 / self.s1) ** 3,
            "d4": (d4 / self.s4) ** 3,
            "rr1": z[1:] ** 2 * z[:-1] - z[1:] * z[:-1] ** 2,
            "ord": ((a < b) & (b < c)).astype(float) - ((a > b) & (b > c)).astype(float),
        })
        self.zb = [float(v) for v in z[-4:]]                 # oldest first: z_{t-4} .. z_{t-1}
        self._setup_buffers()

    def update(self, ctx):
        z = min(max(float(ctx.zr), -self.C), self.C)
        zb = self.zb
        z1, z2, z4 = zb[-1], zb[-2], zb[-4]
        up = 1.0 if (z2 < z1 < z) else 0.0
        dn = 1.0 if (z2 > z1 > z) else 0.0
        raw = {"d1": ((z - z1) / self.s1) ** 3,
               "d4": ((z - z4) / self.s4) ** 3,
               "rr1": z * z * z1 - z * z1 * z1,
               "ord": up - dn}
        self.zb = zb[1:] + [z]
        return self._emit(raw)


# JumpVariationFamily (jump): rejected -- delta -0.00150 | se 0.00222 | -0.7 sigma | better 2/6 (on base111)
class JumpVariationFamily(_Lit5Base):
    """How variance arrives: many small moves or a few jumps. (Barndorff-Nielsen & Shephard 2004,
    2006; Lee & Mykland 2008). BV_loc = EWMA of the bipower term (pi/2)|r_t||r_{t-1}|, which is
    robust to isolated jumps, lagged one step, so every stream is SCALE-FREE:
    jshare : (r_t^2 - (pi/2)|r_t||r_{t-1}|) / BV_loc -- realized minus bipower = the jump part
    jflag3 : 1[r_t^2 > 9 BV_loc]    jflag4 : 1[r_t^2 > 16 BV_loc]   (jump indicators, 3 and 4 sigma)
    A pure volatility rise moves r^2 and BV_loc together and leaves all three near zero."""
    name = "jump"
    STREAMS = ("jshare", "jflag3", "jflag4")
    A = 0.05
    RCAP = 50.0

    def init(self, hist):
        _, r = self._setup_innov(hist)
        r = np.clip(r, -self.RCAP, self.RCAP)
        loc = float(np.mean(_BV * np.abs(r[1:31]) * np.abs(r[:30]))) if r.size > 31 else 1.0
        loc = max(loc, 1e-6)
        S = {k: [] for k in self.STREAMS}
        prev = float(r[0]) if r.size else 0.0
        for j in range(1, r.size):
            x = float(r[j])
            b = _BV * abs(x) * abs(prev)
            L = max(loc, 1e-6)
            if j >= self.WARM:
                S["jshare"].append((x * x - b) / L)
                S["jflag3"].append(1.0 if x * x > 9.0 * L else 0.0)
                S["jflag4"].append(1.0 if x * x > 16.0 * L else 0.0)
            loc += self.A * (b - loc)
            prev = x
        self._calibrate(S)
        self.loc, self.prev = max(loc, 1e-6), prev
        self._setup_buffers()

    def update(self, ctx):
        x = min(max(self._innov(float(ctx.zr)), -self.RCAP), self.RCAP)
        b = _BV * abs(x) * abs(self.prev)
        L = max(self.loc, 1e-6)
        raw = {"jshare": (x * x - b) / L,
               "jflag3": 1.0 if x * x > 9.0 * L else 0.0,
               "jflag4": 1.0 if x * x > 16.0 * L else 0.0}
        self.loc += self.A * (b - self.loc)
        self.prev = x
        return self._emit(raw)


# NonlinearMeanFamily (nlmean): rejected -- delta -0.00271 | se 0.00057 | -4.8 sigma | better 0/6 (on base111)
class NonlinearMeanFamily(_Lit5Base):
    """Does the next value start depending NONLINEARLY on the past? Lagrange-multiplier streams:
    under the history's linear AR(3) the innovation r_t is uncorrelated with any function of the
    lags, so each product averages to its history value; a new nonlinearity makes it drift.
    quad : r_t (z_{t-1}^2 - m2)          Keenan (1985), quadratic
    bil  : r_t z_{t-1} z_{t-2}            bilinear / cross-product
    cub  : r_t z_{t-1}^3                  Luukkonen, Saikkonen & Terasvirta (1988), smooth transition
    thr  : r_t sign(z_{t-1} - median)     Tsay (1989), threshold AR
    Signed, so the aggregates are CUSUM-like. recres only tracks LINEAR coefficient drift."""
    name = "nlmean"
    STREAMS = ("quad", "bil", "cub", "thr")
    ZC, RC = 5.0, 6.0

    def init(self, hist):
        zh, r = self._setup_innov(hist)
        p = len(self.ar)
        zc = np.clip(zh, -self.ZC, self.ZC)
        rc = np.clip(r, -self.RC, self.RC)
        z1, z2 = zc[p - 1:zc.size - 1], zc[p - 2:zc.size - 2]
        self.m2 = float(np.mean(zc * zc))
        self.med = float(np.median(zc))
        self._calibrate({"quad": rc * (z1 * z1 - self.m2), "bil": rc * z1 * z2,
                         "cub": rc * z1 ** 3, "thr": rc * np.sign(z1 - self.med)})
        self.z1, self.z2 = float(zc[-1]), float(zc[-2])
        self._setup_buffers()

    def update(self, ctx):
        zr = float(ctx.zr)
        x = min(max(self._innov(zr), -self.RC), self.RC)
        z1, z2 = self.z1, self.z2
        s = 1.0 if z1 > self.med else (-1.0 if z1 < self.med else 0.0)
        raw = {"quad": x * (z1 * z1 - self.m2), "bil": x * z1 * z2, "cub": x * z1 ** 3, "thr": x * s}
        self.z2, self.z1 = z1, min(max(zr, -self.ZC), self.ZC)
        return self._emit(raw)


# LevelVarianceFamily (lvlvar): rejected -- delta -0.00159 | se 0.00089 | -1.8 sigma | better 1/6 (on base111)
class LevelVarianceFamily(_Lit5Base):
    """Is volatility a function of WHERE the series is (not of past shocks, as in GARCH)?
    Breusch & Pagan (1979) / White (1980) score streams on the squared innovation, against a
    short EWMA of the level (lagged): L = EWMA_0.25 of the clipped standardized level.
    bp : (min(r_t^2, 36) - 1) (L_{t-1} - mean L)      wh : (min(r_t^2, 36) - 1) (L_{t-1}^2 - mean L^2)
    (Chan, Karolyi, Longstaff & Sanders 1992 for level-dependent volatility.)"""
    name = "lvlvar"
    STREAMS = ("bp", "wh")
    A, ZC = 0.25, 5.0

    def init(self, hist):
        zh, r = self._setup_innov(hist)
        p = len(self.ar)
        zc = np.clip(zh, -self.ZC, self.ZC)
        L = np.empty(zc.size)
        l = float(zc[0])
        for i in range(zc.size):
            l += self.A * (zc[i] - l)
            L[i] = l
        Ll = L[p - 1:L.size - 1]                     # level up to t-1, aligned with r (time p + j)
        self.mL, self.mL2 = float(Ll.mean()), float(np.mean(Ll * Ll))
        q = np.minimum(r * r, 36.0) - 1.0
        k = min(self.WARM, max(q.size - 20, 0))
        self._calibrate({"bp": (q * (Ll - self.mL))[k:], "wh": (q * (Ll * Ll - self.mL2))[k:]})
        self.L = l
        self._setup_buffers()

    def update(self, ctx):
        zr = float(ctx.zr)
        x = self._innov(zr)
        q = min(x * x, 36.0) - 1.0
        L = self.L
        raw = {"bp": q * (L - self.mL), "wh": q * (L * L - self.mL2)}
        self.L += self.A * (min(max(zr, -self.ZC), self.ZC) - L)
        return self._emit(raw)


# ExtremeClusterFamily (xclust): rejected
class ExtremeClusterFamily(_Lit5Base):
    """Do extremes arrive in bursts, HOLDING THEIR RATE FIXED? (Christoffersen 1998 independence
    test; Ferro & Segers 2003). Hits: I = 1[|r| > q95 of history], Iu = 1[r > q97.5], Il = 1[r < q2.5].
    p_loc = EWMA_0.02 of the hit indicator (lagged), so a plain rise in the hit RATE cancels:
    c1 : I_{t-1} (I_t - p_loc)            hit right after a hit, beyond the current rate
    c5 : any(I_{t-5..t-1}) (I_t - p_loc)  hit within 5 steps of a hit
    cu / cl : the c1 stream for the upper / lower tail separately (asymmetric clustering).
    tail / evt / caviar measure how OFTEN and how BIG; none measures whether extremes cluster."""
    name = "xclust"
    STREAMS = ("c1", "c5", "cu", "cl")
    A = 0.02

    def init(self, hist):
        _, r = self._setup_innov(hist)
        if r.size < 40:
            self.q, self.qu, self.ql = 2.0, 2.0, -2.0
        else:
            self.q = float(np.quantile(np.abs(r), 0.95))
            self.qu, self.ql = float(np.quantile(r, 0.975)), float(np.quantile(r, 0.025))
        self.p, self.pu, self.pl = 0.05, 0.025, 0.025
        self.last = [0.0] * 5                              # I_{t-5} .. I_{t-1}
        self.lu = self.ll = 0.0
        S = {k: [] for k in self.STREAMS}
        for j in range(r.size):
            raw = self._step(float(r[j]))
            if j >= self.WARM:
                for k in self.STREAMS:
                    S[k].append(raw[k])
        self._calibrate(S)
        self._setup_buffers()

    def _step(self, x):
        I = 1.0 if abs(x) > self.q else 0.0
        Iu = 1.0 if x > self.qu else 0.0
        Il = 1.0 if x < self.ql else 0.0
        raw = {"c1": self.last[-1] * (I - self.p),
               "c5": (1.0 if max(self.last) > 0.0 else 0.0) * (I - self.p),
               "cu": self.lu * (Iu - self.pu),
               "cl": self.ll * (Il - self.pl)}
        self.p += self.A * (I - self.p)
        self.pu += self.A * (Iu - self.pu)
        self.pl += self.A * (Il - self.pl)
        self.last = self.last[1:] + [I]
        self.lu, self.ll = Iu, Il
        return raw

    def update(self, ctx):
        return self._emit(self._step(self._innov(float(ctx.zr))))


# AggKurtosisFamily (aggkurt): rejected
class AggKurtosisFamily(_Lit5Base):
    """Do heavy tails survive aggregation? For tau = 1, 4, 16 the tau-step change of the clipped
    level, y = (z_t - z_{t-tau}) / sd_hist(tau), gives a kurtosis-type stream q = y^4 / v^2 with
    v = EWMA_0.02 of y^2 (lagged) -- scale-free, so a volatility change alone does not move it.
    q1 / q4 / q16, and qr = standardized q16 - standardized q1: the CONTRAST (tails persisting across
    horizons = intermittency / multifractality), given as its own column because trees split one
    column at a time. (Kantelhardt et al. 2002; Cont 2001, aggregational Gaussianity.)"""
    name = "aggkurt"
    STREAMS = ("q1", "q4", "q16", "qr")
    TAUS = (1, 4, 16)
    A, ZC, QCAP = 0.02, 8.0, 400.0

    def init(self, hist):
        z = np.clip(_zr_hist(hist), -self.ZC, self.ZC)
        T = max(self.TAUS)
        self.sd = {}
        for tau in self.TAUS:
            d = z[tau:] - z[:-tau] if z.size > tau + 2 else np.ones(3)
            self.sd[tau] = max(float(d.std()), 1e-6)
        self.v = {tau: 1.0 for tau in self.TAUS}
        self.zb = [float(v) for v in z[-T:]]
        Q = {tau: [] for tau in self.TAUS}
        buf = [float(v) for v in z[:T]]
        for i in range(T, z.size):
            q = self._q(buf, float(z[i]))
            if i >= T + self.WARM:
                for tau in self.TAUS:
                    Q[tau].append(q[tau])
            buf = buf[1:] + [float(z[i])]
        S = {"q%d" % tau: Q[tau] for tau in self.TAUS}
        self.STREAMS = ("q1", "q4", "q16")
        self._calibrate(S)                                       # first pass: the three q's
        std = {tau: (np.asarray(Q[tau]) - self.mu_s["q%d" % tau]) / self.sd_s["q%d" % tau]
               for tau in self.TAUS}
        S["qr"] = (std[16] - std[1]) if len(Q[1]) else []
        self.STREAMS = AggKurtosisFamily.STREAMS
        self._calibrate(S)                                       # second pass: all four
        # the init loop warmed self.v; restart the online buffer from the true history tail
        self.zb = [float(v) for v in z[-T:]]
        self._setup_buffers()

    def _q(self, buf, z):
        out = {}
        for tau in self.TAUS:
            y = (z - buf[-tau]) / self.sd[tau]
            v = max(self.v[tau], 1e-4)
            out[tau] = min(y ** 4, self.QCAP) / (v * v)
            self.v[tau] += self.A * (y * y - self.v[tau])
        return out

    def update(self, ctx):
        z = min(max(float(ctx.zr), -self.ZC), self.ZC)
        q = self._q(self.zb, z)
        self.zb = self.zb[1:] + [z]
        raw = {"q%d" % tau: q[tau] for tau in self.TAUS}
        raw["qr"] = ((raw["q16"] - self.mu_s["q16"]) / self.sd_s["q16"]
                     - (raw["q1"] - self.mu_s["q1"]) / self.sd_s["q1"])
        return self._emit(raw)


# PeriodicityFamily (period): rejected
class PeriodicityFamily(_Lit5Base):
    """Does a sharp cycle appear or vanish? Sliding DFT (W = 128) of the AR innovations -- whitened,
    so the broad AR spectrum does not masquerade as a peak -- at the 63 interior Fourier frequencies.
    g  : Fisher's (1929) g = max periodogram share (sum of shares = 1)
    pk : log share at the HISTORY's dominant frequency (does the history's own cycle persist?)
    Band energies (temporal, wavvar) smear a narrow peak; g measures its concentration.
    The DFT is updated in O(K) per step and recomputed exactly every W steps (no drift)."""
    name = "period"
    STREAMS = ("g", "pk")
    W = 128
    RC = 6.0
    MAXWIN = 2048

    def init(self, hist):
        _, r = self._setup_innov(hist)
        x = np.clip(r, -self.RC, self.RC)
        W = self.W
        K = W // 2 - 1
        self.k = np.arange(1, W // 2)
        self.rot = np.exp(2j * np.pi * self.k / W)
        if x.size >= 2 * W + 32:
            M = np.lib.stride_tricks.sliding_window_view(x, W)[-self.MAXWIN:]
            P = np.abs(np.fft.rfft(M, axis=1)[:, 1:W // 2]) ** 2
            sh = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
            # pick the dominant frequency on the EARLIER half of the windows and calibrate pk on the
            # later half: picking and calibrating on the same windows is a winner's curse (on a
            # peak-free history the chosen bin looks stronger than it is, so pk drifts negative)
            h = sh.shape[0] // 2
            self.kstar = int(np.argmax(sh[:h].mean(axis=0)))
            self._calibrate({"g": sh.max(axis=1), "pk": np.log(np.maximum(sh[h:, self.kstar], 1e-12))})
        else:                       # short history: white-noise (exponential ordinates) approximations
            self.kstar = 0
            self.mu_s = {"g": (math.log(K) + 0.5772) / K, "pk": -math.log(K) - 0.5772}
            self.sd_s = {"g": 1.2825 / K, "pk": 1.2825}
        seed = x[-W:] if x.size >= W else np.r_[np.zeros(W - x.size), x]
        self.xb = np.ascontiguousarray(seed, dtype=np.float64)   # ring buffer, oldest at self.pos
        self.pos = 0
        self.X = np.fft.rfft(self.xb)[1:W // 2]
        self.since = 0
        self._setup_buffers()

    def update(self, ctx):
        x = min(max(self._innov(float(ctx.zr)), -self.RC), self.RC)
        old = self.xb[self.pos]
        self.xb[self.pos] = x
        self.pos = (self.pos + 1) % self.W
        self.since += 1
        if self.since >= self.W:                                  # exact refresh
            self.X = np.fft.rfft(np.roll(self.xb, -self.pos))[1:self.W // 2]
            self.since = 0
        else:
            self.X = self.rot * (self.X - old + x)
        P = self.X.real ** 2 + self.X.imag ** 2
        tot = float(P.sum())
        if tot <= 1e-12:
            raw = {"g": self.mu_s["g"], "pk": self.mu_s["pk"]}
        else:
            raw = {"g": float(P.max()) / tot, "pk": math.log(max(float(P[self.kstar]) / tot, 1e-12))}
        return self._emit(raw)


# RecordingProcessFamily (recproc): rejected
class RecordingProcessFamily(_Lit5Base):
    """Did the way the series is MEASURED change -- rounding / tick size, stale repeated values?
    Every other feature treats values as continuous. Equal raw values give bit-equal zr, so ties
    are exact. (Lesmond, Ogden & Trzcinka 1999 on zero returns; Bandi, Pirino & Reno 2017 on
    staleness.)
    tie  : 1[z_t == z_{t-1}]                    rep : 1[z_t equals any of the previous 5 values]
    fine : 1[0 < |dz| < 1st percentile of the history's nonzero |dz|]   (finer grid than before)
    Dead (constant) on continuous data -- check the tie rate before spending a CV run on it."""
    name = "recproc"
    STREAMS = ("tie", "rep", "fine")

    def init(self, hist):
        z = _zr_hist(hist)
        d = np.abs(np.diff(z))
        nz = d[d > 0]
        self.g1 = float(np.quantile(nz, 0.01)) if nz.size >= 20 else 0.0
        rep = np.zeros(max(z.size - 5, 0), dtype=bool)
        for k in range(1, 6):
            rep |= z[5:] == z[5 - k:z.size - k]
        self._calibrate({"tie": (d == 0).astype(float),
                         "rep": rep.astype(float),
                         "fine": ((d > 0) & (d < self.g1)).astype(float)})
        self.zb = [float(v) for v in z[-5:]]
        self._setup_buffers()

    def update(self, ctx):
        z = float(ctx.zr)
        d = abs(z - self.zb[-1])
        raw = {"tie": 1.0 if d == 0.0 else 0.0,
               "rep": 1.0 if z in self.zb else 0.0,
               "fine": 1.0 if 0.0 < d < self.g1 else 0.0}
        self.zb = self.zb[1:] + [z]
        return self._emit(raw)
