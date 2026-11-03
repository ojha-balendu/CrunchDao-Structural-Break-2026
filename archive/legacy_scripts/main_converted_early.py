# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

#import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')


#%pip install crunch-cli --upgrade --quiet --progress-bar off
#!crunch setup-notebook structural-break-real-time <YOUR_CRUNCH_TOKEN>


import bisect
import copy
import math
import os
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import special as _sp

def ndtri(x):                 # plain-Python wrapper: scipy's ufunc has __module__ = None,
    return _sp.ndtri(x)       # so cloudpickle would mis-resolve it to __mp_main__ in a
                              # loky worker. The wrapper pickles by value and carries a
                              # proper module reference to scipy.special instead.
from sklearn.metrics import roc_auc_score


import crunch

#crunch_tools = crunch.load_notebook()


#train_data, test_data = crunch_tools.load_data()
#print("train series:", len(train_data))


# =====================================================================
#  HISTORICAL REFERENCE  (one pass over the historical segment)
# =====================================================================
def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    d = float(np.dot(x, x))
    if d <= 0.0:
        return np.zeros(nlags)
    return np.array([float(np.dot(x[k:], x[:n - k])) / d for k in range(1, nlags + 1)])


def _yule_walker(r: np.ndarray, p: int) -> np.ndarray:
    R = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i != j:
                R[i, j] = r[abs(i - j) - 1]
    try:
        a = np.linalg.solve(R, r[:p])
    except np.linalg.LinAlgError:
        a = np.zeros(p)
    if not np.all(np.isfinite(a)):
        a = np.zeros(p)
    return a


def _pit_z_of(values: np.ndarray) -> np.ndarray:
    """Map an array to its own PIT-normal scores (used for historical diagnostics)."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ndtri((ranks + 0.5) / n)


def _l_moments(sorted_x: np.ndarray):
    """Hosking (1990) unbiased sample L-moments l1..l4 from a SORTED array."""
    n = sorted_x.size
    if n < 4:
        return (0.0, 1e-6, 0.0, 0.0)
    x = sorted_x
    i = np.arange(1, n + 1, dtype=np.float64)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n if n > 2 else 0.0
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n if n > 3 else 0.0
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return (float(l1), float(max(l2, 1e-12)), float(l3), float(l4))


class HistoricalRef:
    """Everything derived once from the (break-free) historical segment."""

    def __init__(self, x_hist):
        xh = np.asarray(x_hist, dtype=np.float64).ravel()
        xh = xh[np.isfinite(xh)]
        if xh.size < 4 * AR_P + 8:                       # pathological guard
            xh = np.concatenate([xh, np.zeros(4 * AR_P + 8 - xh.size)])

        n = xh.size
        self.n = n
        self.mu = float(xh.mean())
        self.sd = max(float(xh.std(ddof=1)), 1e-12)
        med = float(np.median(xh))
        self.mad = max(1.4826 * float(np.median(np.abs(xh - med))), 1e-12)
        self.xmin = float(xh.min())
        self.xmax = float(xh.max())

        # ---- PIT table for the raw value -----------------------------
        self.xs_l = np.sort(xh).tolist()
        self.ztab = ndtri((np.arange(n + 1) + 0.5) / (n + 1)).tolist()

        # ---- standardized moments / tail rates -----------------------
        zr = (xh - self.mu) / self.sd
        self.skew = float((zr ** 3).mean())
        self.kurt = float((zr ** 4).mean() - 3.0)
        self.p2 = max(float((np.abs(zr) > 2.0).mean()), 1.0 / (n + 1))
        self.p3 = max(float((np.abs(zr) > 3.0).mean()), 1.0 / (n + 1))

        # ---- AR(p) whitening ----------------------------------------
        r = _acf(xh, max(AR_P, 2))
        self.acf1, self.acf2 = float(r[0]), float(r[1])
        a = _yule_walker(r, AR_P)
        self.ar_l = [float(v) for v in a]

        c = xh - self.mu
        res = c[AR_P:].copy()
        for k in range(AR_P):
            res -= self.ar_l[k] * c[AR_P - 1 - k: c.size - 1 - k]
        self.rsd = max(float(res.std(ddof=1)), 1e-12)

        m = res.size
        self.es_l = np.sort(res).tolist()
        self.etab = ndtri((np.arange(m + 1) + 0.5) / (m + 1)).tolist()

        # ---- lag seeds so the online stream starts warm ---------------
        zh = _pit_z_of(xh)
        rz = _acf(zh, 2)
        self.zacf1, self.zacf2 = float(rz[0]), float(rz[1])
        self.z_seed1 = float(zh[-1])
        self.z_seed2 = float(zh[-2])
        self.tail_x = [float(v) for v in xh[-AR_P:]]

        eh = _pit_z_of(res)
        self.e_seed1 = float(eh[-1])
        self.e_seed2 = float(eh[-2])

        # ---- context block (constant for the whole series) ------------
        self.context = [
            float(n), math.log(n),
            self.skew, self.kurt, self.acf1, self.acf2,
            self.ar_l[0], self.ar_l[1], self.ar_l[2],
            self.rsd / self.sd, self.sd / self.mad,
            self.p2, self.p3, self.zacf1,
        ]

        # ---- Tier 1-4: deterministic PIT-normal quantile grids, used as the
        # fixed "historical sample" side of every two-sample test (see 1b markdown).
        # REF_MAX lives in the CONFIG cell below: the per-family
        # cache key (section 2b) has to hash it.
        _zr = np.asarray(self.ztab[1:-1], dtype=np.float64)
        _er = np.asarray(self.etab[1:-1], dtype=np.float64)
        if _zr.size > REF_MAX:
            _zr = np.ascontiguousarray(_zr[np.linspace(0, _zr.size - 1, REF_MAX).astype(int)])
        if _er.size > REF_MAX:
            _er = np.ascontiguousarray(_er[np.linspace(0, _er.size - 1, REF_MAX).astype(int)])
        self.z_ref, self.e_ref = _zr, _er
        self.z_ref_lmom = _l_moments(np.sort(self.z_ref))
        self.e_ref_lmom = _l_moments(np.sort(self.e_ref))


# =====================================================================
#  CONFIG
# =====================================================================
# @crunch/keep:on
WINDOWS = (32, 64, 256)      # trailing windows = candidate post-break lengths
TWOSAMPLE_CHANNELS = ("e",)      # was ("z","e"); all 30 top diagnostic features were e-channel
ALPHAS  = (0.02, 0.08, 0.25)     # EWMA decays: slow / medium / fast
DRIFTS  = (0.25, 0.75, 1.50)     # Page-CUSUM reference drifts, in sigma units
ACF_W   = 64                     # window for streaming autocorrelation
SLOPE_W = 64                     # window for streaming regression slope
SIGN_W  = (32, 128)              # windows for the sign / runs statistics
TAIL_W  = 64                     # window for tail-exceedance counts
BIN_W   = 128                    # window for the binned chi-square (shape change)
NBINS   = 8
AR_P    = 3                      # AR order for the whitening channel
REF_MAX = 256                    # knots in the deterministic PIT-normal
                                 # quantile grid that is the reference side
                                 # of every two-sample test. The grid is
                                 # smooth, not raw data, so 256 knots
                                 # reproduce it well inside the sampling
                                 # noise of any window of length <= 256.
                                 # Module-level so the per-family cache key
                                 # (section 2b) can hash it.

CH = ("z", "z2", "e", "e2")      # channels: loc / scale / loc-of-residual / scale-of-residual
LBUF = max(max(WINDOWS), BIN_W, ACF_W, SLOPE_W, max(SIGN_W), TAIL_W) + 1
INV_SQRT2 = 0.7071067811865476
# @crunch/keep:off

# FEATURES / N_FEATURES are no longer hand-written here -- they're derived from
# REGISTRY once every FeatureFamily has been defined and registered (see the end
# of section 1i, after the Tier 1-4 families). Everything between here and there
# is registry plumbing + the actual family classes.


# @crunch/keep:on
CH_IDX = {name: i for i, name in enumerate(CH)}  # z:0 z2:1 e:2 e2:3
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


class ChannelState:
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


class Ctx:
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


def stride_for(cost: str, t: int) -> int:
    """Pure function of t only -- every series at step t is equally stale, so
    within-t ranking is unaffected. Same argument as PREDICT_STRIDE / _stride_for
    in infer(): free speed, no data leakage, no series-dependent staleness."""
    if cost == "O1":
        return 1
    if cost == "sorted":
        if t < 128:
            return 1
        if t < 512:
            return 4
        return 16
    # "heavy"
    if t < 128:
        return 2
    if t < 512:
        return 8
    return 32


def _safe(v: float) -> float:
    return v if math.isfinite(v) else 0.0


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0.0 or not math.isfinite(num) or not math.isfinite(den):
        return 0.0
    v = num / den
    return v if math.isfinite(v) else 0.0


class FeatureFamily:
    """Base class for one entry in REGISTRY. init() does one-off O(n_H) work on
    the historical segment; update() consumes ctx (the shared per-step state) and
    returns len(self.names()) floats, in emit order.

    IMPORTANT: REGISTRY holds one prototype *instance* per family (so a family
    with constructor args, like ChannelBankFamily(channel), only needs to be
    built once). init()/update() mutate `self` (e.g. TailFamily's ring buffers and
    running counts), so every StreamFeatures MUST work on its own
    `clone()` of each family, never the REGISTRY prototype directly --
    build_dataset() runs StreamFeatures for different series concurrently
    (joblib threading backend), and sharing mutable family state across those
    threads would silently corrupt features across series."""
    name: str = "base"
    channels: tuple = ("z", "e")
    cost: str = "O1"
    version: int = 1                  # bump by hand after editing this family's body:
                                      # it is part of the per-family cache key (2b), and
                                      # inspect.getsource() usually cannot see a class
                                      # defined in a notebook cell, so `version` is the
                                      # only thing that invalidates stale columns.

    def names(self) -> List[str]:
        raise NotImplementedError

    def init(self, hist: HistoricalRef) -> None:
        pass

    def update(self, ctx: Ctx) -> List[float]:
        raise NotImplementedError

    def clone(self) -> "FeatureFamily":
        return copy.deepcopy(self)


# @crunch/keep:on
REGISTRY: List[FeatureFamily] = []
# @crunch/keep:off


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


# Only the two SQUARED (scale) channels are registered as feature banks. The raw
# location banks "z" / "e" were pruned -- but ChannelState is still maintained for
# every channel in CH, because LMomentsFamily reads "z" and "e" and the two-sample
# families read TWOSAMPLE_CHANNELS. Don't prune the channel plumbing along with the
# banks.
# @crunch/keep:on
BANK_CHANNELS = ("z2", "e2")

for _ch in BANK_CHANNELS:
    REGISTRY.append(ChannelBankFamily(_ch))
REGISTRY.append(TailFamily())
REGISTRY.append(TimeFamily())
# @crunch/keep:off

#print("ported families registered:", len(REGISTRY),
#      "| features so far:", sum(len(f.names()) for f in REGISTRY))


# =====================================================================
#  STREAMING FEATURE EXTRACTOR  (thin registry driver)
# =====================================================================
class StreamFeatures:
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

        if t == 0:
            # Every family computes at t == 0 whatever its stride, so this is the one
            # step where the width check is free. It replaces the original driver's
            # per-step `assert len(r) == len(f.names())`.
            for f, nout in zip(self.families, self._nout):
                if len(f._last) != nout:
                    raise AssertionError(
                        "%s.update() returned %d values, names() has %d" % (f.name, len(f._last), nout))
        return out


def _mean_var(x: np.ndarray):
    n = x.size
    if n < 2:
        return 0.0, 1e-6
    mu = float(x.mean())
    v = float(x.var(ddof=1))
    return mu, max(v, 1e-12)


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Tie-averaged (1-based) ranks of one array, via argsort over the whole array.

    No family in the pruned registry calls this any more. Every rank-based family
    (fligner, ansari, mood) now ranks with _avg_ranks_of(), which searchsorts into a
    reference that was sorted once in init() instead of concatenating and re-sorting
    window ++ reference on every evaluation; sigtuk needs stable-sort POSITIONS rather
    than averaged ranks and derives them the same way.

    It stays for two reasons: it is what a new family will reach for before there is a
    fast path for it, and it is the definition _avg_ranks_of() is checked against in
    tools/test_phase2.py."""
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


# =====================================================================
#  Exact fast paths shared by the two-sample families (Phase 2)
#
#  Every one of these is an algebraic rewrite, not an approximation: the only
#  difference from the code they replace is floating-point summation order.
#  `tools/reference_impl.py` holds the arithmetic they replaced and
#  tools/test_phase2.py diffs them at atol=rtol=1e-8.
# =====================================================================
# @crunch/keep:on
_COEF_CACHE = {}
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


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


class RefCachedTwoSample(TwoSampleFamily):
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


class FTestFamily(RefCachedTwoSample):
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


# @crunch/keep:on
REGISTRY.extend([
    FTestFamily(), BartlettFamily(), LeveneFamily(), FlignerKilleenFamily(),
])
# @crunch/keep:off
#print("Tier 1 registered | features so far:", sum(len(f.names()) for f in REGISTRY))


class LMomentsFamily(FeatureFamily):
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


class _RankScaleFamily(RefCachedTwoSample):
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


# @crunch/keep:on
REGISTRY.extend([
    LMomentsFamily(), IQRRatioFamily(),
    AnsariBradleyFamily(), MoodFamily(), SiegelTukeyFamily(),
])
# @crunch/keep:off
#print("Tier 2 registered | features so far:", sum(len(f.names()) for f in REGISTRY))


class WassersteinFamily(RefCachedTwoSample):
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


# @crunch/keep:on
REGISTRY.extend([
    WassersteinFamily(), EnergyDistanceFamily(), MMDRbfFamily(),
])
# @crunch/keep:off
#print("Tier 3 registered | features so far:", sum(len(f.names()) for f in REGISTRY))


class DiffEntropyFamily(TwoSampleFamily):
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


# @crunch/keep:on
REGISTRY.extend([
    DiffEntropyFamily(),
])

# =====================================================================
#  REGISTRY IS NOW COMPLETE -- derive FEATURES / N_FEATURES from it
# =====================================================================
FEATURES = [nm for f in REGISTRY for nm in f.names()]
N_FEATURES = len(FEATURES)
# @crunch/keep:off
#assert len(set(FEATURES)) == N_FEATURES, "duplicate feature names in REGISTRY"
#print("REGISTRY: %d families, %d total features" % (len(REGISTRY), N_FEATURES))


def resolve_active_families(active_families):
    """Validate a family-name subset against REGISTRY -- fails loudly on a typo
    instead of silently emitting zero columns for that family."""
    if active_families is None:
        return None
    known = {f.name for f in REGISTRY}
    unknown = set(active_families) - known
    assert not unknown, "unknown family name(s) in ACTIVE_FAMILIES: %s" % sorted(unknown)
    return set(active_families)


# The registry IS the selection now: the Phase 1 cull removed the 25 families that
# sat at or below the permutation-null cutoff, so "every registered family" and
# "every family worth extracting" are the same set. None = every family; set it to
# a subset of
#     e2, z2, tail, time, ftest, bartlett, levene, fligner, lmom, iqr,
#     ansari, mood, sigtuk, wass, energy, mmd, entropy
# only to trade accuracy for extraction speed.
# @crunch/keep:on
ACTIVE_FAMILIES = resolve_active_families(None)
# @crunch/keep:off


import math
from scipy.special import gammaln as _gammaln

# @crunch/keep:on
_LOG_2PI = math.log(2.0 * math.pi)
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


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

# @crunch/keep:on
pass  # anchor: crunch-convert only sees keep markers above a simple statement
if not any(f.name == "bocpd" for f in REGISTRY):
    REGISTRY.append(BOCPDVarianceFamily())
FEATURES = [nm for f in REGISTRY for nm in f.names()]
N_FEATURES = len(FEATURES)
# @crunch/keep:off
#print("REGISTRY: %d families, %d total features" % (len(REGISTRY), N_FEATURES))  # expect 18 and 85
#print("REGISTRY: %d families, %d total features" % (len(REGISTRY), N_FEATURES))  # expect 18 and 85
#print(len(REGISTRY), N_FEATURES, [f.name for f in REGISTRY][-1])


# @crunch/keep:on
KEEP_EVERY = ((24, 1), (128, 2), (512, 4), (10 ** 9, 8))   # (up_to_t, stride)
# @crunch/keep:off

import functools
import multiprocessing
import sys

# @crunch/keep:on
_USE_GLOBAL_ACTIVE = object()   # sentinel: "use whatever ACTIVE_FAMILIES is right now"

# "auto" | "loky" | "fork" | "threading" | "serial"
# "auto" is the offline default: try loky, fall back to a fork Pool, then to serial.
# Naming one explicitly honors it with NO fallback, which is what you want when you
# are diagnosing a backend rather than getting work done.
EXTRACT_BACKEND = "auto"
FORK_CHUNKSIZE = 8
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


def _stride_at(t: int) -> int:
    for up_to, stride in KEEP_EVERY:
        if t < up_to:
            return stride
    return KEEP_EVERY[-1][1]


def extract_series(x_hist, x_online, tau=None, subsample=True, active_families=_USE_GLOBAL_ACTIVE):
    """Run the extractor over one series. Returns (X, y, tvec, w).

    active_families: _USE_GLOBAL_ACTIVE (default) resolves to the current
    ACTIVE_FAMILIES global at call time (this is what train()/infer() want --
    fast, submission-path extraction). Pass None explicitly to force the FULL
    registry regardless of the global (this is what the offline cache/diagnostic/
    ablate loop wants -- always extract everything, select columns after)."""
    fams = ACTIVE_FAMILIES if active_families is _USE_GLOBAL_ACTIVE else active_families
    fx = StreamFeatures(x_hist, active_families=fams)
    rows, ys, ts, ws = [], [], [], []
    for t, point in enumerate(x_online):
        f = fx.update(point)
        if subsample:
            stride = _stride_at(t)
            if t % stride != 0:
                continue
            ws.append(float(stride))
        else:
            ws.append(1.0)
        rows.append(list(f))
        ys.append(0 if (tau is None or t < tau) else 1)
        ts.append(t)
    X = np.asarray(rows, dtype=np.float32)
    return X, np.asarray(ys, np.int8), np.asarray(ts, np.int32), np.asarray(ws, np.float32)


def _extract_one(rec, subsample=True, active_families=None):
    """Module-level on purpose: multiprocessing.Pool pickles the callable BY REFERENCE,
    and a closure defined inside build_dataset cannot be pickled at all."""
    did, xh, xo, tau = rec
    X, y, t, w = extract_series(xh, xo, tau, subsample=subsample,
                                 active_families=active_families)
    g = np.full(t.shape, did, dtype=np.int64)
    return X, y, t, w, g


def _loky_can_start() -> bool:
    """loky spawns fresh interpreters that re-import __main__. crunch_tools.test()
    installs a process initializer pointing __main__ at a module path that does not
    exist in this environment, and every worker then dies with
    FileNotFoundError: /kaggle/working/main.py before running any of our code.

    If __main__ claims a __file__ that isn't on disk, spawning is going to fail, so
    don't pay for the attempt."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return not (path is not None and not os.path.exists(path))


def _backend_chain(backend):
    if backend != "auto":
        return [backend]                       # explicit request: no silent fallback
    chain = []
    if _loky_can_start():
        chain.append("loky")
    if "fork" in multiprocessing.get_all_start_methods():
        chain.append("fork")
    chain.append("serial")
    return chain


def _run_fork(fn, items, n_jobs):
    ctx = multiprocessing.get_context("fork")
    procs = None if n_jobs in (-1, 0) else n_jobs
    with ctx.Pool(processes=procs) as pool:
        return pool.map(fn, items, chunksize=FORK_CHUNKSIZE)


def _stack(res):
    return (np.concatenate([r[0] for r in res]),
            np.concatenate([r[1] for r in res]),
            np.concatenate([r[2] for r in res]),
            np.concatenate([r[3] for r in res]),
            np.concatenate([r[4] for r in res]))


def build_dataset(datasets, subsample=True, max_series=None, n_jobs=-1, verbose=1,
                   active_families=_USE_GLOBAL_ACTIVE, backend=None):
    """Extract many series in parallel. `backend` overrides EXTRACT_BACKEND for this
    call; None means "use the global".

    Each family is .clone()-d off the REGISTRY prototype (see FeatureFamily.clone()),
    so concurrent extraction never shares mutable per-series state.
    """
    items = list(datasets)[: max_series] if max_series else list(datasets)
    if not items:
        raise ValueError("build_dataset got no series")

    # Resolve the sentinel HERE, in the parent. It is a bare object(); pickled into a
    # worker it becomes a different object, the `is` test in extract_series fails, and
    # the sentinel gets read as a set of family names.
    fams = ACTIVE_FAMILIES if active_families is _USE_GLOBAL_ACTIVE else active_families
    worker = functools.partial(_extract_one, subsample=subsample, active_families=fams)

    if n_jobs == 1 or len(items) == 1:
        return _stack([worker(r) for r in items])

    chain = _backend_chain(backend or EXTRACT_BACKEND)
    errors = []
    for b in chain:
        try:
            if b == "serial":
                if verbose:
                    print("build_dataset: serial over %d series" % len(items))
                return _stack([worker(r) for r in items])
            if b == "fork":
                if verbose:
                    print("build_dataset: fork Pool over %d series (chunksize %d)" % (len(items), FORK_CHUNKSIZE))
                return _stack(_run_fork(worker, items, n_jobs))
            res = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=8, backend=b)(
                joblib.delayed(worker)(r) for r in items)
            return _stack(res)
        except Exception as exc:
            errors.append((b, type(exc).__name__, str(exc)[:200]))
            if b == chain[-1]:
                raise RuntimeError(
                    "every extraction backend failed: %s" % "; ".join("%s -> %s: %s" % e for e in errors)) from exc
            print("build_dataset: backend %r failed (%s: %s) -- falling back to %r" % (b, type(exc).__name__, str(exc)[:160], chain[chain.index(b) + 1]))


import fnmatch
import glob
import hashlib
import inspect
import json
import shutil

# @crunch/keep:on
CACHE_DIR = "cache"
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


def feature_to_family():
    """Exact feature -> family-name map, built from REGISTRY at CALL time.

    Rebuilt on every call on purpose: `research_step` (section 3d) adds and removes
    families from REGISTRY between calls, and a dict frozen when this cell ran would
    silently miss every family registered afterwards.

    Built from REGISTRY, not guessed from the feature string -- string-splitting
    breaks for any family whose .name itself contains an underscore. The archived
    "perm_ent" family is the worked example: "perm_ent_z_w16" would split to "perm".
    """
    return {nm: f.name for f in REGISTRY for nm in f.names()}


#FEATURE_TO_FAMILY = feature_to_family()   # convenience snapshot for interactive use;
                                          # every function below calls the function


# =====================================================================
#  Cache keys
# =====================================================================
def _short_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _extractor_config():
    """The module-level constants the extracted numbers depend on. Changing any one of
    these changes what every family emits, so all of them go into every family key."""
    return {
        "WINDOWS": list(WINDOWS),
        "TWOSAMPLE_CHANNELS": list(TWOSAMPLE_CHANNELS),
        "ALPHAS": list(ALPHAS),
        "DRIFTS": list(DRIFTS),
        "TAIL_W": TAIL_W,
        "BIN_W": BIN_W,
        "NBINS": NBINS,
        "AR_P": AR_P,
        "REFSUB_CAP": RefSub.CAP,
        "REF_MAX": REF_MAX,
    }


def sample_key(series_ids, subsample=True, seed=0) -> str:
    """Identity of the ROW SET: which series, at what subsample schedule, which seed."""
    return _short_hash({
        "series_ids": sorted(str(s) for s in series_ids),
        "schedule": [list(p) for p in KEEP_EVERY] if subsample else [],
        "subsample": bool(subsample),
        "seed": seed,
    })


def family_key(fam) -> str:
    """Identity of one family's OUTPUT: name, column names, a manual version counter,
    its source, and the extractor config it reads.

    inspect.getsource() has no file to read for a class defined in a notebook cell and
    raises there; the fallback is silent, which is exactly why `version` exists. Bump
    `version` by hand after editing a family body, or the stale columns are reused.
    """
    try:
        source = inspect.getsource(type(fam))
    except Exception:
        source = ""
    return _short_hash({
        "name": fam.name,
        "names": list(fam.names()),
        "version": int(getattr(fam, "version", 1)),
        "source": source,
        "config": _extractor_config(),
    })


def _sample_dir(skey, cache_dir=CACHE_DIR):
    return os.path.join(cache_dir, skey)


def _fam_path(d, fam):
    return os.path.join(d, "fam__%s__%s.npy" % (fam.name, family_key(fam)))


# =====================================================================
#  The store
# =====================================================================
def get_cached_matrix(datasets, subsample=True, max_series=None, seed=0,
                       cache_dir=CACHE_DIR, families=None, verbose=True):
    """Assemble (X, y, t, w, g, feature_names) for `datasets`, extracting only the
    families that are not already on disk.

    Same return signature as before, and the columns come back in REGISTRY order, so
    quick_train / ablate / cv_evaluate / select_features need no change.

    `families=None` means every registered family; pass a set of names to assemble a
    subset (it still only extracts what is missing from that subset).
    """
    global FEATURE_TO_FAMILY
    FEATURE_TO_FAMILY = feature_to_family()

    items = list(datasets)[:max_series] if max_series else list(datasets)
    series_ids = [r[0] for r in items]
    skey = sample_key(series_ids, subsample, seed)
    d = _sample_dir(skey, cache_dir)
    os.makedirs(d, exist_ok=True)

    wanted = [f for f in REGISTRY if families is None or f.name in families]
    meta_path = os.path.join(d, "meta.joblib")
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else None

    have, missing = {}, []
    for f in wanted:
        p = _fam_path(d, f)
        if meta is not None and os.path.exists(p):
            have[f.name] = np.load(p)
        else:
            missing.append(f)

    if missing:
        if verbose:
            print("cache %s | extracting %d of %d families over %d series: %s" % (skey, len(missing), len(wanted), len(items),
                     ", ".join(f.name for f in missing)))
        Xm, y, t, w, g = build_dataset(items, subsample=subsample,
                                        active_families={f.name for f in missing},
                                        verbose=0)
        if meta is None:
            meta = {"y": y, "t": t, "w": w, "g": g,
                    "series_ids": [str(s) for s in series_ids],
                    "KEEP_EVERY": [list(p) for p in KEEP_EVERY],
                    "subsample": bool(subsample), "seed": seed}
            joblib.dump(meta, meta_path)
        else:
            # A later extraction has to line up row-for-row with the rows the cached
            # families were built against. If it does not, the assembled matrix would
            # silently splice columns from two different row sets together.
            assert np.array_equal(y, meta["y"]), "fresh y != meta.joblib y"
            assert np.array_equal(t, meta["t"]), "fresh t != meta.joblib t"
            assert np.array_equal(g, meta["g"]), "fresh g != meta.joblib g"

        off = 0
        for f in missing:
            k = len(f.names())
            block = np.ascontiguousarray(Xm[:, off:off + k], dtype=np.float32)
            off += k
            for stale in glob.glob(os.path.join(d, "fam__%s__*.npy" % f.name)):
                os.remove(stale)          # a previous version of this family's columns
            np.save(_fam_path(d, f), block)
            have[f.name] = block
        assert off == Xm.shape[1], (
            "column accounting: consumed %d of %d extracted columns" % (off, Xm.shape[1]))
    elif verbose:
        print("cache %s | all %d families cached (%d series, %d rows)" % (skey, len(wanted), len(items), len(meta["y"])))

    names = []
    blocks = []
    for f in wanted:
        blocks.append(have[f.name])
        names.extend(f.names())
    X = (np.concatenate(blocks, axis=1) if blocks
         else np.empty((len(meta["y"]), 0), dtype=np.float32))
    return X, meta["y"], meta["t"], meta["w"], meta["g"], names


def clear_cache(family=None, cache_dir=CACHE_DIR):
    """clear_cache() wipes everything; clear_cache("mmd") drops one family's columns
    from every sample, leaving meta/null and the other families intact."""
    if not os.path.isdir(cache_dir):
        print("no cache directory at", cache_dir)
        return
    if family is None:
        shutil.rmtree(cache_dir)
        print("cache cleared:", cache_dir)
        return
    n = 0
    for skey in sorted(os.listdir(cache_dir)):
        for p in glob.glob(os.path.join(cache_dir, skey, "fam__%s__*.npy" % family)):
            os.remove(p)
            n += 1
    print("cleared %d cached file(s) for family %r" % (n, family))


def cache_status(datasets, subsample=True, max_series=None, seed=0, cache_dir=CACHE_DIR):
    """What is on disk for this sample, family by family."""
    items = list(datasets)[:max_series] if max_series else list(datasets)
    skey = sample_key([r[0] for r in items], subsample, seed)
    d = _sample_dir(skey, cache_dir)
    print("sample %s (%d series) -> %s" % (skey, len(items), d))
    for f in REGISTRY:
        p = _fam_path(d, f)
        print("   %-10s %-8s %s" % (f.name, "CACHED" if os.path.exists(p) else "missing",
                                     os.path.basename(p)))
    return skey


# ---- permutation-null side-cache, keyed by the feature set it was computed over ----
def _null_path(skey, cache_dir=CACHE_DIR):
    return os.path.join(_sample_dir(skey, cache_dir), "null.joblib")


def load_null(skey, key, cache_dir=CACHE_DIR):
    p = _null_path(skey, cache_dir)
    if not os.path.exists(p):
        return None
    return joblib.load(p).get(key)


def save_null(skey, key, value, cache_dir=CACHE_DIR):
    p = _null_path(skey, cache_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    store = joblib.load(p) if os.path.exists(p) else {}
    store[key] = value
    joblib.dump(store, p)


def select_features(feature_names, families=None, features=None, patterns=None,
                     min_ts_auc=None, diag=None):
    """Resolve any combination of family names / explicit feature names / glob
    patterns / a diagnostic-driven min_ts_auc rule into column indices. Returns
    (idx, matching_names). No selector given -> every column."""
    feature_names = list(feature_names)
    f2f = feature_to_family()
    chosen = set()
    any_selector = False
    if families:
        any_selector = True
        fams = set(families)
        chosen |= {i for i, nm in enumerate(feature_names) if f2f.get(nm) in fams}
    if features:
        any_selector = True
        want = set(features)
        chosen |= {i for i, nm in enumerate(feature_names) if nm in want}
    if patterns:
        any_selector = True
        for pat in patterns:
            chosen |= {i for i, nm in enumerate(feature_names) if fnmatch.fnmatch(nm, pat)}
    if min_ts_auc is not None:
        any_selector = True
        assert diag is not None, "min_ts_auc requires diag= a diagnostic DataFrame (section 3)"
        keep = set(diag.loc[diag["ts_auc_abs"] >= min_ts_auc, "feature"])
        chosen |= {i for i, nm in enumerate(feature_names) if nm in keep}
    if not any_selector:
        chosen = set(range(len(feature_names)))
    idx = np.array(sorted(chosen), dtype=np.int64)
    return idx, [feature_names[i] for i in idx]


class TSRanker:
    """Groups rows by `t` once, so a TS-AUC over that row set costs two bincounts.

    Within a t-slice, AUC = (sum of positive ranks - P(P+1)/2) / (P*N). The within-t
    ranks of a feature do not depend on the labels at all, so the permutation null can
    rank each feature ONCE and then re-score every permutation from the same ranks --
    which is what turns an O(n_perm * n_features * n log n) loop into O(n_features *
    n log n) plus O(n_perm * n_features * n).

    Build one per row set and share it: `run_feature_diagnostic` hands the same
    instance to `_folded_ts_auc_table` and `permutation_null_ts_auc`."""

    def __init__(self, t):
        t = np.asarray(t)
        self.order_t = np.argsort(t, kind="mergesort")
        ts = t[self.order_t]
        _, self.gid_sorted = np.unique(ts, return_inverse=True)
        self.gid = np.empty_like(self.gid_sorted)
        self.gid[self.order_t] = self.gid_sorted          # group id per original row
        self.n_groups = int(self.gid_sorted.max()) + 1 if ts.size else 0

    def ranks(self, score):
        """Average within-t ranks (1-based, ties averaged) for every row, original order."""
        s = np.asarray(score, dtype=np.float64)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        o = np.lexsort((s, self.gid))                     # sort by group, then score
        g_o, s_o = self.gid[o], s[o]
        n = s.size
        new_run = np.empty(n, dtype=bool)
        new_run[0] = True
        new_run[1:] = (g_o[1:] != g_o[:-1]) | (s_o[1:] != s_o[:-1])
        new_grp = np.empty(n, dtype=bool)
        new_grp[0] = True
        new_grp[1:] = g_o[1:] != g_o[:-1]
        pos = np.arange(n, dtype=np.float64)
        grp_start = np.maximum.accumulate(np.where(new_grp, pos, 0.0))
        run_id = np.cumsum(new_run) - 1
        run_start = pos[new_run]
        run_len = np.diff(np.append(run_start, n))
        avg = (run_start - grp_start[new_run]) + (run_len + 1) / 2.0
        r = np.empty(n, dtype=np.float64)
        r[o] = avg[run_id]
        return r

    def auc_from_ranks(self, y, r):
        """Row-pair-weighted mean of the per-t AUCs, from precomputed ranks."""
        y = np.asarray(y).astype(np.float64)
        G = self.n_groups
        npos = np.bincount(self.gid, weights=y, minlength=G)
        ntot = np.bincount(self.gid, minlength=G).astype(np.float64)
        nneg = ntot - npos
        rsum = np.bincount(self.gid, weights=r * y, minlength=G)
        ok = (npos > 0) & (nneg > 0)
        if not ok.any():
            return 0.5
        w = npos[ok] * nneg[ok]
        auc = (rsum[ok] - npos[ok] * (npos[ok] + 1) / 2.0) / w
        return float((w * auc).sum() / w.sum())


def ts_auc(y, t, score):
    """The leaderboard metric: AUC within each t-slice, weighted by that slice's
    positive-negative pair count. Same value as the sklearn roc_auc_score loop it
    replaces, without the per-slice Python call."""
    rk = TSRanker(t)
    return rk.auc_from_ranks(y, rk.ranks(score))


def _folded_ts_auc_table(X, y, t, feature_names, ranker=None):
    rk = ranker if ranker is not None else TSRanker(t)
    rows = []
    for k, nm in enumerate(feature_names):
        a = rk.auc_from_ranks(y, rk.ranks(X[:, k]))
        rows.append((nm, max(a, 1 - a), a))
    return pd.DataFrame(rows, columns=["feature", "ts_auc_abs", "ts_auc_raw"])


def permutation_null_ts_auc(X, y, t, g, feature_idx, feature_names, n_perm=100, seed=0,
                             ranker=None):
    """Series-level label permutation: shuffle which SERIES are treated as
    breaking, keep each series' own row/t structure intact. A relabeled
    "breaker" reuses its own original first-break t as a synthetic tau (so the
    permuted series still has a well-defined post-break region); a real
    breaker relabeled non-breaking has all its rows set to y=0. Never shuffle
    rows independently -- that destroys the within-series correlation that is
    exactly why the naive (row-count) null is too tight.

    Same permutation scheme and the same RNG stream as the row-masking version it
    replaces: one rng.permutation(breaks_arr) per iteration, on an array of the same
    length. The per-series `g == sid` mask loop is gone -- the branch is now a
    vectorized np.where over rows, and the feature ranks are computed once up front
    because they do not depend on the labels."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); t = np.asarray(t); g = np.asarray(g)
    uniq_series, inv = np.unique(g, return_inverse=True)
    S = uniq_series.size
    breaks_arr = np.zeros(S, dtype=np.int64)
    np.maximum.at(breaks_arr, inv, y.astype(np.int64))

    # per-series synthetic tau = tt[len(tt) // 2] in the series' own row order
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=S)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    cut = t[order[starts + counts // 2]]
    synth_y = (t >= cut[inv]).astype(y.dtype)
    orig_break_row = breaks_arr[inv] == 1

    rk = ranker if ranker is not None else TSRanker(t)
    R = [rk.ranks(X[:, j]) for j in feature_idx]
    null = [[] for _ in feature_idx]
    for _ in range(n_perm):
        pb_row = rng.permutation(breaks_arr)[inv] == 1
        y_perm = np.where(pb_row, np.where(orig_break_row, y, synth_y), 0).astype(y.dtype)
        for k, r in enumerate(R):
            a = rk.auc_from_ranks(y_perm, r)
            null[k].append(max(a, 1 - a))

    rows = [(nm, float(np.mean(v)), float(np.percentile(v, 95)))
            for nm, v in zip(feature_names, null)]
    return pd.DataFrame(rows, columns=["feature", "null_mean", "null_p95"])


def run_feature_diagnostic(train_data, seed=0, n_series=10_000, n_perm=100,
                            perm_features=None, cache_dir=CACHE_DIR):
    """Sample n_series uniformly at random, extract the full registry (through
    the section 2b cache), fold TS-AUC per feature, and attach an honest
    permutation-null cutoff. Returns (results_df, family_rollup_df, cutoff,
    (X, y, t, w, g, feature_names)) -- the last element is the raw cached matrix
    for the sampled series, handed back so downstream cells (e.g. the per-feature
    AUC-by-age plot at the bottom of the notebook) don't need a second extraction."""
    items = list(train_data)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n_series, len(items)), replace=False)
    sample = [items[i] for i in idx]

    X, y, t, w, g, feature_names = get_cached_matrix(sample, subsample=True, seed=seed,
                                                       cache_dir=cache_dir)
    # A column that is constant across the whole matrix has TS-AUC exactly 0.5 and
    # tells you nothing -- but it looks identical to "real feature, no signal" in the
    # results table. `iqr` scored exactly 0.500000 in an earlier run for this reason.
    # Flag it here so a degenerate extraction can never be misread as a dead family.
    _sd = X.std(axis=0)
    _const = [nm for nm, s in zip(feature_names, _sd) if not (s > 0)]
    if _const:
        print("WARNING: %d/%d column(s) are CONSTANT across the matrix (std == 0) -- "
              "their TS-AUC of 0.5 means 'degenerate', not 'no signal':" % (len(_const), len(feature_names)))
        for nm in _const:
            print("    ", nm)

    # one grouping of the rows by t, reused by the table and the null
    ranker = TSRanker(t)
    diag = _folded_ts_auc_table(X, y, t, feature_names, ranker=ranker)
    diag["n_series"] = len(sample)
    diag["n_rows"] = X.shape[0]
    diag["family"] = diag["feature"].map(feature_to_family())

    # permutation null on a representative subset (best feature per family)
    # unless the caller asks for a specific list -- running it on all ~1500
    # features is possible but slow; the family-representative set is enough
    # to set CUTOFF.
    if perm_features is None:
        rep = diag.sort_values("ts_auc_abs", ascending=False).drop_duplicates(subset="family")
        perm_features = rep["feature"].tolist()
    perm_idx = [feature_names.index(nm) for nm in perm_features]
    null_df = permutation_null_ts_auc(X, y, t, g, perm_idx, perm_features,
                                       n_perm=n_perm, seed=seed, ranker=ranker)

    cutoff = float(null_df["null_p95"].max())
    diag = diag.merge(null_df[["feature", "null_mean", "null_p95"]], on="feature", how="left")
    diag["above_null"] = diag["ts_auc_abs"] >= cutoff
    diag = diag.sort_values("ts_auc_abs", ascending=False).reset_index(drop=True)

    rollup = (diag.groupby("family")["ts_auc_abs"]
              .agg(["max", "mean"]).sort_values("max", ascending=False).reset_index())

    print("permutation-null CUTOFF (P95 over %d representative features, %d perms): %.5f" % (len(perm_features), n_perm, cutoff))
    return diag, rollup, cutoff, (X, y, t, w, g, feature_names)


import time


def bench_extractor(datasets, n_series=3, n_project=2500, n_workers=4):
    """Per-family extraction cost for the current driver, and what it projects to.

    `n_project` / `n_workers` are what build_dataset will actually face -- the
    projection is just mean-seconds-per-series * n_project / n_workers, so it ignores
    worker start-up and the tail effect of uneven series lengths. Treat it as a floor.
    """
    items = list(datasets)[:n_series]
    base = {f.name for f in REGISTRY if f.cost == "O1"}

    def t_of(rec, active):
        xh, xo = rec[1], rec[2]
        fx = StreamFeatures(xh, active_families=active)
        t0 = time.perf_counter()
        for v in xo:
            fx.update(v)
        return time.perf_counter() - t0

    mean_len = float(np.mean([len(r[2]) for r in items]))
    b = float(np.mean([t_of(r, base) for r in items]))
    full = float(np.mean([t_of(r, None) for r in items]))

    print("%d series, mean online length %.0f steps" % (len(items), mean_len))
    print("  O(1) baseline (%s) : %7.4f s/series" % (",".join(sorted(base)), b))
    print("  FULL registry (%d families, %d features): %7.4f s/series" % (len(REGISTRY), N_FEATURES, full))
    print()
    print("  projected for %d series on %d workers: %5.1f min  (baseline only: %.1f min)" % (n_project, n_workers, full * n_project / n_workers / 60,
             b * n_project / n_workers / 60))
    for n in (10_000, 25_000):
        print("  projected for %6d series on %d workers: %5.1f min" % (n, n_workers, full * n / n_workers / 60))

    print("\n  marginal cost per family (on top of the O(1) baseline):")
    rows = sorted(((float(np.mean([t_of(r, base | {f.name}) for r in items])) - b,
                    f.name, f.cost) for f in REGISTRY if f.cost != "O1"), reverse=True)
    tot = sum(d for d, _, _ in rows)
    for d, n, c in rows:
        print("    %-10s %-7s %7.4f s  (%4.1f%% of the non-baseline cost)" % (n, c, d, 100.0 * d / tot if tot > 0 else 0.0))
    return full


#bench_extractor(train_data)


# Sample size / seed for the diagnostic AND for research_step (section 3d), so a
# research step scores its candidate on exactly the series the diagnostic reported on.
#DIAG_SEED = 0
#DIAG_N_SERIES = 2500


#diag, family_rollup, CUTOFF, diag_cache = run_feature_diagnostic(
#    train_data, seed=DIAG_SEED, n_series=DIAG_N_SERIES, n_perm=100,
#)
#Xd, yd, td, wd, gd, FEATURES_D = diag_cache   # full-registry matrix for the sampled series --
                                               # reusable by quick_train/ablate (section 3c) and
                                               # by plot_feature_auc_by_d (bottom of notebook)
#print("CUTOFF (permutation-null P95):", CUTOFF)
#diag.head(30)


# Where does the signal live, by feature family? (returned by run_feature_diagnostic,
# shown here for convenience) -- families whose max ts_auc_abs clears CUTOFF are the
# ones worth ablating in section 3c; below it, they're indistinguishable from noise
# at this sample size.
#family_rollup["clears_cutoff"] = family_rollup["max"] >= CUTOFF
#family_rollup.head(80)


def _fit_and_score(Xtr, ytr, wtr, feature_names, Xva, yva, tva, num_rounds=None):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feature_names, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, feature_name=feature_names, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=num_rounds or NUM_ROUNDS,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    pva = booster.predict(Xva, num_iteration=booster.best_iteration)
    return ts_auc(yva, tva, pva)


def cv_evaluate(datasets=None, n_folds: int = 5, seed: int = 0,
                 max_series=None, num_rounds=None, verbose: bool = True,
                 X=None, y=None, t=None, g=None, feature_idx=None, feature_names=None):
    """K-fold cross-validation over series.

    Two modes:
    - `datasets=` (original behavior): re-extracts each fold from raw series via
      build_dataset (train subsampled, valid at full resolution -- the fidelity-
      first path, what train() itself uses).
    - `X=, y=, t=, g=` (fast-iteration mode, section 3c): works directly off an
      already-extracted matrix (e.g. from get_cached_matrix), optionally
      restricted to `feature_idx` columns. NOTE: both "train" and "valid" rows
      here share whatever subsample schedule the cache was built with (unlike
      the datasets= path, which validates at full resolution) -- this trades a
      little fidelity for speed, which is the point of this path. Confirm a
      real decision with the datasets= path or crunch_tools.test() before
      committing to it.
    """
    if X is not None:
        assert y is not None and t is not None and g is not None, \
            "matrix mode requires X, y, t, g"
        names = list(feature_names) if feature_names is not None else list(FEATURES)
        if feature_idx is not None:
            Xsel = X[:, feature_idx]
            names = [names[i] for i in feature_idx]
        else:
            Xsel = X

        uniq = np.unique(g)
        if max_series:
            uniq = uniq[:max_series]
            mask = np.isin(g, uniq)
            Xsel, y, t, g = Xsel[mask], y[mask], t[mask], g[mask]
            uniq = np.unique(g)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(uniq))
        folds = np.array_split(perm, n_folds)

        fold_aucs = []
        for k in range(n_folds):
            va_ids = set(uniq[folds[k]].tolist())
            va_mask = np.isin(g, list(va_ids))
            tr_mask = ~va_mask
            auc = _fit_and_score(Xsel[tr_mask], y[tr_mask], np.ones(tr_mask.sum(), dtype=np.float32),
                                  names, Xsel[va_mask], y[va_mask], t[va_mask], num_rounds)
            fold_aucs.append(auc)
            if verbose:
                print("fold %d/%d | train %5d rows | valid %5d rows | TS-AUC %.5f" % (k + 1, n_folds, int(tr_mask.sum()), int(va_mask.sum()), auc))
    else:
        items = list(datasets)
        if max_series:
            items = items[:max_series]

        n = len(items)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        folds = np.array_split(perm, n_folds)

        fold_aucs = []
        for k in range(n_folds):
            va_idx = set(folds[k].tolist())
            tr_items = [items[i] for i in range(n) if i not in va_idx]
            va_items = [items[i] for i in folds[k]]

            Xtr, ytr, ttr, wtr, _ = build_dataset(tr_items, subsample=True, verbose=0)
            Xva, yva, tva, _, _ = build_dataset(va_items, subsample=False, verbose=0)

            auc = _fit_and_score(Xtr, ytr, wtr, FEATURES, Xva, yva, tva, num_rounds)
            fold_aucs.append(auc)
            if verbose:
                print("fold %d/%d | train %4d series | valid %4d series | TS-AUC %.5f" % (k + 1, n_folds, len(tr_items), len(va_items), auc))

    fold_aucs = np.array(fold_aucs)
    if verbose:
        print("\nCV TS-AUC: %.5f +/- %.5f  (min %.5f, max %.5f)" % (fold_aucs.mean(), fold_aucs.std(), fold_aucs.min(), fold_aucs.max()))
    return fold_aucs


def quick_train(cache, selection, max_series=None, num_rounds=None, seed=0, valid_fraction=None):
    """Thin wrapper over the same fit logic as train(), on an already-cached
    matrix + column selection. No re-extraction, no disk writes, no artifact --
    just a hold-out TS-AUC. `selection` is either an index array or a
    (idx, names) tuple, e.g. straight from select_features(...)."""
    X, y, t, w, g, feature_names = cache
    idx = selection[0] if isinstance(selection, tuple) else np.asarray(selection)
    names = [feature_names[i] for i in idx]
    Xsel = X[:, idx]

    uniq = np.unique(g)
    if max_series:
        uniq = uniq[:max_series]
        mask = np.isin(g, uniq)
        Xsel, y, t, w, g = Xsel[mask], y[mask], t[mask], w[mask], g[mask]
        uniq = np.unique(g)

    vf = valid_fraction if valid_fraction is not None else VALID_FRACTION
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(uniq) * vf)))
    val_ids = set(rng.choice(uniq, size=n_val, replace=False).tolist())
    va_mask = np.isin(g, list(val_ids))
    tr_mask = ~va_mask

    return _fit_and_score(Xsel[tr_mask], y[tr_mask], w[tr_mask], names,
                           Xsel[va_mask], y[va_mask], t[va_mask], num_rounds)


def ablate(cache, baseline_idx, candidates="families", n_folds=5, seed=0,
           num_rounds=None, max_series=None, verbose=True):
    """For each candidate family, try dropping it (if in the baseline selection)
    or adding it (if not), re-run cv_evaluate on the cached matrix, and report
    the delta vs. the baseline CV mean. This is the tool that makes keep/drop
    decisions -- it sees complementarity that the univariate diagnostic (section
    3) cannot. `baseline_idx` is a column-index array, e.g. from select_features."""
    X, y, t, w, g, feature_names = cache
    f2f = feature_to_family()
    name_to_idx = {nm: i for i, nm in enumerate(feature_names)}
    baseline_idx = np.asarray(baseline_idx)
    baseline_names = {feature_names[i] for i in baseline_idx}
    baseline_families = {f2f[nm] for nm in baseline_names}

    if candidates == "families":
        candidate_families = sorted({f.name for f in REGISTRY} - {"time"})   # t_step/t_log always kept
    else:
        candidate_families = list(candidates)

    base_fold = cv_evaluate(X=X, y=y, t=t, g=g, feature_idx=baseline_idx, feature_names=feature_names,
                             n_folds=n_folds, seed=seed, num_rounds=num_rounds, max_series=max_series,
                             verbose=False)
    base_mean, base_std = float(base_fold.mean()), float(base_fold.std())
    if verbose:
        print("baseline CV TS-AUC: %.5f +/- %.5f (%d features)\n" % (base_mean, base_std, len(baseline_idx)))

    rows = [("__baseline__", base_mean, base_std, 0.0, len(baseline_idx))]
    for fam in candidate_families:
        in_baseline = fam in baseline_families
        if in_baseline:
            mod_names = {nm for nm in baseline_names if f2f[nm] != fam}
            action = "drop"
        else:
            fam_names = {nm for nm in feature_names if f2f[nm] == fam}
            mod_names = baseline_names | fam_names
            action = "add"
        if not mod_names:
            continue
        mod_idx = np.array(sorted(name_to_idx[nm] for nm in mod_names), dtype=np.int64)
        fold_aucs = cv_evaluate(X=X, y=y, t=t, g=g, feature_idx=mod_idx, feature_names=feature_names,
                                 n_folds=n_folds, seed=seed, num_rounds=num_rounds, max_series=max_series,
                                 verbose=False)
        mean_, std_ = float(fold_aucs.mean()), float(fold_aucs.std())
        delta = mean_ - base_mean
        label = "%s %s" % (action, fam)
        rows.append((label, mean_, std_, delta, len(mod_idx)))
        if verbose:
            print("%-16s | CV %.5f +/- %.5f | delta %+.5f | n_features %d" % (label, mean_, std_, delta, len(mod_idx)))

    return pd.DataFrame(rows, columns=["change", "cv_mean", "cv_std", "delta_mean", "n_features"])


# Example usage of the full loop, using the diagnostic's cached matrix (Xd/yd/td/wd/gd,
# section 3) so nothing here re-extracts:
#
#   baseline_idx, baseline_names = select_features(
#       FEATURES_D, min_ts_auc=CUTOFF, diag=diag)   # everything that cleared the null,
#                                                     # plus t_step/t_log/h_* need adding back
#                                                     # explicitly since they're exempt from
#                                                     # the cutoff rule (see section 3 markdown)
#   baseline_idx, baseline_names = select_features(
#       FEATURES_D, families=set(baseline_names_families) | {"time", "h"},
#       patterns=None)   # (illustrative -- build the real baseline however fits your workflow)
#
#   quick_train((Xd, yd, td, wd, gd, FEATURES_D), (baseline_idx, baseline_names), max_series=2000)
#
#   ablate((Xd, yd, td, wd, gd, FEATURES_D), baseline_idx, candidates="families", n_folds=5)
#
# ablate()'s output table (change, cv_mean, cv_std, delta_mean, n_features) is what decides
# keep/drop -- per the accept-rule above, only act on a delta_mean that exceeds cv_std.
#print("quick_train / ablate defined -- see the commented example above for usage.")


# =====================================================================
#  3d. RESEARCH LOOP
# =====================================================================
import time

# Two named fit configurations. `overrides` are applied on top of LGB_PARAMS, which is
# defined in section 4 -- read at call time, never at import time, so this cell can sit
# above it.
# @crunch/keep:on
FAST_LGB = dict(
    overrides=dict(learning_rate=0.1, max_bin=63),
    num_boost_round=400,
    early_stopping=30,
)
FINAL_LGB = dict(
    overrides={},
    num_boost_round=None,        # None -> NUM_ROUNDS
    early_stopping=80,
)
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


def _lgb_config(fast_lgb=True):
    cfg = FAST_LGB if fast_lgb else FINAL_LGB
    params = dict(LGB_PARAMS)
    params.update(cfg["overrides"])
    return params, cfg["num_boost_round"] or NUM_ROUNDS, cfg["early_stopping"]


def _sample_for(datasets, n_series, seed):
    """The same seeded sample run_feature_diagnostic draws, so a research step scores
    its candidate on the series the diagnostic reported on."""
    items = list(datasets)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n_series, len(items)), replace=False)
    return [items[i] for i in idx]


def smoke_test_family(fam, datasets, n_series=3, verbose=True):
    """Cheap sanity checks before a candidate costs a full extraction.

    Catches the four ways a new family wastes an hour: duplicate names inside itself,
    a name that collides with an existing feature, a name/width disagreement, and a
    column that is constant (which scores exactly 0.500000 and looks like 'no signal')."""
    names = list(fam.names())
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise AssertionError("%s.names() has duplicates: %s" % (fam.name, dupes))

    existing_families = {f.name for f in REGISTRY}
    if fam.name in existing_families:
        raise AssertionError(
            "family name %r is already in REGISTRY -- pick another .name, or drop the "
            "old one first with research_step(..., drop={%r})" % (fam.name, fam.name))
    clash = sorted(set(names) & set(feature_to_family()))
    if clash:
        raise AssertionError(
            "%s emits feature name(s) that already exist: %s" % (fam.name, clash))

    reg = list(REGISTRY) + [fam]
    rows = []
    for did, xh, xo, tau in list(datasets)[:n_series]:
        fx = StreamFeatures(xh, active_families={fam.name}, registry=reg)
        if fx.names != names:
            raise AssertionError(
                "%s: driver emits %d columns, names() promises %d" % (fam.name, len(fx.names), len(names)))
        rows.append(np.asarray([fx.update(v) for v in xo], dtype=np.float64))
    R = np.concatenate(rows)

    if R.shape[1] != len(names):
        raise AssertionError("%s.update() returned %d values, names() has %d" % (fam.name, R.shape[1], len(names)))
    bad = ~np.isfinite(R)
    if bad.any():
        cols = [names[c] for c in np.unique(np.nonzero(bad)[1])]
        raise AssertionError("%s produced non-finite values in: %s" % (fam.name, cols))
    const = [names[c] for c in range(R.shape[1]) if R[:, c].std() == 0]
    if const:
        raise AssertionError(
            "%s produced CONSTANT column(s) over %d smoke rows: %s -- a constant column "
            "scores exactly 0.5 and is indistinguishable from 'no signal' in the "
            "diagnostic, so fix it before spending an extraction on it" % (fam.name, R.shape[0], const))

    if verbose:
        print("  smoke test OK: %d column(s), %d rows over %d series, all finite, none constant" % (R.shape[1], R.shape[0], min(n_series, len(list(datasets)))))
    return R


def _fixed_folds(g, n_folds, seed):
    """Series-level folds that depend only on (series set, n_folds, seed) -- so the
    baseline and the candidate are always scored on identical splits."""
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    return [tuple(sorted(uniq[ix].tolist())) for ix in np.array_split(perm, n_folds)]


def _cv_fold_aucs(X, y, t, w, g, names, idx, folds, params, num_rounds, early_stop):
    Xsel = X[:, idx]
    sel = [names[i] for i in idx]
    out = []
    for va_ids in folds:
        va = np.isin(g, np.asarray(va_ids))
        tr = ~va
        dtr = lgb.Dataset(Xsel[tr], label=y[tr], weight=w[tr],
                          feature_name=sel, free_raw_data=False)
        dva = lgb.Dataset(Xsel[va], label=y[va], feature_name=sel,
                          reference=dtr, free_raw_data=False)
        booster = lgb.train(
            params, dtr, num_boost_round=num_rounds,
            valid_sets=[dva], valid_names=["valid"],
            callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)],
        )
        pva = booster.predict(Xsel[va], num_iteration=booster.best_iteration)
        out.append(ts_auc(y[va], t[va], pva))
    return np.asarray(out, dtype=np.float64)


def _cached_null(skey, X, y, t, g, names, fam_names, n_perm, seed, ranker, cache_dir,
                  verbose=True):
    """Permutation-null cutoff over one representative feature per BASELINE family.

    Keyed on the baseline family set (not the candidate's), so adding a family does not
    invalidate the cutoff and every later research_step is a dict lookup."""
    key = "null::" + _short_hash({"families": sorted(fam_names), "n_perm": n_perm,
                                   "seed": seed, "n_rows": int(len(y))})
    hit = load_null(skey, key, cache_dir=cache_dir)
    if hit is not None:
        if verbose:
            print("  null: cached (cutoff %.5f, %d perms)" % (hit["cutoff"], hit["n_perm"]))
        return hit["cutoff"], hit["null_df"]

    f2f = feature_to_family()
    base_cols = [i for i, nm in enumerate(names) if f2f.get(nm) in fam_names]
    tab = _folded_ts_auc_table(X[:, base_cols], y, t, [names[i] for i in base_cols],
                                ranker=ranker)
    tab["family"] = tab["feature"].map(f2f)
    rep = tab.sort_values("ts_auc_abs", ascending=False).drop_duplicates(subset="family")
    perm_features = rep["feature"].tolist()
    perm_idx = [names.index(nm) for nm in perm_features]
    if verbose:
        print("  null: computing over %d representative features x %d perms ..." % (len(perm_features), n_perm))
    null_df = permutation_null_ts_auc(X, y, t, g, perm_idx, perm_features,
                                       n_perm=n_perm, seed=seed, ranker=ranker)
    cutoff = float(null_df["null_p95"].max())
    save_null(skey, key, {"cutoff": cutoff, "null_df": null_df, "n_perm": n_perm},
              cache_dir=cache_dir)
    return cutoff, null_df


def research_step(new_family=None, drop=None, n_series=None, seed=0,
                   n_folds=3, fast_lgb=True, register=False,
                   datasets=None, n_perm=100, cache_dir=CACHE_DIR, verbose=True):
    """Score one idea against the current registry and print a verdict.

    new_family : a FeatureFamily INSTANCE to evaluate, or None to only measure `drop`.
    drop       : family names to remove from the baseline.
    register   : keep `new_family` in REGISTRY afterwards (default: restore the registry).

    Returns the univariate table for the candidate's columns; the CV numbers, the
    verdict and the stage timings are also on `df.attrs`.
    """
    datasets = train_data if datasets is None else datasets
    n_series = DIAG_N_SERIES if n_series is None else n_series
    drop = set(drop or ())
    sample = _sample_for(datasets, n_series, seed)

    registry_backup = list(REGISTRY)
    timings = {}
    try:
        # ---------------------------------------------------- 0. smoke test
        if new_family is not None:
            if verbose:
                print("== smoke test: %s (%d columns) ==" % (new_family.name,
                                                             len(new_family.names())))
            t0 = time.perf_counter()
            smoke_test_family(new_family, sample, n_series=3, verbose=verbose)
            timings["smoke"] = time.perf_counter() - t0
            REGISTRY.append(new_family)

        base_families = {f.name for f in REGISTRY} - drop
        if new_family is not None:
            base_families -= {new_family.name}
            cand_families = base_families | {new_family.name}
            label = "+%s" % new_family.name
        else:
            cand_families = base_families | drop
            label = "+%s" % ",".join(sorted(drop)) if drop else "(no change)"
        assert cand_families != base_families or not drop, \
            "nothing to compare: pass new_family= and/or drop="

        # ---------------------------------------------------- 1. extraction
        if verbose:
            print("\n== extraction (%d series, seed %d) ==" % (len(sample), seed))
        t0 = time.perf_counter()
        X, y, t, w, g, names = get_cached_matrix(sample, subsample=True, seed=seed,
                                                  cache_dir=cache_dir, verbose=verbose)
        timings["extract"] = time.perf_counter() - t0
        skey = sample_key([r[0] for r in sample], True, seed)
        f2f = feature_to_family()

        # ---------------------------------------------------- 2. univariate
        if verbose:
            print("\n== univariate ==")
        t0 = time.perf_counter()
        ranker = TSRanker(t)
        cutoff, _null_df = _cached_null(skey, X, y, t, g, names, base_families,
                                         n_perm, seed, ranker, cache_dir, verbose)
        if new_family is not None:
            cols = [i for i, nm in enumerate(names) if f2f.get(nm) == new_family.name]
            uni = _folded_ts_auc_table(X[:, cols], y, t, [names[i] for i in cols],
                                        ranker=ranker)
        else:
            uni = _folded_ts_auc_table(X[:, :0], y, t, [], ranker=ranker)
        uni["cutoff"] = cutoff
        uni["above_null"] = uni["ts_auc_abs"] >= cutoff
        uni = uni.sort_values("ts_auc_abs", ascending=False).reset_index(drop=True)
        timings["univariate"] = time.perf_counter() - t0
        if verbose and len(uni):
            print(uni.to_string(index=False))
            print("  %d/%d of the candidate's columns clear the null cutoff %.5f" % (int(uni["above_null"].sum()), len(uni), cutoff))

        # ---------------------------------------------------- 3. CV
        if verbose:
            print("\n== %d-fold CV (folds fixed by seed=%d, %s) ==" % (n_folds, seed, "FAST_LGB" if fast_lgb else "FINAL_LGB"))
        t0 = time.perf_counter()
        params, n_rounds, early = _lgb_config(fast_lgb)
        folds = _fixed_folds(g, n_folds, seed)

        base_idx = np.array([i for i, nm in enumerate(names)
                             if f2f.get(nm) in base_families], dtype=np.int64)
        cand_idx = np.array([i for i, nm in enumerate(names)
                             if f2f.get(nm) in cand_families], dtype=np.int64)

        fit_key = {"params": {k: params[k] for k in sorted(params)},
                   "num_rounds": n_rounds, "early": early,
                   "folds": [list(f) for f in folds]}
        base_key = "cv::" + _short_hash(dict(fit_key, features=[names[i] for i in base_idx]))
        base_fold = load_null(skey, base_key, cache_dir=cache_dir)
        if base_fold is None:
            base_fold = _cv_fold_aucs(X, y, t, w, g, names, base_idx, folds,
                                       params, n_rounds, early)
            save_null(skey, base_key, base_fold, cache_dir=cache_dir)
            base_src = "trained"
        else:
            base_src = "cached"
        cand_key = "cv::" + _short_hash(dict(fit_key, features=[names[i] for i in cand_idx]))
        cand_fold = load_null(skey, cand_key, cache_dir=cache_dir)
        if cand_fold is None:
            cand_fold = _cv_fold_aucs(X, y, t, w, g, names, cand_idx, folds,
                                       params, n_rounds, early)
            save_null(skey, cand_key, cand_fold, cache_dir=cache_dir)
            cand_src = "trained"
        else:
            cand_src = "cached"
        timings["cv"] = time.perf_counter() - t0

        deltas = cand_fold - base_fold
        mean_delta = float(deltas.mean())
        paired_std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        keep = mean_delta > paired_std

        if verbose:
            print("  baseline  (%2d families, %3d features, %s): %s | mean %.5f" % (len(base_families), len(base_idx), base_src,
                     " ".join("%.5f" % a for a in base_fold), base_fold.mean()))
            print("  candidate (%2d families, %3d features, %s): %s | mean %.5f" % (len(cand_families), len(cand_idx), cand_src,
                     " ".join("%.5f" % a for a in cand_fold), cand_fold.mean()))
            print("  per-fold delta: %s" % " ".join("%+.5f" % d for d in deltas))
            print("  mean delta %+.5f | paired std %.5f (ddof=1)" % (mean_delta, paired_std))
            print()
            print("  %-9s %s  mean delta %+.5f vs paired std %.5f   "
                  "[smoke %.1fs | extract %.1fs | univariate %.1fs | cv %.1fs]" % ("KEEP" if keep else "NO GAIN", label, mean_delta, paired_std,
                     timings.get("smoke", 0.0), timings["extract"],
                     timings["univariate"], timings["cv"]))
            if register and new_family is not None:
                print("  registered %r -- REGISTRY is now %d families" % (new_family.name, len(REGISTRY)))
            elif new_family is not None:
                print("  registry restored (pass register=True to keep %r)" % new_family.name)

        uni.attrs.update(
            verdict="KEEP" if keep else "NO GAIN",
            label=label,
            baseline_folds=base_fold, candidate_folds=cand_fold, deltas=deltas,
            mean_delta=mean_delta, paired_std=paired_std, cutoff=cutoff,
            timings=timings, folds=folds, sample_key=skey,
            baseline_families=sorted(base_families),
            candidate_families=sorted(cand_families),
        )
        return uni
    finally:
        if not (register and new_family is not None):
            REGISTRY[:] = registry_backup


def local_test(**kwargs):
    """The real end-to-end crunch run: train() + infer() + determinism check on the
    actual competition data. Minutes to hours -- this is the occasional final smoke
    test, not part of the iteration loop. Everything above exists so that this is
    rarely the thing you are waiting on."""
    return crunch_tools.test(**kwargs)


#print("research_step / local_test defined -- see the markdown above for the 4-line workflow.")


# @crunch/keep:on
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.04,
    num_leaves=63,
    min_data_in_leaf=300,
    feature_fraction=0.70,
    bagging_fraction=0.80,
    bagging_freq=1,
    lambda_l2=5.0,
    max_bin=127,
    num_threads=max(1, (os.cpu_count() or 4) - 1),
    verbosity=-1,
)
LGB_PARAMS.update(
    learning_rate=0.037212,
    num_leaves=81,
    min_data_in_leaf=1860,
    feature_fraction=0.871187,
    bagging_fraction=0.641272,
    lambda_l2=43.842790,
    extra_trees=True,
)
NUM_ROUNDS = 3000
VALID_FRACTION = 0.15
TRAIN_MAX_SERIES = None      # set an int for fast iteration
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


def active_feature_names(active_families=None) -> List[str]:
    """The feature name list StreamFeatures(..., active_families=active_families)
    actually emits, in order -- FEATURES itself only covers active_families=None
    (the full registry). train()/infer() must always pass their booster's
    feature_name= this, never the bare FEATURES global, whenever ACTIVE_FAMILIES
    is a strict subset."""
    return [nm for f in REGISTRY if active_families is None or f.name in active_families
            for nm in f.names()]


def train(datasets, model_directory_path: str):
    items = list(datasets)
    if TRAIN_MAX_SERIES:
        items = items[:TRAIN_MAX_SERIES]

    ids = np.array([r[0] for r in items])
    rng = np.random.default_rng(0)
    is_val = rng.random(len(items)) < VALID_FRACTION

    tr_items = [r for r, v in zip(items, is_val) if not v]
    va_items = [r for r, v in zip(items, is_val) if v]
    print("train series %d | valid series %d" % (len(tr_items), len(va_items)))

    # active_families left at the sentinel default -> resolves to the current
    # ACTIVE_FAMILIES global, so this is exactly what infer() will compute too.
    Xtr, ytr, ttr, wtr, _ = build_dataset(tr_items, subsample=True)
    Xva, yva, tva, _, gva = build_dataset(va_items, subsample=False)
    print("rows: train %s  valid %s" % (Xtr.shape, Xva.shape))

    feat_names = active_feature_names(ACTIVE_FAMILIES)
    assert Xtr.shape[1] == len(feat_names), (
        "extracted %d columns but active_feature_names() expects %d -- "
        "ACTIVE_FAMILIES changed between extraction and here?" % (Xtr.shape[1], len(feat_names)))

    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feat_names, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, feature_name=feat_names, reference=dtr, free_raw_data=False)

    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=NUM_ROUNDS,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)],
    )

    pva = booster.predict(Xva, num_iteration=booster.best_iteration)
    print("HOLD-OUT TS-AUC: %.5f" % ts_auc(yva, tva, pva))
    # Kaggle-only diagnostic. Crunch's cloud has no /kaggle, and this runs BEFORE
    # joblib.dump below -- writing it unconditionally kills train() before the model
    # is saved, and infer() then dies on a missing model.joblib.
    if os.path.isdir("/kaggle/working"):
        np.savez_compressed("/kaggle/working/holdout_scores.npz",
                            p=pva, y=yva, t=tva, g=gva)
        print("saved hold-out scores -> /kaggle/working/holdout_scores.npz")

    imp = pd.Series(booster.feature_importance("gain"), index=feat_names).sort_values(ascending=False)
    print(imp.head(25))

    # Model artifact is self-describing: it carries the exact active-family set
    # and feature list it was trained with, so infer() can reconstruct the same
    # extractor from the artifact itself rather than trusting the notebook's
    # ACTIVE_FAMILIES global at inference time (which could have drifted).
    joblib.dump(
        {
            "booster": booster,
            "features": feat_names,
            "active_families": sorted(ACTIVE_FAMILIES) if ACTIVE_FAMILIES else None,
            "best_iteration": booster.best_iteration,
        },
        os.path.join(model_directory_path, "model.joblib"),
    )


# @crunch/keep:on
SMOOTH_ALPHA = 0.0     # 0 = off; e.g. 0.3 = EWMA-smooth the score across t
RUNNING_MAX = False    # enforce monotone non-decreasing score
# @crunch/keep:off
#pass  # anchor: crunch-convert only sees keep:off on a simple statement


def _stride_for(t: int) -> int:
    if t < 64:
        return 1
    if t < 256:
        return 2
    return 4


def infer(datasets, model_directory_path: str):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    art = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    booster = art["booster"]
    best_it = art["best_iteration"]
    active_families = set(art["active_families"]) if art.get("active_families") else None
    n_feat = len(art["features"])
    booster.params["num_threads"] = 1

    buf = np.empty((1, n_feat), dtype=np.float64)

    yield  # signal readiness

    for x_historical, x_online in datasets:
        fx = StreamFeatures(x_historical, active_families=active_families)
        prev = 0.0
        smoothed = 0.0
        running = 0.0

        for t, point in enumerate(x_online):
            f = fx.update(point)

            if t % _stride_for(t) == 0:
                buf[0, :] = f
                prev = float(booster.predict(buf, num_iteration=best_it)[0])
            p = prev

            if SMOOTH_ALPHA > 0.0:
                smoothed = (1.0 - SMOOTH_ALPHA) * smoothed + SMOOTH_ALPHA * p if t else p
                p = smoothed
            if RUNNING_MAX:
                running = p if p > running else running
                p = running

            if p < 0.0:
                p = 0.0
            elif p > 1.0:
                p = 1.0
            yield float(p)


def check_train_infer_parity(model_directory_path, datasets, n_series=20):
    art = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    infer_active = set(art["active_families"]) if art.get("active_families") else None
    train_active = ACTIVE_FAMILIES
    assert infer_active == train_active, (
        "artifact's active_families (%s) != current ACTIVE_FAMILIES (%s) -- "
        "you're checking parity against the wrong global; re-set ACTIVE_FAMILIES "
        "to what this artifact was trained with." % (infer_active, train_active))
    assert art["features"] == active_feature_names(infer_active), (
        "artifact's stored feature list doesn't match active_feature_names(infer_active) -- "
        "REGISTRY must have changed since this model was trained.")

    items = list(datasets)[:n_series]
    max_abs_diff = 0.0
    for did, xh, xo, tau in items:
        fx_infer = StreamFeatures(xh, active_families=infer_active)     # infer()'s construction
        fx_train = StreamFeatures(xh, active_families=train_active)     # train()'s construction
        for point in xo:
            r_i = np.asarray(fx_infer.update(point), dtype=np.float64)
            r_t = np.asarray(fx_train.update(point), dtype=np.float64)
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(r_i - r_t))))
            assert np.allclose(r_i, r_t, atol=1e-9, rtol=1e-9), (
                "train/infer feature mismatch for series %s" % did)
    print("PARITY OK over %d series, max abs diff = %.3e" % (len(items), max_abs_diff))


# Run after train() (section 4) / crunch_tools.test() (section 6) has produced model.joblib.
# Fill in the actual model_directory_path crunch used (crunch_tools.test() prints/creates it,
# commonly "resources" or wherever `train()`'s model_directory_path argument pointed):
# check_train_infer_parity("resources", train_data, n_series=20)


import os
#os.makedirs("resources", exist_ok=True)
#train(train_data, "resources")


# @crunch/keep:on
INFER_PARALLELISM = 8
# @crunch/keep:off


#crunch_tools.test(
#    # force_first_train=False,
#    # no_determinism_check=True,
#)


#prediction = pd.read_parquet("prediction/prediction.parquet")
#y_test = pd.read_parquet("data/y_test.reduced.parquet")

#merged = prediction.merge(y_test, how="left", left_index=True, right_index=True)
#merged["time_online"] = merged.groupby("id").cumcount()

#print("Local TS-AUC: %.5f" % ts_auc(
#    merged["target"].values, merged["time_online"].values, merged["prediction"].values
#))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# TEST 1: Per-step AUC Curve (AUC(t) vs t)
def safe_auc(g):
    if g['target'].nunique() > 1:
        return roc_auc_score(g['target'], g['prediction'])
    return np.nan

#auc_t = merged.groupby('time_online').apply(safe_auc)

#plt.figure(figsize=(10, 5))
#plt.plot(auc_t.index, auc_t.values, label='Cross-sectional AUC', color='blue', alpha=0.7)
#plt.axhline(0.5, color='red', linestyle='--', label='Random (0.5)')
#plt.title('Test 1: Per-Step AUC(t)')
#plt.xlabel('Time Step (t)')
#plt.ylabel('ROC AUC')
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.show()


# TEST 2: AUC Stratified by Elapsed Time Since Break (d)
# 1. Find tau for each series (first step where target == 1)
#tau_df = merged[merged['target'] == 1].groupby('id')['time_online'].min().reset_index().rename(columns={'time_online': 'tau'})
#df_d = merged.merge(tau_df, on='id', how='left')

# 2. Calculate age of break 'd'
#df_d['d'] = df_d['time_online'] - df_d['tau']

# 3. Vectorized Exact Individual AUC calculation
#df_d['rank_all'] = df_d.groupby('time_online')['prediction'].rank()
#pos_mask = df_d['target'] == 1
#df_d.loc[pos_mask, 'rank_pos'] = df_d[pos_mask].groupby('time_online')['prediction'].rank()

#neg_counts = df_d[~pos_mask].groupby('time_online').size()
#df_d['n_neg'] = df_d['time_online'].map(neg_counts)

#df_d.loc[pos_mask, 'indiv_auc'] = (df_d.loc[pos_mask, 'rank_all'] - df_d.loc[pos_mask, 'rank_pos']) / df_d.loc[pos_mask, 'n_neg']

#auc_d = df_d[pos_mask].groupby('d')['indiv_auc'].mean()
#auc_d_smooth = auc_d.rolling(window=10, min_periods=1, center=True).mean()

#plt.figure(figsize=(10, 5))
#plt.plot(auc_d.index, auc_d.values, color='gray', alpha=0.4, label='Raw AUC(d)')
#plt.plot(auc_d_smooth.index, auc_d_smooth.values, color='green', linewidth=2.5, label='Smoothed AUC(d)')
#plt.axhline(0.5, color='red', linestyle='--')
#plt.title('Test 2: AUC Stratified by Age of Break (d = t - τ)')
#plt.xlabel('Elapsed Time Since Break (d)')
#plt.ylabel('ROC AUC')
#plt.legend()
#plt.grid(True, alpha=0.3)
#plt.show()


# TEST 3: Cross-sectional Std of Predictions at Fixed t
#std_t = merged.groupby('time_online')['prediction'].std()

#plt.figure(figsize=(10, 5))
#plt.plot(std_t.index, std_t.values, color='purple')
#plt.title('Test 3: Cross-Sectional Prediction Std')
#plt.xlabel('Time Step (t)')
#plt.ylabel('Standard Deviation of Scores')
#plt.grid(True, alpha=0.3)
#plt.show()


#"""
#Causal sup-KS scan statistic — corrected diagnostic.
#
#Fully CAUSAL (uses only data up to t, never tau), so its TS-AUC is a real
#competing baseline, not a ceiling. If it beats the LightGBM bank on the SAME
#held-out series, the bank has a problem.
#"""

import glob
import numpy as np

#MIN_W = 12          # smallest window length considered
#MAX_W = 512         # cap: longer windows cost more, add little power
#T_STRIDE = 1        # evaluate every T_STRIDE steps (safe: depends only on t)

#_ARANGE = {}


def _idx(m):
    """Cached np.arange(1, m+1) — window lengths only take ~7 distinct values."""
    a = _ARANGE.get(m)
    if a is None:
        a = _ARANGE[m] = np.arange(1, m + 1, dtype=np.float64)
    return a


def _window_lengths(m_max):
    """Geometric ladder of trailing-window lengths, plus the full history so far."""
    ms = []
    m = MIN_W
    while m <= min(m_max, MAX_W):
        ms.append(m)
        m *= 2
    if m_max <= MAX_W and (not ms or ms[-1] != m_max):
        ms.append(m_max)
    if not ms:
        ms = [m_max]
    return ms


def ks_scan_scores(x_hist, x_online):
    """One causal sup-KS score per online step."""
    xh = np.asarray(x_hist, dtype=np.float64).ravel()
    xo = np.asarray(x_online, dtype=np.float64).ravel()
    n = xh.size
    T = xo.size

    # PIT the whole online segment once: u ~ Uniform(0,1) under "no break".
    xs = np.sort(xh)
    u = np.searchsorted(xs, xo, side="right") / float(n)

    out = np.zeros(T, dtype=np.float64)
    last = 0.0

    for t in range(T):
        if t % T_STRIDE != 0:
            out[t] = last
            continue

        m_max = t + 1
        best = 0.0
        for m in _window_lengths(m_max):
            w = np.sort(u[t + 1 - m: t + 1])
            i = _idx(m)
            d_plus = np.max(i / m - w)
            d_minus = np.max(w - (i - 1.0) / m)
            d = d_plus if d_plus > d_minus else d_minus
            en = np.sqrt(n * m / float(n + m))     # two-sample scaling
            v = en * d
            if v > best:
                best = v

        last = best
        out[t] = best

    return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_scan(items, ts_auc, n_series=500, verbose=True):
    ys, ts, ps, ids = [], [], [], []

    for i, (did, xh, xo, tau) in enumerate(list(items)[:n_series]):
        s = ks_scan_scores(xh, xo)
        T = len(s)
        ys.extend(0 if (tau is None or k < tau) else 1 for k in range(T))
        ts.extend(range(T))
        ps.extend(s.tolist())
        ids.extend([did] * T)
        if verbose and (i + 1) % 100 == 0:
            print("  %d/%d series" % (i + 1, n_series))

    auc = ts_auc(np.asarray(ys), np.asarray(ts), np.asarray(ps))
    print("Causal sup-KS scan TS-AUC: %.5f  (%d series)" % (auc, min(n_series, len(list(items)))))
    return auc, np.asarray(ids), np.asarray(ys), np.asarray(ts), np.asarray(ps)


def evaluate_model_same_subset(items, booster, best_it, extract_series,
                               ts_auc, n_series=500):
    ys, ts, ps = [], [], []
    for did, xh, xo, tau in list(items)[:n_series]:
        X, y, tt, _ = extract_series(xh, xo, tau, subsample=False)
        p = booster.predict(X, num_iteration=best_it)
        ys.append(y); ts.append(tt); ps.append(p)
    ys = np.concatenate(ys); ts = np.concatenate(ts); ps = np.concatenate(ps)
    auc = ts_auc(ys, ts, ps)
    print("LightGBM TS-AUC on same subset: %.5f" % auc)
    return auc


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
# 1. Recover the trained booster.  It was a LOCAL inside train(); crunch_tools.test()
#    called train(), which dumped it to disk and returned nothing to this namespace.
#paths = glob.glob("**/model.joblib", recursive=True)
#print("found artefacts:", paths)
#art = joblib.load(paths[0])
#booster, best_it = art["booster"], art["best_iteration"]

# 2. Reproduce train()'s held-out split EXACTLY (same seed, same VALID_FRACTION).
#    Scoring the model on train_data[:500] would be in-sample; the scan statistic
#    has no fitted parameters, so that comparison would be meaningless.
#items = list(train_data)
#rng = np.random.default_rng(0)
#is_val = rng.random(len(items)) < VALID_FRACTION
#va_items = [r for r, v in zip(items, is_val) if v][:500]
#print("held-out series:", len(va_items))

# 3. Same series, both detectors.
#scan_auc, scan_ids, scan_y, scan_t, scan_p = evaluate_scan(
#    va_items, ts_auc, n_series=len(va_items))
#model_auc = evaluate_model_same_subset(
#    va_items, booster, best_it, extract_series, ts_auc, n_series=len(va_items))

#print()
#print("sup-KS scan : %.5f" % scan_auc)
#print("LightGBM    : %.5f" % model_auc)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_auc_by_d(X, y, t, g, features_to_plot, all_feature_names):
    # 1. Reconstruct a DataFrame with the necessary columns
    df = pd.DataFrame({'id': g, 't': t, 'target': y})

    # Map the requested features from the X matrix
    for f in features_to_plot:
        if f in all_feature_names:
            idx = all_feature_names.index(f)
            df[f] = X[:, idx]
        else:
            print(f"Warning: Feature '{f}' not found in the feature list.")
            return

    # 2. Find tau for each series
    tau_df = df[df['target'] == 1].groupby('id')['t'].min().reset_index().rename(columns={'t': 'tau'})
    df_d = df.merge(tau_df, on='id', how='left')

    # 3. Calculate age of break 'd'
    df_d['d'] = df_d['t'] - df_d['tau']
    pos_mask = df_d['target'] == 1

    # Count negatives at each t
    neg_counts = df_d[~pos_mask].groupby('t').size()
    df_d['n_neg'] = df_d['t'].map(neg_counts)

    plt.figure(figsize=(12, 6))

    # 4. Calculate and plot AUC(d) for each feature
    for f in features_to_plot:
        # Fractional rank calculation
        df_d['rank_all'] = df_d.groupby('t')[f].rank()
        df_d.loc[pos_mask, 'rank_pos'] = df_d[pos_mask].groupby('t')[f].rank()

        # Individual AUC
        df_d.loc[pos_mask, 'indiv_auc'] = (df_d.loc[pos_mask, 'rank_all'] - df_d.loc[pos_mask, 'rank_pos']) / df_d.loc[pos_mask, 'n_neg']

        # Group by age 'd'
        auc_d = df_d[pos_mask].groupby('d')['indiv_auc'].mean()

        # If the feature is negatively correlated (AUC < 0.5), flip it so signal strength is comparable
        if auc_d.mean() < 0.5:
            auc_d = 1.0 - auc_d

        # Smooth the curve for readability
        auc_d_smooth = auc_d.rolling(window=10, min_periods=1, center=True).mean()

        plt.plot(auc_d_smooth.index, auc_d_smooth.values, linewidth=2.5, label=f)

    plt.axhline(0.5, color='red', linestyle='--', label='Random (0.5)')
    plt.title('Feature Signal Strength Stratified by Age of Break (d)')
    plt.xlabel('Elapsed Time Since Break (d = t - τ)')
    plt.ylabel('ROC AUC (Smoothed & Direction-Adjusted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Xd/yd/td/gd/FEATURES_D come from the diagnostic run (section 3) -- a full-registry
# matrix over DIAG_N_SERIES sampled series, so any feature name in FEATURES_D works here.
# The old z_* location bank is gone (Phase 1 cull), so these pick one feature from each
# of the three surviving signal axes: the EWMA scale bank, a windowed variance test, and
# an L-moment shape statistic.
#features_to_test = ['e2_ew0.02', 'ftest_e_w256', 'lmom_lkurt_e_w32']
#plot_feature_auc_by_d(Xd, yd, td, gd, features_to_test, FEATURES_D)


import os, time
import numpy as np

#P2_N_SERIES = 3000
#P2_SEED = 123
#P2_PATH = "/kaggle/working/p2_cache.npz"

#if os.path.exists(P2_PATH):
#    Z = np.load(P2_PATH)
#    X_all, y_all, t_all, g_all = Z["X"], Z["y"], Z["t"], Z["g"]
#    feat_names_all = [str(s) for s in Z["names"]]
#    print("loaded cache:", X_all.shape)
#else:
#    items = list(train_data)
#    pick = np.random.default_rng(P2_SEED).choice(len(items), size=P2_N_SERIES, replace=False)
#    sample = [items[i] for i in sorted(pick)]
#    t0 = time.time()
#    # full resolution; the training subsample is re-created in memory in the next cell
#    X_all, y_all, t_all, _, g_all = build_dataset(sample, subsample=False)
#    feat_names_all = active_feature_names(ACTIVE_FAMILIES)
#    assert X_all.shape[1] == len(feat_names_all)
#    np.savez(P2_PATH, X=X_all, y=y_all, t=t_all, g=g_all, names=np.array(feat_names_all))
#    print(f"extracted {X_all.shape} in {(time.time()-t0)/60:.1f} min -> {P2_PATH}")


import time
import numpy as np, pandas as pd, lightgbm as lgb

#P2_FOLDS = 3

def _stride_arr(t):
    s = np.full(t.shape, KEEP_EVERY[-1][1], dtype=np.int32)
    for up_to, stride in reversed(KEEP_EVERY):
        s[t < up_to] = stride
    return s

def _sorted_by_t(mask):
    idx = np.flatnonzero(mask)
    return idx[np.argsort(t_all[idx], kind="stable")]

#uniq = np.unique(g_all)
#fold_u = np.empty(uniq.size, dtype=np.int32)
#for k, part in enumerate(np.array_split(np.random.default_rng(0).permutation(uniq.size), P2_FOLDS)):
#    fold_u[part] = k
#fold_row = fold_u[np.searchsorted(uniq, g_all)]
#stride_all = _stride_arr(t_all)
#sub_all = (t_all % stride_all) == 0           # same rows extract_series(subsample=True) keeps

# each validation fold is split in two by series: "stop" rows pick the number of trees,
# "score" rows are only used for the reported TS-AUC -- so no variant is graded on
# the same rows it early-stopped on (that would flatter the TS-AUC-stopped variants)
#half_u = np.random.default_rng(1).random(uniq.size) < 0.5
#half_row = half_u[np.searchsorted(uniq, g_all)]

#FOLD_DATA = []
#for k in range(P2_FOLDS):
#    tr = _sorted_by_t((fold_row != k) & sub_all)          # train: subsampled, like train()
#    st = _sorted_by_t((fold_row == k) & half_row)         # early-stopping half, full resolution
#    sc = _sorted_by_t((fold_row == k) & ~half_row)        # scoring half, full resolution
#    rk = TSRanker(t_all[st])
#    yst = y_all[st]
#    FOLD_DATA.append(dict(
#        Xtr=X_all[tr], ytr=y_all[tr], wtr=stride_all[tr].astype(np.float32),
#        gtr=np.unique(t_all[tr], return_counts=True)[1],
#        Xst=X_all[st], yst=yst, tst=t_all[st], gst=np.unique(t_all[st], return_counts=True)[1],
#        Xsc=X_all[sc], ysc=y_all[sc], tsc=t_all[sc],
#        feval=(lambda rk_, y_: (lambda preds, data: ("ts_auc", rk_.auc_from_ranks(y_, rk_.ranks(preds)), True)))(rk, yst),
#    ))
#    print(f"fold {k}: train rows {tr.size:,} | stop rows {st.size:,} | score rows {sc.size:,}")

#RESULTS = {}

def run_variant(name, objective="binary", stop_on="ts_auc", extra=None, num_rounds=3000, patience=100):
    aucs, iters, secs = [], [], []
    for k, F in enumerate(FOLD_DATA):
        params = dict(LGB_PARAMS)
        params.update(extra or {})
        params["objective"] = objective
        params["metric"] = "binary_logloss" if stop_on == "logloss" else "None"
        if objective == "binary":
            dtr = lgb.Dataset(F["Xtr"], label=F["ytr"], weight=F["wtr"], feature_name=feat_names_all, free_raw_data=False)
            dva = lgb.Dataset(F["Xst"], label=F["yst"], reference=dtr, free_raw_data=False)
        else:   # ranking: one query group per t, so the model learns the within-t ordering TS-AUC scores
            dtr = lgb.Dataset(F["Xtr"], label=F["ytr"], group=F["gtr"], feature_name=feat_names_all, free_raw_data=False)
            dva = lgb.Dataset(F["Xst"], label=F["yst"], group=F["gst"], reference=dtr, free_raw_data=False)
        t0 = time.time()
        booster = lgb.train(
            params, dtr, num_boost_round=num_rounds, valid_sets=[dva], valid_names=["valid"],
            feval=None if stop_on == "logloss" else F["feval"],
            callbacks=[lgb.early_stopping(patience, verbose=False)],
        )
        psc = booster.predict(F["Xsc"], num_iteration=booster.best_iteration)
        aucs.append(ts_auc(F["ysc"], F["tsc"], psc))
        iters.append(booster.best_iteration)
        secs.append(time.time() - t0)
        print(f"  [{name}] fold {k}: TS-AUC {aucs[-1]:.5f} | best_iter {iters[-1]} | {secs[-1]:.0f}s")
    RESULTS[name] = dict(aucs=np.array(aucs), iters=iters, secs=secs)
    print(f"[{name}] mean TS-AUC {np.mean(aucs):.5f}")

def summarize(baseline):
    b = RESULTS[baseline]["aucs"]
    rows = []
    for name, r in RESULTS.items():
        d = r["aucs"] - b
        rows.append(dict(variant=name, mean_ts_auc=r["aucs"].mean(), delta_vs_base=d.mean(),
                         paired_std=d.std(ddof=1) if len(d) > 1 else 0.0,
                         folds_better=int((d > 0).sum()), mean_best_iter=int(np.mean(r["iters"])),
                         min_per_fold=np.mean(r["secs"]) / 60))
    out = pd.DataFrame(rows).sort_values("mean_ts_auc", ascending=False).reset_index(drop=True)
    return out.round(5)

#pd.set_option("display.width", 200); #pd.set_option("display.max_columns", 20)


#print(len(FOLD_DATA), "folds |", FOLD_DATA[0]["Xtr"].shape[1], "columns |", "tst" in FOLD_DATA[0])


# P2-3: baseline, identical to what train() does today (~10-15 min)
#BASE = "A binary / stop logloss"
#run_variant(BASE, stop_on="logloss")


# P2-4: same model, but stop on TS-AUC instead of logloss (~10-20 min)
#run_variant("B binary / stop TS-AUC")


# P2-5: ranking objective, rows grouped by t (~10-20 min)
#run_variant("C rank_xendcg / stop TS-AUC", objective="rank_xendcg")


#summarize(BASE)


#run_variant("D lambdarank / stop TS-AUC", objective="lambdarank")
#summarize(BASE)


import time, os
#NEW_FAMS = {"glr", "bocpd"}
#NEW_PATH = "/kaggle/working/p2_newfams_10k.npz"

#if os.path.exists(NEW_PATH):
#    Z = np.load(NEW_PATH); Xn = Z["X"]; new_names = [str(s) for s in Z["names"]]
#else:
#    t0 = time.time()
#    Xn, yn, tn, _, gn = build_dataset(list(train_data), subsample=False, active_families=NEW_FAMS)
#    assert np.array_equal(yn, y_all) and np.array_equal(tn, t_all) and np.array_equal(gn, g_all), \
#        "row order differs from the 10k cache"
#    new_names = active_feature_names(NEW_FAMS)
#    np.savez(NEW_PATH, X=Xn, names=np.array(new_names))
#    print(f"extracted {Xn.shape} in {(time.time()-t0)/60:.1f} min")
#print(new_names)


#tab = _folded_ts_auc_table(Xn, y_all, t_all, new_names)
#ref = X_all[:, feat_names_all.index("e2_ew0.02")].astype(np.float64)
#tab["corr_with_e2_ew0.02"] = [abs(np.corrcoef(Xn[:, i].astype(np.float64), ref)[0, 1]) for i in range(len(new_names))]
#print("reference: e2_ew0.02 =", round(ts_auc(y_all, t_all, ref), 5))
#tab.sort_values("ts_auc_abs", ascending=False).round(4)


#X_all = np.hstack([X_all, Xn]); #del Xn
#feat_names_all = list(feat_names_all) + new_names
import gc; #gc.collect()
#print(X_all.shape)


import gc, os, time
#for _v in ("X_all", "y_all", "t_all", "g_all", "FOLD_DATA"):
#    globals().pop(_v, None)
#gc.collect()

#P2_PATH = "/kaggle/working/p2_cache_10k.npz"
#if os.path.exists(P2_PATH):
#    Z = np.load(P2_PATH)
#    X_all, y_all, t_all, g_all = Z["X"], Z["y"], Z["t"], Z["g"]
#    feat_names_all = [str(s) for s in Z["names"]]
#    print("loaded cache:", X_all.shape)
#else:
#    t0 = time.time()
#    X_all, y_all, t_all, _, g_all = build_dataset(list(train_data), subsample=False)
#    feat_names_all = active_feature_names(ACTIVE_FAMILIES)
#    np.savez(P2_PATH, X=X_all, y=y_all, t=t_all, g=g_all, names=np.array(feat_names_all))
#    print(f"extracted {X_all.shape} in {(time.time()-t0)/60:.1f} min")


#CMP = {}
def cv_cols(label, cols):
    names = [feat_names_all[i] for i in cols]
    aucs = []
    for k, F in enumerate(FOLD_DATA):
        for swap in (False, True):
            Xs, ys = (F["Xsc"], F["ysc"]) if swap else (F["Xst"], F["yst"])
            Xe, ye, te = (F["Xst"], F["yst"], F["tst"]) if swap else (F["Xsc"], F["ysc"], F["tsc"])
            params = dict(LGB_PARAMS); params.update(objective="binary", metric="binary_logloss")
            dtr = lgb.Dataset(F["Xtr"][:, cols], label=F["ytr"], weight=F["wtr"], feature_name=names, free_raw_data=False)
            dva = lgb.Dataset(Xs[:, cols], label=ys, reference=dtr, free_raw_data=False)
            t0 = time.time()
            bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            aucs.append(ts_auc(ye, te, bst.predict(Xe[:, cols], num_iteration=bst.best_iteration)))
            print(f"  [{label}] fold {k} swap={swap}: {aucs[-1]:.5f} | iters {bst.best_iteration} | {time.time()-t0:.0f}s")
    CMP[label] = np.array(aucs)
    print(f"[{label}] mean {CMP[label].mean():.5f}")

def cmp_table(base="base"):
    b = CMP[base]; rows = []
    for lab, a in CMP.items():
        d = a - b
        rows.append(dict(config=lab, mean=a.mean(), delta=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)) if lab != base else 0.0,
                         better=f"{int((d > 0).sum())}/{len(d)}"))
    return pd.DataFrame(rows).sort_values("mean", ascending=False).round(5)

#idx = {n: i for i, n in enumerate(feat_names_all)}
#BASE_COLS = [i for i, n in enumerate(feat_names_all) if n not in new_names]
#GLR_COLS = [idx[n] for n in new_names if n.startswith("glr_")]
#BOC_COLS = [idx[n] for n in new_names if n.startswith("bocpd_")]


#cv_cols("base", BASE_COLS)


#cv_cols("+glr+bocpd", BASE_COLS + GLR_COLS + BOC_COLS)


#cmp_table()


#cv_cols("+glr", BASE_COLS + GLR_COLS)


#cv_cols("+bocpd", BASE_COLS + BOC_COLS)
#cmp_table()


#d = CMP["+glr+bocpd"] - CMP["+bocpd"]
#print("per fit:", np.round(d, 5))
#print(f"glr on top of bocpd: {d.mean():+.5f} | se {d.std(ddof=1)/np.sqrt(len(d)):.5f} | better {int((d>0).sum())}/6")


#REGISTRY[:] = [f for f in REGISTRY if f.name != "glr"]
#FEAT_COLS = BASE_COLS + BOC_COLS
#print([f.name for f in REGISTRY][-2:], "| model columns:", len(FEAT_COLS))


import optuna, time
#optuna.logging.set_verbosity(optuna.logging.WARNING)
#FEAT_NAMES = [feat_names_all[i] for i in FEAT_COLS]

def fit_fold(F, extra, cols=None, swap=False, num_rounds=3000, patience=100):
    # cols resolved in the body, not as a default: crunch-convert comments out the
    # research-only FEAT_COLS, and a default arg would be evaluated at import time.
    cols = FEAT_COLS if cols is None else cols
    Xs, ys = (F["Xsc"], F["ysc"]) if swap else (F["Xst"], F["yst"])
    Xe, ye, te = (F["Xst"], F["yst"], F["tst"]) if swap else (F["Xsc"], F["ysc"], F["tsc"])
    params = dict(LGB_PARAMS); params.update(extra)
    params.update(objective="binary", metric="binary_logloss")
    names = [feat_names_all[i] for i in cols]
    dtr = lgb.Dataset(F["Xtr"][:, cols], label=F["ytr"], weight=F["wtr"], feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(Xs[:, cols], label=ys, reference=dtr, free_raw_data=False)
    bst = lgb.train(params, dtr, num_boost_round=num_rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(patience, verbose=False)])
    return ts_auc(ye, te, bst.predict(Xe[:, cols], num_iteration=bst.best_iteration)), bst.best_iteration

def objective(trial):
    extra = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 100, 5000, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l2=trial.suggest_float("lambda_l2", 0.01, 100.0, log=True),
        extra_trees=trial.suggest_categorical("extra_trees", [False, True]),
    )
    aucs, iters, t0 = [], [], time.time()
    for k, F in enumerate(FOLD_DATA):
        auc, it = fit_fold(F, extra)
        aucs.append(auc); iters.append(it)
        trial.report(float(np.mean(aucs)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("fold_aucs", [round(a, 5) for a in aucs])
    trial.set_user_attr("mean_best_iter", int(np.mean(iters)))
    print(f"trial {trial.number}: {np.mean(aucs):.5f} | iters {int(np.mean(iters))} | "
          f"{(time.time()-t0)/60:.1f} min | {extra}")
    return float(np.mean(aucs))

#study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
#                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
#study.enqueue_trial(dict(learning_rate=LGB_PARAMS["learning_rate"], num_leaves=LGB_PARAMS["num_leaves"],
#                         min_data_in_leaf=LGB_PARAMS["min_data_in_leaf"],
#                         feature_fraction=LGB_PARAMS["feature_fraction"],
#                         bagging_fraction=LGB_PARAMS["bagging_fraction"],
#                         lambda_l2=LGB_PARAMS["lambda_l2"], extra_trees=False))


#study.optimize(objective, n_trials=40, timeout=3 * 3600)


#df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
#df = df[df.state == "COMPLETE"].sort_values("value", ascending=False)
#df["delta_vs_trial0"] = df.value - df.loc[df.number == 0, "value"].item()
#df.head(8)


import numpy as np, pandas as pd

#CAND = {
#    "trial0 (current)": dict(learning_rate=LGB_PARAMS["learning_rate"], num_leaves=LGB_PARAMS["num_leaves"],
#                             min_data_in_leaf=LGB_PARAMS["min_data_in_leaf"],
#                             feature_fraction=LGB_PARAMS["feature_fraction"],
#                             bagging_fraction=LGB_PARAMS["bagging_fraction"],
#                             lambda_l2=LGB_PARAMS["lambda_l2"], extra_trees=False),
#    "trial31": dict(study.trials[31].params),
#    "trial27": dict(study.trials[27].params),
#}

#CONF = {}
#for lab, extra in CAND.items():
#    aucs, iters = [], []
#    for k, F in enumerate(FOLD_DATA):
#        for swap in (False, True):
#            a, it = fit_fold(F, extra, swap=swap)
#            aucs.append(a); iters.append(it)
#    CONF[lab] = (np.array(aucs), np.mean(iters))
#    print(f"{lab}: mean {np.mean(aucs):.5f} | mean trees {np.mean(iters):.0f}")

#b = CONF["trial0 (current)"][0]
#rows = []
#for lab, (a, it) in CONF.items():
#    d = a - b
#    rows.append(dict(config=lab, mean=a.mean(), delta=d.mean(),
#                     se=d.std(ddof=1) / np.sqrt(len(d)) if lab != "trial0 (current)" else 0.0,
#                     better=f"{int((d > 0).sum())}/{len(d)}", trees=int(it)))
#pd.DataFrame(rows).sort_values("mean", ascending=False).round(5)
