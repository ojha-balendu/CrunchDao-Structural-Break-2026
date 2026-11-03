"""PHASE 2 test -- exact equivalence of the fast families vs. tools/reference_impl.py,
then a per-family benchmark.

    python tools/test_phase2.py [n_series] [n_bench]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nb_exec import run_notebook, make_synthetic
from reference_impl import build_reference

ATOL = RTOL = 1e-8
BASELINE = {"e2", "z2", "tail", "time"}

# Which class each family MUST actually be running, and a marker proving it.
#   ("base", X)  -> X is in the class's MRO
#   ("attr", a)  -> the initialised instance carries attribute `a`
#   ("calls", f) -> some method of the class references the global name `f`
#
# This exists because equivalence alone cannot detect a failed port: a family left as
# the original matches the reference at exactly 0.000e+00. That is precisely what went
# wrong the first time -- bartlett, levene, fligner, mood, sigtuk, energy and mmd were
# silently still the originals and the equivalence test passed.
FAST_MARKERS = {
    "ftest":    ("attr", "v2"),
    "bartlett": ("attr", "v2"),
    "levene":   ("attr", "rc"),
    "fligner":  ("attr", "ar"),
    "lmom":     ("calls", "_l_moments_fast"),
    "iqr":      ("calls", "_sorted_percentile"),
    "ansari":   ("base", "_RankScaleFamily"),
    "mood":     ("base", "_RankScaleFamily"),
    "sigtuk":   ("base", "RefCachedTwoSample"),
    "wass":     ("attr", "_qcache"),
    "energy":   ("attr", "pre"),
    "mmd":      ("attr", "krr"),
    "entropy":  ("calls", "_vasicek_fast"),
}


def _referenced_names(cls):
    """Every global name any method of `cls` (or its non-base ancestors) references."""
    out = set()
    for klass in cls.__mro__:
        if klass.__name__ in ("FeatureFamily", "object"):
            continue
        for v in vars(klass).values():
            code = getattr(v, "__code__", None)
            if code is not None:
                out.update(code.co_names)
                for const in code.co_consts:
                    if hasattr(const, "co_names"):
                        out.update(const.co_names)
    return out


def check_fast_installed(ns, hist):
    """Assert every family is running the FAST implementation, not the original."""
    print("-- which implementation is actually registered --")
    bad = []
    for f in ns["REGISTRY"]:
        if f.name in BASELINE:
            print("    %-10s %-24s (O(1) baseline, not ported)"
                  % (f.name, type(f).__name__))
            continue
        kind, marker = FAST_MARKERS[f.name]
        cls = type(f)
        if kind == "base":
            ok = marker in [c.__name__ for c in cls.__mro__]
        elif kind == "attr":
            probe = f.clone()
            probe.init(hist)
            ok = hasattr(probe, marker)
        else:
            ok = marker in _referenced_names(cls)
        print("    %-10s %-24s %-7s %s=%s"
              % (f.name, cls.__name__, "FAST" if ok else "ORIGINAL", kind, marker))
        if not ok:
            bad.append(f.name)
    if bad:
        raise AssertionError(
            "these families are still the ORIGINAL implementation: %s" % sorted(bad))
    print("  [ok] all 13 non-baseline families are the fast implementation\n")


def check_rank_rewrite(ns, n=200, seed=5):
    """_avg_ranks_of() must equal _avg_ranks() over the concatenation, ties included."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for trial in range(40):
        # round hard so there are ties both within each sample and across them
        w = np.sort(np.round(rng.normal(size=rng.integers(8, 60)), 1))
        r = np.sort(np.round(rng.normal(size=rng.integers(8, 90)), 1))
        want = ns["_avg_ranks"](np.concatenate([w, r]))[:w.size]
        got = ns["_avg_ranks_of"](w, w, r)
        worst = max(worst, float(np.max(np.abs(want - got))))
    print("-- _avg_ranks_of vs _avg_ranks over 40 tied samples: max abs diff %.3e" % worst)
    assert worst == 0.0, "the searchsorted rank rewrite is not exact"
    print("  [ok] exact\n")


def check_fast_equivalence(ns, ref, datasets, active_families=None, atol=ATOL, rtol=RTOL):
    """Drive the notebook's StreamFeatures and the frozen reference over the same
    series, step for step, and diff every feature at every step."""
    worst = 0.0
    worst_where = None
    for did, xh, xo, tau in datasets:
        fast = ns["StreamFeatures"](xh, active_families=active_families)
        slow = ref["StreamFeatures"](xh, active_families=active_families)
        assert fast.names == slow.names, "name/order mismatch on %s" % did
        for step, v in enumerate(xo):
            ra = np.asarray(slow.update(v), dtype=np.float64)
            rb = np.asarray(fast.update(v), dtype=np.float64)
            d = np.abs(ra - rb)
            k = int(np.argmax(d))
            if d[k] > worst:
                worst, worst_where = float(d[k]), (did, step, fast.names[k], ra[k], rb[k])
            if not np.allclose(ra, rb, atol=atol, rtol=rtol):
                raise AssertionError(
                    "series %s step %d feature %s: reference=%.12g fast=%.12g (diff %.3e)"
                    % (did, step, fast.names[k], ra[k], rb[k], d[k]))
    return worst, worst_where


def main():
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    n_bench = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print("=" * 74)
    print("PHASE 2 TEST -- fast families vs. frozen reference")
    print("=" * 74)

    ns = run_notebook(synthetic_train_data=make_synthetic(1, seed=0))
    ref = build_reference(ns)
    fams = [f.name for f in ns["REGISTRY"]]
    assert len(fams) == 17 and ns["N_FEATURES"] == 82
    assert ref["FEATURES"] == ns["FEATURES"], "reference feature order drifted"

    # tie_every=3 -> a third of the series are rounded to 2dp, which is the path that
    # separates tie-averaged ranks (ansari/mood/fligner) and stable-sort positions
    # (sigtuk) from a naive argsort.
    check_fast_installed(ns, ns["HistoricalRef"](make_synthetic(1, seed=2)[0][1]))
    check_rank_rewrite(ns)

    data = make_synthetic(n_series, seed=11, tie_every=3)
    n_tied = sum(1 for i in range(n_series) if i % 3 == 0)
    print("\n%d synthetic series (%d heavily tied, %d with a break), all 17 families active"
          % (n_series, n_tied, sum(1 for d in data if d[3] is not None)))
    print("tolerance: atol=rtol=%g\n" % ATOL)

    t0 = time.perf_counter()
    worst, where = check_fast_equivalence(ns, ref, data, active_families=None)
    dt = time.perf_counter() - t0
    n_rows = sum(len(d[2]) for d in data)
    print("EQUIVALENCE OK over %d series / %d steps / %d feature values in %.1fs"
          % (n_series, n_rows, n_rows * 82, dt))
    print("max abs diff = %.3e" % worst)
    if where:
        print("  (worst at series %s step %d, feature %s: ref=%.12g fast=%.12g)" % where)
    assert worst < ATOL or np.isclose(worst, 0), "max abs diff %.3e exceeds atol" % worst

    # ---------------------------------------------------------------- per-family
    print("\n-- per-family equivalence (each family alone, on the tied series) --")
    tied = [d for i, d in enumerate(data) if i % 3 == 0][:4]
    for fam in fams:
        w, _ = check_fast_equivalence(ns, ref, tied, active_families=BASELINE | {fam})
        print("    %-10s max abs diff %.3e" % (fam, w))

    # ---------------------------------------------------------------- benchmark
    print("\n" + "=" * 74)
    print("BENCHMARK  (%d series, mean seconds/series)" % n_bench)
    print("=" * 74)
    bench = make_synthetic(n_bench, seed=99, tie_every=3)

    def timeit(builder, active):
        out = []
        for did, xh, xo, tau in bench:
            fx = builder(xh, active_families=active)
            t0 = time.perf_counter()
            for v in xo:
                fx.update(v)
            out.append(time.perf_counter() - t0)
        return float(np.mean(out))

    slow_b = timeit(ref["StreamFeatures"], BASELINE)
    fast_b = timeit(ns["StreamFeatures"], BASELINE)
    slow_f = timeit(ref["StreamFeatures"], None)
    fast_f = timeit(ns["StreamFeatures"], None)

    print("\n  %-24s %10s %10s %8s" % ("", "reference", "fast", "speedup"))
    print("  %-24s %10.4f %10.4f %7.2fx"
          % ("baseline (e2,z2,tail,time)", slow_b, fast_b, slow_b / fast_b))
    print("  %-24s %10.4f %10.4f %7.2fx"
          % ("FULL registry", slow_f, fast_f, slow_f / fast_f))

    print("\n  marginal cost of each family (on top of the O(1) baseline):")
    print("  %-10s %10s %10s %8s" % ("family", "reference", "fast", "speedup"))
    rows = []
    for fam in fams:
        if fam in BASELINE:
            continue
        s = timeit(ref["StreamFeatures"], BASELINE | {fam}) - slow_b
        f = timeit(ns["StreamFeatures"], BASELINE | {fam}) - fast_b
        rows.append((s, fam, f))
    for s, fam, f in sorted(rows, reverse=True):
        sp = ("%7.2fx" % (s / f)) if f > 1e-6 else "      -"
        print("  %-10s %10.4f %10.4f %s" % (fam, s, f, sp))

    mean_len = np.mean([len(d[2]) for d in bench])
    print("\n  (mean online length %.0f steps)" % mean_len)
    for label, sec in (("reference", slow_f), ("fast", fast_f)):
        print("  %-10s 2500 series / 4 workers: %6.1f min   10k series: %6.1f min"
              % (label, sec * 2500 / 4 / 60, sec * 10000 / 4 / 60))

    print("\n" + "=" * 74)
    print("PHASE 2: PASS")
    print("=" * 74)


if __name__ == "__main__":
    main()
