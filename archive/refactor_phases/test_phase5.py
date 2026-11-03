"""PHASE 5 test -- research_step end to end on synthetic data with a toy family.

The toy family is a window MAD ratio: median absolute deviation of the trailing window
against the reference's, log-ratio, scaled by sqrt(m). It is a real (if redundant)
scale statistic, so it should behave sensibly rather than trivially.
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nb_exec import run_notebook, make_synthetic

SCRATCH = os.environ.get("PHASE5_CACHE", "/tmp/phase5_cache_test")

TOY_SRC = '''
class MADRatioFamily(TwoSampleFamily):
    """Toy candidate: log ratio of window MAD to reference MAD, scaled by sqrt(m)."""
    name = "madratio"
    version = 1

    def init(self, hist):
        super().init(hist)
        self.ref_mad = {}
        for ch in ("z", "e"):
            r = self.ref[ch]
            self.ref_mad[ch] = float(np.median(np.abs(r - np.median(r))))

    def stat(self, w, r):
        m = w.size
        if m < 8:
            return 0.0
        mad_w = float(np.median(np.abs(w - _sorted_median(w))))
        mad_r = self.ref_mad[self._cur_ch]
        if mad_w <= 0 or mad_r <= 0:
            return 0.0
        return math.log(mad_w / mad_r) * math.sqrt(m)

    def update(self, ctx):
        out = []
        for ch in self.channels:
            self._cur_ch = ch
            for W in WINDOWS:
                out.append(_safe(self.stat(ctx.sorted_window(ch, W), self.ref[ch])))
        return out
'''


BAD_FAMILIES_SRC = '''
class Collide(TwoSampleFamily):
    name = "ftest"                      # already registered
    def stat(self, w, r):
        return 1.0


class ConstFam(TwoSampleFamily):
    name = "constfam"                   # every column constant
    def stat(self, w, r):
        return 1.0


class NaNFam(TwoSampleFamily):
    """Returns NaN WITHOUT going through _safe() -- a family that forgot the guard.
    (A family that does call _safe() turns its NaN into 0.0 and gets caught by the
    constant-column check instead, which is also a real failure.)"""
    name = "nanfam"

    def update(self, ctx):
        return [float("nan") if W > 32 else float(ctx.t)
                for ch in self.channels for W in WINDOWS]


class DupNames(TwoSampleFamily):
    name = "dupfam"

    def names(self):
        return ["dupfam_a", "dupfam_a", "dupfam_b"]

    def stat(self, w, r):
        return 1.0
'''


def main():
    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH)

    print("=" * 74)
    print("PHASE 5 TEST -- research_step with a toy family")
    print("=" * 74)

    data = make_synthetic(200, seed=33)
    ns = run_notebook(synthetic_train_data=data)
    exec(compile(TOY_SRC, "<toy>", "exec"), ns)
    MADRatioFamily = ns["MADRatioFamily"]

    n_before = len(ns["REGISTRY"])
    print("\nregistry before: %d families, %d features" % (n_before, ns["N_FEATURES"]))

    # ---------------------------------------------------------------- call 1 (cold)
    print("\n" + "-" * 74)
    print("CALL 1 -- cold caches")
    print("-" * 74)
    t0 = time.perf_counter()
    df1 = ns["research_step"](MADRatioFamily(), n_series=200, seed=0, n_folds=3,
                              fast_lgb=True, cache_dir=SCRATCH, n_perm=60)
    dt1 = time.perf_counter() - t0
    print("\nCALL 1 total wall clock: %.1fs" % dt1)
    assert len(ns["REGISTRY"]) == n_before, "registry not restored"
    print("registry restored: %d families" % len(ns["REGISTRY"]))

    # ---------------------------------------------------------------- call 2 (warm)
    print("\n" + "-" * 74)
    print("CALL 2 -- same arguments, everything should hit a cache")
    print("-" * 74)
    t0 = time.perf_counter()
    df2 = ns["research_step"](MADRatioFamily(), n_series=200, seed=0, n_folds=3,
                              fast_lgb=True, cache_dir=SCRATCH, n_perm=60)
    dt2 = time.perf_counter() - t0
    print("\nCALL 2 total wall clock: %.1fs   (call 1 was %.1fs -> %.1fx faster)"
          % (dt2, dt1, dt1 / dt2))

    a1, a2 = df1.attrs, df2.attrs
    assert np.allclose(a1["baseline_folds"], a2["baseline_folds"])
    assert np.allclose(a1["candidate_folds"], a2["candidate_folds"])
    assert a1["folds"] == a2["folds"], "folds are not fixed across calls"
    assert a1["verdict"] == a2["verdict"]
    assert np.allclose(df1["ts_auc_abs"].to_numpy(), df2["ts_auc_abs"].to_numpy())
    assert dt2 < dt1 * 0.5, "second call did not benefit from the caches"
    print("identical verdict/folds/AUCs across the two calls: ok")

    print("\n-- returned table (call 1) --")
    print(df1.to_string(index=False))
    print("\n-- attrs --")
    for k in ("verdict", "label", "mean_delta", "paired_std", "cutoff", "sample_key"):
        print("   %-14s %s" % (k, a1[k]))
    print("   %-14s %s" % ("baseline_folds", np.round(a1["baseline_folds"], 5)))
    print("   %-14s %s" % ("candidate_folds", np.round(a1["candidate_folds"], 5)))
    print("   %-14s %s" % ("deltas", np.round(a1["deltas"], 5)))
    print("   %-14s call1 %s" % ("timings", {k: round(v, 2) for k, v in a1["timings"].items()}))
    print("   %-14s call2 %s" % ("", {k: round(v, 2) for k, v in a2["timings"].items()}))

    # ---------------------------------------------------------------- register=True
    print("\n" + "-" * 74)
    print("CALL 3 -- register=True keeps the family")
    print("-" * 74)
    ns["research_step"](MADRatioFamily(), n_series=200, seed=0, n_folds=3,
                        fast_lgb=True, cache_dir=SCRATCH, n_perm=60, register=True)
    assert len(ns["REGISTRY"]) == n_before + 1, "register=True did not keep the family"
    assert "madratio" in {f.name for f in ns["REGISTRY"]}
    print("\nregistry after register=True: %d families" % len(ns["REGISTRY"]))
    ns["REGISTRY"].pop()

    # ---------------------------------------------------------------- smoke failures
    print("\n" + "-" * 74)
    print("SMOKE TEST -- fails fast on a bad candidate")
    print("-" * 74)

    # defined inside the notebook namespace so they see WINDOWS / _safe like a real one
    exec(compile(BAD_FAMILIES_SRC, "<bad>", "exec"), ns)

    for fam, why in ((ns["Collide"](), "name already in REGISTRY"),
                     (ns["ConstFam"](), "constant column"),
                     (ns["NaNFam"](), "non-finite"),
                     (ns["DupNames"](), "duplicate names")):
        try:
            ns["research_step"](fam, n_series=200, seed=0, cache_dir=SCRATCH, verbose=False)
            raise SystemExit("FAIL: %s was not rejected" % fam.name)
        except AssertionError as exc:
            print("   %-10s rejected (%s):" % (fam.name, why))
            print("      %s" % str(exc)[:150])
    assert len(ns["REGISTRY"]) == n_before, "registry leaked after a failed smoke test"
    print("\n   registry intact after 4 rejected candidates: %d families" % len(ns["REGISTRY"]))

    # ---------------------------------------------------------------- drop=
    print("\n" + "-" * 74)
    print("CALL 4 -- drop= only (what is a family worth?)")
    print("-" * 74)
    df4 = ns["research_step"](None, drop={"mmd"}, n_series=200, seed=0, n_folds=3,
                              fast_lgb=True, cache_dir=SCRATCH, n_perm=60)
    print("\n   verdict %s | mean delta %+.5f" % (df4.attrs["verdict"], df4.attrs["mean_delta"]))

    print("\n" + "=" * 74)
    print("PHASE 5: PASS")
    print("=" * 74)


if __name__ == "__main__":
    main()
