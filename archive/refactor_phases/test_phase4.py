"""PHASE 4 test -- per-family column cache.

Extract 50 synthetic series with every family, delete one family's file, re-request,
and check that only that family is re-extracted and the assembled matrix is identical.
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nb_exec import run_notebook, make_synthetic

SCRATCH = os.environ.get("PHASE4_CACHE", "/tmp/phase4_cache_test")


def main():
    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH)

    print("=" * 74)
    print("PHASE 4 TEST -- per-family column cache")
    print("=" * 74)

    data = make_synthetic(50, seed=21)
    ns = run_notebook(synthetic_train_data=data)
    gcm = ns["get_cached_matrix"]
    fams = [f.name for f in ns["REGISTRY"]]

    # ------------------------------------------------- 1. cold: everything extracted
    print("\n-- 1. cold cache: all 17 families --")
    t0 = time.perf_counter()
    X1, y1, t1, w1, g1, n1 = gcm(data, cache_dir=SCRATCH, seed=0)
    dt_cold = time.perf_counter() - t0
    print("   -> X %s, %d feature names, %.1fs" % (X1.shape, len(n1), dt_cold))
    assert X1.shape[1] == 82 and n1 == ns["FEATURES"]

    skey = ns["sample_key"]([r[0] for r in data], True, 0)
    d = os.path.join(SCRATCH, skey)
    files = sorted(os.listdir(d))
    print("   files on disk:", len(files))
    for f in files[:4]:
        print("      ", f)
    print("       ... (%d family files + meta.joblib)" % (len(files) - 1))
    assert "meta.joblib" in files
    assert sum(1 for f in files if f.startswith("fam__")) == 17

    # ------------------------------------------------- 2. warm: nothing extracted
    print("\n-- 2. warm cache: nothing should be extracted --")
    t0 = time.perf_counter()
    X2, y2, t2, w2, g2, n2 = gcm(data, cache_dir=SCRATCH, seed=0)
    dt_warm = time.perf_counter() - t0
    print("   -> %.3fs  (cold was %.1fs, %.0fx faster)" % (dt_warm, dt_cold, dt_cold / dt_warm))
    assert np.array_equal(X1, X2) and n1 == n2
    assert np.array_equal(y1, y2) and np.array_equal(t1, t2)
    assert np.array_equal(w1, w2) and np.array_equal(g1, g2)
    print("   [ok] assembled matrix bit-identical")

    # ------------------------------------- 3. delete ONE family -> only it re-extracts
    victim = "mmd"
    print("\n-- 3. delete family %r, re-request --" % victim)
    ns["clear_cache"](family=victim, cache_dir=SCRATCH)
    remaining = sum(1 for f in os.listdir(d) if f.startswith("fam__"))
    print("   family files left: %d" % remaining)
    assert remaining == 16

    t0 = time.perf_counter()
    X3, y3, t3, w3, g3, n3 = gcm(data, cache_dir=SCRATCH, seed=0)
    dt_one = time.perf_counter() - t0
    print("   -> %.1fs  (vs %.1fs for the full cold extraction)" % (dt_one, dt_cold))
    assert np.array_equal(X1, X3), "matrix changed after re-extracting one family"
    assert n1 == n3
    assert np.array_equal(y1, y3) and np.array_equal(t1, t3) and np.array_equal(g1, g3)
    print("   [ok] assembled matrix identical to the original, column for column")
    assert dt_one < dt_cold * 0.6, (
        "re-extracting one family took %.1fs vs %.1fs cold -- it re-extracted too much"
        % (dt_one, dt_cold))

    # ------------------------------------------------- 4. version bump invalidates
    print("\n-- 4. bumping a family's `version` invalidates just that family --")
    fam = [f for f in ns["REGISTRY"] if f.name == "iqr"][0]
    k_before = ns["family_key"](fam)
    type(fam).version = 2
    k_after = ns["family_key"](fam)
    print("   iqr family_key: %s -> %s" % (k_before, k_after))
    assert k_before != k_after
    X4, *_ = gcm(data, cache_dir=SCRATCH, seed=0)
    n_iqr_files = len([f for f in os.listdir(d) if f.startswith("fam__iqr__")])
    print("   iqr files on disk after re-extract: %d (stale one removed)" % n_iqr_files)
    assert n_iqr_files == 1
    assert np.array_equal(X1, X4), "same code, bumped version -> same numbers expected"
    type(fam).version = 1
    print("   [ok]")

    # ------------------------------------------------- 5. subset assembly
    print("\n-- 5. families= subset --")
    Xs, ys, ts, ws, gs, nsub = gcm(data, cache_dir=SCRATCH, seed=0,
                                   families={"e2", "lmom"}, verbose=False)
    print("   families={'e2','lmom'} -> X %s, names %s..." % (Xs.shape, nsub[:3]))
    assert Xs.shape[1] == 10 + 18
    cols = [n1.index(nm) for nm in nsub]
    assert np.array_equal(Xs, X1[:, cols])
    print("   [ok] subset columns match the full matrix")

    # ------------------------------------------------- 6. FEATURE_TO_FAMILY is live
    print("\n-- 6. feature_to_family() is rebuilt at call time --")
    f2f = ns["feature_to_family"]()
    assert f2f["mmd_e_w32"] == "mmd" and f2f["lmom_lkurt_e_w256"] == "lmom"
    n_before = len(f2f)

    class Toy(ns["TwoSampleFamily"]):
        name = "toytest"

        def stat(self, w, r):
            return float(w.size)

    ns["REGISTRY"].append(Toy())
    f2f2 = ns["feature_to_family"]()
    print("   registry %d -> %d families, map %d -> %d entries"
          % (len(fams), len(ns["REGISTRY"]), n_before, len(f2f2)))
    assert f2f2.get("toytest_e_w32") == "toytest", "new family missed by the map"
    ns["REGISTRY"].pop()
    print("   [ok]")

    # ------------------------------------------------- 7. null side-cache
    print("\n-- 7. null.joblib side-cache --")
    ns["save_null"](skey, "fset-abc", {"cutoff": 0.55, "n_perm": 100}, cache_dir=SCRATCH)
    got = ns["load_null"](skey, "fset-abc", cache_dir=SCRATCH)
    miss = ns["load_null"](skey, "fset-zzz", cache_dir=SCRATCH)
    print("   stored:", got, "| unknown key:", miss)
    assert got["cutoff"] == 0.55 and miss is None
    assert "null.joblib" in os.listdir(d)
    print("   [ok]")

    print("\n" + "=" * 74)
    print("PHASE 4: PASS")
    print("=" * 74)


if __name__ == "__main__":
    main()
