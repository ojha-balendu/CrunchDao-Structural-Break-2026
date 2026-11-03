"""PHASE 1 test -- registry shape + finite output on synthetic series."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from nb_exec import run_notebook, make_synthetic

KEEP = ["e2", "z2", "tail", "time", "ftest", "bartlett", "levene", "fligner",
        "lmom", "iqr", "ansari", "mood", "sigtuk", "wass", "energy", "mmd", "entropy"]
REMOVED = ["acf", "ad", "archlm", "bowley", "chi2", "contrast", "cvm", "dak2", "e", "h",
           "hell", "hill", "hurst", "jb", "js", "kurt", "ljungbox", "moors", "mwu",
           "perm_ent", "sign", "skew", "slope", "welch", "z"]


def main():
    data = make_synthetic(3, seed=0)
    print("=" * 72)
    print("PHASE 1 TEST")
    print("=" * 72)
    print("\n-- executing notebook code cells (crunch cells skipped) --")
    ns = run_notebook(synthetic_train_data=data)

    REGISTRY = ns["REGISTRY"]
    names = [f.name for f in REGISTRY]
    print("\nlen(REGISTRY) = %d   (expect 17)" % len(REGISTRY))
    print("N_FEATURES    = %d   (expect 82)" % ns["N_FEATURES"])
    print("ACTIVE_FAMILIES =", ns["ACTIVE_FAMILIES"])
    print("\nfamilies in REGISTRY order:")
    for f in REGISTRY:
        print("    %-10s cost=%-7s n_features=%2d" % (f.name, f.cost, len(f.names())))

    assert len(REGISTRY) == 17, "expected 17 families, got %d" % len(REGISTRY)
    assert ns["N_FEATURES"] == 82, "expected 82 features, got %d" % ns["N_FEATURES"]
    assert sorted(names) == sorted(KEEP), "family set mismatch: %s" % sorted(names)
    assert len(set(names)) == len(names), "duplicate family names"
    for bad in REMOVED:
        assert bad not in names, "%r is still registered" % bad
    assert ns["ACTIVE_FAMILIES"] is None, "ACTIVE_FAMILIES should be None"
    print("\n[ok] registry shape, family set, ACTIVE_FAMILIES")

    # the archived source must still be there, verbatim and complete
    arch = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "archived_families.py")).read()
    for cls in ["ContrastFamily", "AutocorrFamily", "SignRunsFamily", "Chi2Family",
                "SlopeFamily", "ContextFamily", "WelchTFamily", "MannWhitneyFamily",
                "SkewFamily", "KurtFamily", "JarqueBeraFamily", "DAgostinoK2Family",
                "BowleySkewFamily", "MoorsKurtosisFamily", "CvMFamily", "ADFamily",
                "HellingerFamily", "JSFamily", "HillEstimatorFamily",
                "PermutationEntropyFamily", "ArchLMFamily", "LjungBoxFamily", "HurstFamily"]:
        assert "class %s(" % cls in arch, "%s missing from archived_families.py" % cls
    compile(arch, "archived_families.py", "exec")   # must at least parse
    print("[ok] all 23 removed classes archived verbatim and parseable")

    # -------------------------------------------------- 3 series through the extractor
    StreamFeatures = ns["StreamFeatures"]
    FEATURES = ns["FEATURES"]
    print("\n-- 3 synthetic series through StreamFeatures --")
    for did, xh, xo, tau in data:
        fx = StreamFeatures(xh)
        assert fx.names == FEATURES, "driver name order != FEATURES for %s" % did
        rows = np.asarray([fx.update(v) for v in xo], dtype=np.float64)
        assert rows.shape == (len(xo), 82), "bad shape %s for %s" % (rows.shape, did)
        assert np.all(np.isfinite(rows)), (
            "non-finite output for %s at columns %s"
            % (did, [FEATURES[c] for c in np.unique(np.nonzero(~np.isfinite(rows))[1])]))
        nconst = int(np.sum(rows.std(axis=0) == 0))
        print("    %s  hist=%5d online=%5d tau=%-5s -> %s  all finite, %d constant col(s)"
              % (did, len(xh), len(xo), tau, rows.shape, nconst))
    print("[ok] all outputs finite")

    # -------------------------------------------------- iqr is not constant (task 1.5)
    iqr_cols = [i for i, nm in enumerate(FEATURES) if nm.startswith("iqr_")]
    did, xh, xo, tau = data[0]
    fx = StreamFeatures(xh)
    rows = np.asarray([fx.update(v) for v in xo], dtype=np.float64)
    print("\n-- iqr columns on %s --" % did)
    for c in iqr_cols:
        print("    %-14s std=%.6g  min=%.4g  max=%.4g"
              % (FEATURES[c], rows[:, c].std(), rows[:, c].min(), rows[:, c].max()))
    assert all(rows[:, c].std() > 0 for c in iqr_cols), "iqr columns are constant"
    print("[ok] iqr columns are NOT constant in the current code")

    print("\n" + "=" * 72)
    print("PHASE 1: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
