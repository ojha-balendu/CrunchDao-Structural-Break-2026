"""PHASE 4 -- replace the whole-matrix disk cache with a per-family column store.

The old cache key covered the entire feature-name list, so adding one family
invalidated everything and re-extracted all 17. The new layout stores one .npy per
family, keyed by that family's own identity, so adding a family costs that family.

    cache/<sample_key>/meta.joblib                  y, t, w, g, series_ids, KEEP_EVERY, seed
    cache/<sample_key>/fam__<name>__<fam_key>.npy   float32 (n_rows, len(names()))
    cache/<sample_key>/null.joblib                  CUTOFF + null table per feature set

Also promotes REF_MAX from a local inside HistoricalRef.__init__ to a module constant
so family_key() can hash it, and adds FeatureFamily.version.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell

CACHE_CELL = '''import fnmatch
import glob
import hashlib
import inspect
import json
import shutil

CACHE_DIR = "cache"


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


FEATURE_TO_FAMILY = feature_to_family()   # convenience snapshot for interactive use;
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
            print("cache %s | extracting %d of %d families over %d series: %s"
                  % (skey, len(missing), len(wanted), len(items),
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
        print("cache %s | all %d families cached (%d series, %d rows)"
              % (skey, len(wanted), len(items), len(meta["y"])))

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
'''

CACHE_MD = '''## 2b. Per-family disk cache

One `.npy` per family per sample, not one blob per feature list:

```
cache/<sample_key>/meta.joblib                  y, t, w, g, series_ids, KEEP_EVERY, seed
cache/<sample_key>/fam__<name>__<fam_key>.npy   float32 (n_rows, len(names()))
cache/<sample_key>/null.joblib                  CUTOFF + null table, per feature set
```

`sample_key` identifies the **row set** (sorted series ids + `KEEP_EVERY` + subsample
flag + seed). `family_key` identifies one family's **output**: its `.name`, its
`names()`, a manual `version` class attribute, `inspect.getsource(type(fam))`, and the
extractor constants it reads (`WINDOWS`, `TWOSAMPLE_CHANNELS`, `ALPHAS`, `DRIFTS`,
`TAIL_W`, `BIN_W`, `NBINS`, `AR_P`, `RefSub.CAP`, `REF_MAX`).

This is what makes adding a feature cheap. The old cache keyed on the whole feature
list, so one new family invalidated the lot and re-extracted all 17. Now
`get_cached_matrix` loads every family that exists and calls `build_dataset` with
`active_families={the missing ones}` only.

**`version` is not optional bookkeeping.** `inspect.getsource` has no file to read for
a class defined in a notebook cell, so in Jupyter it usually raises and the source
falls back to `""` silently. When that happens the *only* thing separating v1 of a
family from v2 is `version`, so bump it by hand after editing a family body or you
will get the old columns back without a warning.

Rows are checked, not assumed: whenever a later call extracts a new family into an
existing sample, the fresh `y`/`t`/`g` are asserted equal to `meta.joblib`'s before
any column is spliced in.

`clear_cache()` wipes the directory; `clear_cache("mmd")` drops one family everywhere.
`cache_status(sample)` prints what is on disk. `FEATURE_TO_FAMILY` is now built by
`feature_to_family()` **at call time** -- the frozen module-level dict missed any
family registered after this cell ran, which is exactly what section 3d does.
'''


def main():
    nb = nbedit.load()

    # ---- REF_MAX: local inside HistoricalRef.__init__ -> module constant ----
    i = find_cell(nb, "class HistoricalRef:", "code")
    text = src(nb["cells"][i])
    old = """        REF_MAX = 256   # the two-sample reference is a smooth deterministic quantile
                        # grid, not raw data -- 256 knots reproduce it to well within
                        # the sampling noise of any window of length <= 256.
        _zr"""
    assert old in text, "REF_MAX local not found in the expected shape"
    text = text.replace(old, "        # REF_MAX lives in the CONFIG cell below: the per-family\n"
                             "        # cache key (section 2b) has to hash it.\n        _zr")
    set_src(nb["cells"][i], text)

    i = find_cell(nb, "AR_P    = 3", "code")
    text = src(nb["cells"][i])
    old = "AR_P    = 3                      # AR order for the whitening channel\n"
    assert old in text
    text = text.replace(old, old +
                        "REF_MAX = 256                    # knots in the deterministic PIT-normal\n"
                        "                                 # quantile grid that is the reference side\n"
                        "                                 # of every two-sample test. The grid is\n"
                        "                                 # smooth, not raw data, so 256 knots\n"
                        "                                 # reproduce it well inside the sampling\n"
                        "                                 # noise of any window of length <= 256.\n"
                        "                                 # Module-level so the per-family cache key\n"
                        "                                 # (section 2b) can hash it.\n")
    set_src(nb["cells"][i], text)

    # ---- FeatureFamily.version ----
    i = find_cell(nb, "class FeatureFamily:", "code")
    text = src(nb["cells"][i])
    old = '''    name: str = "base"
    channels: tuple = ("z", "e")
    cost: str = "O1"
'''
    assert old in text
    text = text.replace(old, '''    name: str = "base"
    channels: tuple = ("z", "e")
    cost: str = "O1"
    version: int = 1                  # bump by hand after editing this family's body:
                                      # it is part of the per-family cache key (2b), and
                                      # inspect.getsource() usually cannot see a class
                                      # defined in a notebook cell, so `version` is the
                                      # only thing that invalidates stale columns.
''')
    set_src(nb["cells"][i], text)

    # ---- the cache cell ----
    i = find_cell(nb, "FEATURE_TO_FAMILY = {", "code")
    set_src(nb["cells"][i], CACHE_CELL)

    i = find_cell(nb, "## 2b. Disk cache", "markdown")
    set_src(nb["cells"][i], CACHE_MD)

    # ---- callers that read the frozen FEATURE_TO_FAMILY ----
    i = find_cell(nb, "def run_feature_diagnostic(", "code")
    text = src(nb["cells"][i])
    old = '    diag["family"] = diag["feature"].map(FEATURE_TO_FAMILY)\n'
    assert old in text
    set_src(nb["cells"][i], text.replace(old, '    diag["family"] = diag["feature"].map(feature_to_family())\n'))

    i = find_cell(nb, "def ablate(", "code")
    text = src(nb["cells"][i])
    old = """    X, y, t, w, g, feature_names = cache
    name_to_idx = {nm: i for i, nm in enumerate(feature_names)}
    baseline_idx = np.asarray(baseline_idx)
    baseline_names = {feature_names[i] for i in baseline_idx}
    baseline_families = {FEATURE_TO_FAMILY[nm] for nm in baseline_names}
"""
    assert old in text
    text = text.replace(old, """    X, y, t, w, g, feature_names = cache
    f2f = feature_to_family()
    name_to_idx = {nm: i for i, nm in enumerate(feature_names)}
    baseline_idx = np.asarray(baseline_idx)
    baseline_names = {feature_names[i] for i in baseline_idx}
    baseline_families = {f2f[nm] for nm in baseline_names}
""")
    text = text.replace("mod_names = {nm for nm in baseline_names if FEATURE_TO_FAMILY[nm] != fam}",
                        "mod_names = {nm for nm in baseline_names if f2f[nm] != fam}")
    text = text.replace("fam_names = {nm for nm in feature_names if FEATURE_TO_FAMILY[nm] == fam}",
                        "fam_names = {nm for nm in feature_names if f2f[nm] == fam}")
    set_src(nb["cells"][i], text)

    nbedit.save(nb)
    print("phase 4 applied")


if __name__ == "__main__":
    main()
