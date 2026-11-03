# %% [markdown]
# ## Bucket 3 pruning: keep-one-in ablation
# Bucket 3 = centered spread of `e`: ftest, bartlett, levene, sigtuk, lmom_lscale (each at w32/w64/w256).
#
# * `noB3`  = the shipped 151 minus all 15 bucket-3 columns (136)
# * keep-one = noB3 + one test (all three windows), for each of the five tests
# * forward  = add a second (third...) test only if it clears `delta > 2*se` and 5/6 over the current set
# * final    = selected set vs full151 with the pruning rule `delta > -2*se`
#
# Prereqs (a fresh kernel is safest, so REGISTRY holds only the shipped families):
# section 0 (import _engine), section 1 (cache), section 2 (build_folds, CMP),
# and the ONE-CALL FAMILY TEST cell (defines `_cv` and `_stack`).
# Every finished run is written to cache/b3_results.json, so a crash or restart
# resumes where it stopped: re-run cell 1, then the cell you were on.

# %% [1] setup: rebuild the shipped 151 as PROD (cache, minus bucket 3) + fixed extra arrays
import os, json, gc, time
import numpy as np, pandas as pd

SHIPPED = ["z2", "e2", "tail", "time", "ftest", "bartlett", "levene", "lmom", "sigtuk",
           "wass", "energy", "mmd", "entropy", "bocpd",
           "garch", "argarch", "tgarch", "gjr", "recres", "wavvar", "hmm"]
EXTRA_FAMS = ["garch", "argarch", "tgarch", "gjr", "recres", "wavvar", "hmm"]   # live in cache/new_<fam>.npz

B3_TESTS = {
    "ftest":    [f"ftest_e_w{W}"       for W in (32, 64, 256)],
    "bartlett": [f"bartlett_e_w{W}"    for W in (32, 64, 256)],
    "levene":   [f"levene_e_w{W}"      for W in (32, 64, 256)],
    "sigtuk":   [f"sigtuk_e_w{W}"      for W in (32, 64, 256)],
    "lscale":   [f"lmom_lscale_e_w{W}" for W in (32, 64, 256)],
}
B3_ALL = [n for v in B3_TESTS.values() for n in v]

# the shipped feature names, straight from the engine's family classes
FAM_NAMES = {f.name: list(f.names()) for f in REGISTRY if f.name in SHIPPED}
assert set(FAM_NAMES) == set(SHIPPED), f"REGISTRY lacks {set(SHIPPED) - set(FAM_NAMES)} -- re-run section 0"
SHIP_NAMES = [n for fam in SHIPPED for n in FAM_NAMES[fam]]
assert len(SHIP_NAMES) == len(set(SHIP_NAMES)) == 151, len(SHIP_NAMES)
assert set(B3_ALL) <= set(SHIP_NAMES), set(B3_ALL) - set(SHIP_NAMES)

# 85-column cache: the 14 original families live here (mood/ansari/iqr/fligner are simply not selected)
X_all = np.load(f"{CACHE}/X_all.npy", mmap_mode="r")
BASE_NAMES = json.load(open(f"{CACHE}/names.json"))
assert len(BASE_NAMES) == X_all.shape[1] and X_all.shape[0] == y_all.size
cache_names = [n for fam in SHIPPED if fam not in EXTRA_FAMS for n in FAM_NAMES[fam]]
_miss = [n for n in cache_names if n not in BASE_NAMES]
assert not _miss, f"not in the 85-col cache: {_miss}"
assert len(cache_names) == 73, len(cache_names)

# _stack() reads the global PROD: fix it to the cached shipped columns MINUS bucket 3.
# Bucket 3 travels as extra arrays instead, so every config is just a different list of arrays.
PROD = [BASE_NAMES.index(n) for n in cache_names if n not in B3_ALL]
PROD_NAMES = [BASE_NAMES[i] for i in PROD]
assert len(PROD) == 58, len(PROD)


def load_shipped(fam):
    """The shipped columns of one extra family, in engine order (drops garch_persist, gjr asym, ...)."""
    want = FAM_NAMES[fam]
    with np.load(f"{CACHE}/new_{fam}.npz") as Z:
        nm = [str(s) for s in Z["names"]]
        miss = [n for n in want if n not in nm]
        assert not miss, f"{fam}: cached extraction lacks {miss} -- re-extract it with force=True"
        X = np.asarray(Z["X"], dtype=np.float32)
    X = np.ascontiguousarray(X[:, [nm.index(n) for n in want]])
    assert X.shape[0] == y_all.size and np.isfinite(X).all(), fam
    return X, want


REST_ARRAYS, REST_NAMES = [], []
for fam in EXTRA_FAMS:
    _X, _nm = load_shipped(fam)
    REST_ARRAYS.append(_X); REST_NAMES += _nm
    print(f"  {fam:8s} {_X.shape[1]:2d} cols")
del _X; gc.collect()

# bucket-3 columns out of the mmap in ONE pass, then one small array per test (~60 MB each)
_blk = np.asarray(X_all[:, [BASE_NAMES.index(n) for n in B3_ALL]], dtype=np.float32)
B3_X = {t: np.ascontiguousarray(_blk[:, [B3_ALL.index(n) for n in cols]]) for t, cols in B3_TESTS.items()}
del _blk; gc.collect()

assert len(PROD_NAMES) + len(REST_NAMES) + len(B3_ALL) == 151
assert set(PROD_NAMES + REST_NAMES + B3_ALL) == set(SHIP_NAMES)

FOLD_SEED, SEEDS = 11, (0, 1, 2)
FOLD_IDX = build_folds(FOLD_SEED)
RES_PATH = f"{CACHE}/b3_results.json"
RES = json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else {}
print(f"PROD {len(PROD)} | extra {len(REST_NAMES)} | bucket 3 {len(B3_ALL)} | cached runs: {list(RES)}")


def _label(tests):
    if not tests:
        return "noB3"
    if set(tests) == set(B3_TESTS):
        return "full151"
    return "keep[" + "+".join(sorted(tests)) + "]"


def run(tests):
    """CV of noB3 + the listed bucket-3 tests (all three windows each). Resumes from RES_PATH."""
    tests = list(tests)
    label = _label(tests)
    r = RES.get(label)
    if r is not None:
        assert (r["fold_seed"] == FOLD_SEED and r["seeds"] == list(SEEDS)
                and sorted(r["tests"]) == sorted(tests)), \
            f"{label}: cached with different settings -- delete it from {RES_PATH}"
        CMP[label] = np.array(r["aucs"])
        print(f"[{label}] cached | mean {CMP[label].mean():.5f}")
        return label
    arrays = REST_ARRAYS + [B3_X[t] for t in tests]
    names = PROD_NAMES + REST_NAMES + [n for t in tests for n in B3_TESTS[t]]
    t0 = time.time()
    aucs = _cv(label, arrays, names, seeds=SEEDS)
    RES[label] = dict(tests=tests, aucs=[float(a) for a in aucs], n_feats=len(names),
                      fold_seed=FOLD_SEED, seeds=list(SEEDS))
    with open(RES_PATH + ".tmp", "w") as f:
        json.dump(RES, f, indent=1)
    os.replace(RES_PATH + ".tmp", RES_PATH)
    print(f"   {label}: {len(names)} features | {(time.time() - t0) / 60:.1f} min")
    gc.collect()
    return label


def paired(a, b):
    """delta of config a over config b on identical fold-halves: (mean, se, n_better, n)."""
    d = CMP[a] - CMP[b]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size)), int((d > 0).sum()), d.size


def passes_add(a, b):
    """The adoption rule for ADDING: delta > 2*se and better on all but at most one fold-half."""
    m, se, k, n = paired(a, b)
    return m > 2 * se and k >= n - 1


def report(ref, labels):
    rows = []
    for lab in labels:
        m, se, k, n = paired(lab, ref) if lab != ref else (0.0, 0.0, 0, CMP[lab].size)
        rows.append(dict(config=lab, feats=RES[lab]["n_feats"], mean=CMP[lab].mean(), delta=m, se=se,
                         sigma=(m / se if se > 0 else 0.0), better=f"{k}/{n}"))
    df = pd.DataFrame(rows).sort_values("mean", ascending=False).round(5)
    print(f"\n--- vs {ref} ---\n" + df.to_string(index=False))
    return df


# %% [2] reference runs: full151 and noB3 (does the bucket matter at all?)
FULL = run(list(B3_TESTS))
NONE = run([])
assert RES[FULL]["n_feats"] == 151 and RES[NONE]["n_feats"] == 136
m, se, k, n = paired(NONE, FULL)
print(f"\nnoB3 vs full151: {m:+.5f} | se {se:.5f} | {m / se:+.1f} sigma | better {k}/{n}")
print("-> without bucket 3 the model is",
      "NON-INFERIOR (the bucket may be fully redundant)" if m > -2 * se else "measurably WORSE (something in it is needed)")


# %% [3] keep-one-in: noB3 + each test on its own
ONE = {t: run([t]) for t in B3_TESTS}
report(NONE, [NONE] + list(ONE.values()))

best = max(ONE, key=lambda t: CMP[ONE[t]].mean())
tied = [t for t in ONE if t == best or paired(ONE[t], ONE[best])[0] > -2 * paired(ONE[t], ONE[best])[1]]
WINNER = "ftest" if "ftest" in tied else best       # tie-break by design: signed, log-scale
print(f"\nhighest mean: {best} | statistically tied with it: {tied} | winner: {WINNER}")
m, se, k, n = paired(ONE[WINNER], NONE)
print(f"{WINNER} over noB3: {m:+.5f} | se {se:.5f} | {m / se:+.1f} sigma | better {k}/{n}")
SEL = [WINNER] if passes_add(ONE[WINNER], NONE) else []
print("-> first pick:", SEL or "NONE (no single test adds measurably over noB3)")


# %% [4] forward selection on top of the winner (stops at the first step that fails the add rule)
while SEL and len(SEL) < len(B3_TESTS):
    cur = run(SEL)
    cand = {t: run(SEL + [t]) for t in B3_TESTS if t not in SEL}
    report(cur, [cur] + list(cand.values()))
    t_best = max(cand, key=lambda t: CMP[cand[t]].mean())
    m, se, k, n = paired(cand[t_best], cur)
    if passes_add(cand[t_best], cur):
        SEL.append(t_best)
        print(f"-> ADD {t_best}: {m:+.5f} ({m / se:.1f} sigma, {k}/{n})")
    else:
        print(f"-> STOP: best addition {t_best} gives {m:+.5f} ({(m / se if se else 0):.1f} sigma, {k}/{n})")
        break
print("\nselected bucket-3 tests:", SEL or "none")


# %% [5] final verdict: selected set vs full151 (pruning rule: delta > -2*se)
def verdict(sel):
    lab = run(sel)
    m, se, k, n = paired(lab, FULL)
    ok = m > -2 * se
    print(f"{lab} vs full151: {m:+.5f} | se {se:.5f} | {m / se:+.1f} sigma | better {k}/{n} "
          f"-> {'NON-INFERIOR' if ok else 'LOSES'}")
    return ok


ok = verdict(SEL)
if not ok and WINNER not in SEL:            # fallback: the forward pass dropped everything but that lost
    print("fallback: keeping the winner alone")
    SEL = [WINNER]
    ok = verdict(SEL)

dropped = [c for t in B3_TESTS if t not in SEL for c in B3_TESTS[t]]
gone = [fam for fam in SHIPPED if set(FAM_NAMES[fam]) <= set(dropped)]
print("\nVERDICT:", "ADOPT" if ok else "KEEP ALL OF BUCKET 3")
if ok:
    print(f"keep  : {SEL or 'none'}")
    print(f"drop  : {len(dropped)} columns -> {dropped}")
    print(f"whole families removable (these also save extraction time): {gone or 'none'}")
    if any(c.startswith("lmom_") for c in dropped):
        print("note: dropping lscale only removes columns; lmom still runs for lskew/lkurt")