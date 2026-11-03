"""PHASE 1 -- prune the registry from 42 families / 187 features to 17 / 82.

Run once:  python tools/phase1_prune.py

What it does
  1. Lifts the 23 removed family classes out of the notebook verbatim into
     tools/archived_families.py, and unregisters the "z" / "e" ChannelBankFamily
     instances (the class itself stays -- z2 / e2 still use it).
  2. Deletes the dead legacy-equivalence section (markdown 1a, markdown 1e and the
     _run_equivalence_check cell). It calls StreamFeaturesLegacy / FEATURES_LEGACY,
     neither of which exists in the notebook any more, so it NameErrors on a fresh
     kernel.
  3. Fixes the ordering bug at the end of 1i (resolve_active_families was called
     two lines above its own def) and sets ACTIVE_FAMILIES = None.
  4. Repoints the bottom plot cell at features that still exist.
  5. Adds a constant-column warning to run_feature_diagnostic.
  6. Updates the markdown that enumerates families.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell

ARCHIVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archived_families.py")

# family .name -> class header prefix, grouped by the cell that defines it
CELL_1D_REMOVE = [
    ("contrast", "class ContrastFamily("),
    ("acf", "class AutocorrFamily("),
    ("sign", "class SignRunsFamily("),
    ("chi2", "class Chi2Family("),
    ("slope", "class SlopeFamily("),
    ("h", "class ContextFamily("),
]
CELL_1F_REMOVE = [
    ("welch", "class WelchTFamily("),
    ("mwu", "class MannWhitneyFamily("),
    ("skew", "class SkewFamily("),
    ("kurt", "class KurtFamily("),
    ("jb", "class JarqueBeraFamily("),
    ("dak2", "class DAgostinoK2Family("),
]
CELL_1G_REMOVE = [
    ("bowley", "class BowleySkewFamily("),
    ("moors", "class MoorsKurtosisFamily("),
]
CELL_1H_REMOVE = [
    ("cvm", "class CvMFamily("),
    ("ad", "class ADFamily("),
    ("hell", "class HellingerFamily("),
    ("js", "class JSFamily("),
]
CELL_1I_REMOVE = [
    ("hill", "class HillEstimatorFamily("),
    ("perm_ent", "class PermutationEntropyFamily("),
    ("archlm", "class ArchLMFamily("),
    ("ljungbox", "class LjungBoxFamily("),
    ("hurst", "class HurstFamily("),
]

ARCHIVE_HEADER = '''"""Families removed from the notebook's REGISTRY in the Phase 1 cull.

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
'''


def main():
    nb = nbedit.load()
    archive_sections = []

    # ---------------------------------------------------------------- 1d cell
    i = find_cell(nb, "class ChannelBankFamily(FeatureFamily):", "code")
    text = src(nb["cells"][i])
    text, grabbed = nbedit.remove_blocks(text, [hp for _, hp in CELL_1D_REMOVE])
    archive_sections.append(("1d -- ported bank", CELL_1D_REMOVE, grabbed))

    old_reg = text[text.index("for _ch in CH:"):]
    new_reg = '''# Only the two SQUARED (scale) channels are registered as feature banks. The raw
# location banks "z" / "e" were pruned -- but ChannelState is still maintained for
# every channel in CH, because LMomentsFamily reads "z" and "e" and the two-sample
# families read TWOSAMPLE_CHANNELS. Don't prune the channel plumbing along with the
# banks.
BANK_CHANNELS = ("z2", "e2")

for _ch in BANK_CHANNELS:
    REGISTRY.append(ChannelBankFamily(_ch))
REGISTRY.append(TailFamily())
REGISTRY.append(TimeFamily())

print("ported families registered:", len(REGISTRY),
      "| features so far:", sum(len(f.names()) for f in REGISTRY))
'''
    text = text.replace(old_reg, new_reg)
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1f cell
    i = find_cell(nb, "class TwoSampleFamily(FeatureFamily):", "code")
    text = src(nb["cells"][i])
    text, grabbed = nbedit.remove_blocks(text, [hp for _, hp in CELL_1F_REMOVE])
    archive_sections.append(("1f -- Tier 1 moment ladder", CELL_1F_REMOVE, grabbed))
    old_reg = text[text.index("REGISTRY.extend(["):]
    text = text.replace(old_reg, '''REGISTRY.extend([
    FTestFamily(), BartlettFamily(), LeveneFamily(), FlignerKilleenFamily(),
])
print("Tier 1 registered | features so far:", sum(len(f.names()) for f in REGISTRY))
''')
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1g cell
    i = find_cell(nb, "class LMomentsFamily(FeatureFamily):", "code")
    text = src(nb["cells"][i])
    text, grabbed = nbedit.remove_blocks(text, [hp for _, hp in CELL_1G_REMOVE])
    archive_sections.append(("1g -- Tier 2 robust analogues", CELL_1G_REMOVE, grabbed))
    old_reg = text[text.index("REGISTRY.extend(["):]
    text = text.replace(old_reg, '''REGISTRY.extend([
    LMomentsFamily(), IQRRatioFamily(),
    AnsariBradleyFamily(), MoodFamily(), SiegelTukeyFamily(),
])
print("Tier 2 registered | features so far:", sum(len(f.names()) for f in REGISTRY))
''')
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1h cell
    i = find_cell(nb, "class WassersteinFamily(TwoSampleFamily):", "code")
    text = src(nb["cells"][i])
    text, grabbed = nbedit.remove_blocks(text, [hp for _, hp in CELL_1H_REMOVE])
    archive_sections.append(("1h -- Tier 3 distribution distances", CELL_1H_REMOVE, grabbed))
    old_reg = text[text.index("REGISTRY.extend(["):]
    text = text.replace(old_reg, '''REGISTRY.extend([
    WassersteinFamily(), EnergyDistanceFamily(), MMDRbfFamily(),
])
print("Tier 3 registered | features so far:", sum(len(f.names()) for f in REGISTRY))
''')
    set_src(nb["cells"][i], text)

    # ---------------------------------------------------------------- 1i cell
    i = find_cell(nb, "class DiffEntropyFamily(TwoSampleFamily):", "code")
    text = src(nb["cells"][i])
    text, grabbed = nbedit.remove_blocks(text, [hp for _, hp in CELL_1I_REMOVE])
    archive_sections.append(("1i -- Tier 4 dynamics / tails", CELL_1I_REMOVE, grabbed))
    old_reg = text[text.index("REGISTRY.extend(["):]
    text = text.replace(old_reg, '''REGISTRY.extend([
    DiffEntropyFamily(),
])

# =====================================================================
#  REGISTRY IS NOW COMPLETE -- derive FEATURES / N_FEATURES from it
# =====================================================================
FEATURES = [nm for f in REGISTRY for nm in f.names()]
N_FEATURES = len(FEATURES)
assert len(set(FEATURES)) == N_FEATURES, "duplicate feature names in REGISTRY"
print("REGISTRY: %d families, %d total features" % (len(REGISTRY), N_FEATURES))


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
ACTIVE_FAMILIES = resolve_active_families(None)
''')
    set_src(nb["cells"][i], text)

    # ------------------------------------------------- write the archive file
    with open(ARCHIVE_PATH, "w") as fh:
        fh.write(ARCHIVE_HEADER)
        for title, spec, grabbed in archive_sections:
            fh.write("\n\n# %s\n# %s\n" % ("=" * 69, title))
            fh.write("# %s\n\n" % ("=" * 69))
            for fam_name, hp in spec:
                fh.write("\n# --- family %r ---\n%s\n" % (fam_name, grabbed[hp]))

    # ------------------------------------------- delete the legacy-equiv cells
    for needle, ctype in (("### 1a. Legacy snapshot", "markdown"),
                          ("### 1e. Equivalence check", "markdown"),
                          ("def _run_equivalence_check(", "code")):
        del nb["cells"][find_cell(nb, needle, ctype)]

    # -------------------------------------------------- bottom plot cell fix
    i = find_cell(nb, "features_to_test = [", "code")
    text = src(nb["cells"][i])
    old = text[text.index("# Xd/yd/td/gd/FEATURES_D come from"):]
    text = text.replace(old, '''# Xd/yd/td/gd/FEATURES_D come from the diagnostic run (section 3) -- a full-registry
# matrix over DIAG_N_SERIES sampled series, so any feature name in FEATURES_D works here.
# The old z_* location bank is gone (Phase 1 cull), so these pick one feature from each
# of the three surviving signal axes: the EWMA scale bank, a windowed variance test, and
# an L-moment shape statistic.
features_to_test = ['e2_ew0.02', 'ftest_e_w256', 'lmom_lkurt_e_w32']
plot_feature_auc_by_d(Xd, yd, td, gd, features_to_test, FEATURES_D)
''')
    set_src(nb["cells"][i], text)

    # ------------------------------------- constant-column warning (task 1.5)
    i = find_cell(nb, "def run_feature_diagnostic(", "code")
    text = src(nb["cells"][i])
    anchor = '    diag = _folded_ts_auc_table(X, y, t, feature_names)\n'
    assert anchor in text
    text = text.replace(anchor, '''    # A column that is constant across the whole matrix has TS-AUC exactly 0.5 and
    # tells you nothing -- but it looks identical to "real feature, no signal" in the
    # results table. `iqr` scored exactly 0.500000 in an earlier run for this reason.
    # Flag it here so a degenerate extraction can never be misread as a dead family.
    _sd = X.std(axis=0)
    _const = [nm for nm, s in zip(feature_names, _sd) if not (s > 0)]
    if _const:
        print("WARNING: %d/%d column(s) are CONSTANT across the matrix (std == 0) -- "
              "their TS-AUC of 0.5 means 'degenerate', not 'no signal':"
              % (len(_const), len(feature_names)))
        for nm in _const:
            print("    ", nm)

''' + anchor)
    set_src(nb["cells"][i], text)

    _rewrite_markdown(nb)

    nbedit.save(nb)
    print("phase 1 applied. archive ->", ARCHIVE_PATH)


def _rewrite_markdown(nb):
    i = find_cell(nb, "### 1d. Porting the existing bank", "markdown")
    set_src(nb["cells"][i], '''### 1d. Porting the existing bank into families (behavior-preserving)

Same arithmetic as the pre-registry code, relocated into `FeatureFamily` classes.

The Phase 1 cull removed `ContrastFamily`, `AutocorrFamily`, `SignRunsFamily`,
`Chi2Family`, `SlopeFamily` and `ContextFamily` from this cell, and unregistered the `z` /
`e` channel banks -- all of them sat at or below the permutation-null cutoff. Their source is
in `tools/archived_families.py` (offline parts bin, never imported), with restore instructions
at the top of that file.

| Family | Reads from `ctx` | Emits |
|---|---|---|
| `ChannelBankFamily(channel)` x2 (`z2`, `e2`) | `channels[c]` | exp / EWMA x3 / window x3 / CUSUM x3 (10 each) |
| `TailFamily` | `zr` | `tail{2,3}_w64`, `tail{2,3}_exp`, `beyond_range_exp`, `max_absz_exp` |
| `TimeFamily` | `t` | `t_step`, `t_log` -- **always kept**, exempt from every prune rule |

`ChannelState` is still built for all four channels in `CH`, not just the two registered
banks: `LMomentsFamily` reads `z` and `e`, and the two-sample families read
`TWOSAMPLE_CHANNELS`. Pruning a bank is not the same as pruning its channel.
''')

    i = find_cell(nb, "### 1f. Tier 1", "markdown")
    set_src(nb["cells"][i], '''### 1f. Tier 1 -- the moment ladder

Every family below is a **two-sample test: historical reference (`h.e_ref`) vs. the trailing
window ending at `t`**, evaluated at every length in `WINDOWS` on the `TWOSAMPLE_CHANNELS`
channels (3 features per family). Where a closed-form null exists we emit the standardized
statistic directly -- that's what makes it pivotal.

What survived the Phase 1 cull here is exactly the **scale** block: F-test, Bartlett,
Levene (Brown-Forsythe) and Fligner-Killeen. The location tests (Welch t, Mann-Whitney) and
the normality/shape omnibus tests (D'Agostino skewness, Anscombe-Glynn kurtosis, Jarque-Bera,
D'Agostino-Pearson K²) all sat at the noise floor and are archived in
`tools/archived_families.py`.

`TwoSampleFamily` is the shared base: it derives `names()` from `(stat, channel, W)`, pulls
the window (raw or sorted, per subclass) via `Ctx`, and standardizes with a guard against
degenerate windows (returns 0.0, never NaN/inf).
''')

    i = find_cell(nb, "### 1g. Tier 2", "markdown")
    set_src(nb["cells"][i], '''### 1g. Tier 2 -- robust analogues

The diagnostic says we win on essentially **one** scale statistic (`e2_*`). This tier gives
the model several more, with different robustness/efficiency profiles: L-moments
(L-scale/L-skew/L-kurt -- linear combinations of order statistics, far lower sampling variance
than conventional moments under heavy tails), the quantile-based IQR ratio, and three
rank-based distribution-free scale tests (Ansari-Bradley, Mood, Siegel-Tukey -- the scale
analogue of Mann-Whitney).

Bowley skewness and Moors' octile kurtosis were cut in the Phase 1 cull (both shape, both at
the noise floor); they're in `tools/archived_families.py`.

`LMomentsFamily` is the one family here that is **not** a `TwoSampleFamily`: it inherits
`channels = ("z", "e")` from the base class, so it emits 2 channels x 3 statistics x 3 windows
= 18 features, and it is the reason the `z` `ChannelState` still has to be maintained.

Where no simple closed-form standard error exists (L-moments), the standardization is a
documented, dimensionally-consistent approximation (deviation from the reference value, scaled
by `~1/sqrt(m)`) rather than an exact asymptotic pivot -- a deliberate tradeoff, measured by
the permutation-null diagnostic in section 3 rather than assumed here.
''')

    i = find_cell(nb, "### 1h. Tier 3", "markdown")
    set_src(nb["cells"][i], '''### 1h. Tier 3 -- omnibus / full-distribution distance

Wasserstein-1, energy distance, and MMD with an RBF kernel (bandwidth via the
median-pairwise-distance heuristic). Cramér-von Mises, Anderson-Darling, Hellinger and
Jensen-Shannon were cut in the Phase 1 cull -- see `tools/archived_families.py`. Note
`JSFamily` subclasses `HellingerFamily`, so restoring `js` means restoring `hell` too.

**KS is not reimplemented here.** The causal sup-KS scan near the bottom of the notebook
(`ks_scan_scores`) already is this test, fully causal, and is evaluated separately as a
standalone competing baseline against the whole LightGBM bank.

`cost = "heavy"`: these are `O(W log W)` to `O(W * n_ref_sub)` per evaluation (pairwise
distances for energy distance / MMD, capped via `RefSub` at 192 reference points so a large
`n_H` doesn't blow up the cost) -- they get the coarsest stride.

`MMDRbfFamily` inherits `init()` from `EnergyDistanceFamily`, which builds a `RefSub` for
**both** `z` and `e`; MMD's bandwidth is the median pairwise distance over the two
concatenated. That `z` reference is load-bearing even though `TWOSAMPLE_CHANNELS == ("e",)`.
''')

    i = find_cell(nb, "### 1i. Tier 4", "markdown")
    set_src(nb["cells"][i], '''### 1i. Tier 4 -- properties no moment captures, then close out the registry

All that survived the Phase 1 cull in this tier is the Vasicek differential-entropy spacings
estimator. The Hill tail-index estimator, permutation entropy, order-1 ARCH-LM, Ljung-Box and
the R/S Hurst exponent are archived in `tools/archived_families.py`.

Once `entropy` is registered, `FEATURES` / `N_FEATURES` are derived from the complete
`REGISTRY` -- the only place they're computed, never hand-written. The pruned registry is
**17 families / 82 features**:

| | families |
|---|---|
| bank + structural | `e2`, `z2`, `tail`, `time` |
| scale tests | `ftest`, `bartlett`, `levene`, `fligner` |
| robust / rank scale | `lmom`, `iqr`, `ansari`, `mood`, `sigtuk` |
| distribution distance | `wass`, `energy`, `mmd` |
| entropy | `entropy` |

`ACTIVE_FAMILIES` is the speed/selection switch used by `infer()` and the crunch-submission
`train()` (sections 4/5): `None` means "every registered family", a `set` of family `.name`s
restricts to that subset. After the cull those are the same set, so it is `None` --
the registry itself is the selection. `resolve_active_families()` validates any subset against
`REGISTRY` up front so a typo'd family name fails loudly instead of silently dropping every
feature it would have added.

(`resolve_active_families` is defined **above** its first call in the cell below. It used to be
called two lines before its own `def`, which NameErrors on a fresh kernel.)
''')


if __name__ == "__main__":
    main()
