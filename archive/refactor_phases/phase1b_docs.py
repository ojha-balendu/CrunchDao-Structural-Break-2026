"""PHASE 1 (part b) -- fix the prose/comments left stale by the registry cull.

Separate from phase1_prune.py only so the code edit and the doc edit are reviewable
apart from each other. Run once, after phase1_prune.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell


def sub(nb, needle, old, new, ctype=None):
    i = find_cell(nb, needle, ctype)
    text = src(nb["cells"][i])
    assert old in text, "anchor not found in cell %d: %r" % (i, old[:60])
    set_src(nb["cells"][i], text.replace(old, new))


def main():
    nb = nbedit.load()

    # ---- title cell: principles 3/4/5 -------------------------------------
    sub(nb, "# Structural Break (Real-Time)",
        """3. **Feature bank = (what changed) x (how to aggregate under unknown tau).**
   - what changed: location / scale / shape / tails / dependence / trend
   - how to aggregate: expanding, EWMA (3 decays), trailing windows (4 lengths),
     Page-CUSUM (3 drifts).  Multi-scale replaces the O(t) "max over candidate tau".

4. **Two channels.** The same bank is run on
   - `z`  = PIT-normal of the raw value        -> marginal / distributional change
   - `ze` = PIT-normal of the AR(p) residual   -> dependence / dynamics change

5. **Context features** (historical kurtosis, AR coefs, sd/MAD, n_H) carry no break evidence.
   They let LightGBM learn *how much to trust* each evidence feature for this kind of process.
""",
        """3. **Feature bank = (what changed) x (how to aggregate under unknown tau).**
   - what changed: after the Phase 1 cull, **scale and shape** -- every family that
     measured location, dependence or trend scored at the permutation-null floor and
     was archived (`tools/archived_families.py`).
   - how to aggregate: expanding, EWMA (3 decays in `ALPHAS`), trailing windows
     (3 lengths in `WINDOWS`), Page-CUSUM (3 drifts in `DRIFTS`). Multi-scale replaces
     the O(t) "max over candidate tau".

4. **Four channels, two registered banks.** `ChannelState` is maintained for all of
   `CH = ("z", "z2", "e", "e2")`:
   - `z`  = PIT-normal of the raw value,     `z2` = `(z^2 - 1)/sqrt(2)` -> its scale
   - `e`  = PIT-normal of the AR(p) residual, `e2` = the same, for the residual
   Only `z2` / `e2` are registered as ChannelBankFamily instances; `z` and `e` are read
   by `LMomentsFamily` and the two-sample families, so their channels stay live even
   though their banks are gone.

5. **Context features were cut.** The `h_*` block (historical kurtosis, AR coefs, sd/MAD,
   n_H) was meant to let LightGBM learn *how much to trust* each evidence feature. It never
   paid for itself in ablation, so it is archived along with the rest of the cull.
   `HistoricalRef.context` is still computed (it is a handful of floats, once per series)
   in case the block is restored.
""")

    # ---- 1c markdown: `u` has no consumer any more ------------------------
    sub(nb, "### 1c. Registry infrastructure",
        "values, `zr`/`u` (needed by the tail/chi2 families), the `channels` dict, and `HistoricalRef`.",
        "values, `zr` (read by `TailFamily`), `u`, the `channels` dict, and `HistoricalRef`.\n"
        "`u` currently has **no consumer** -- `Chi2Family` was its only reader and was cut in the\n"
        "Phase 1 cull. It is left on `Ctx` because it is one float per step and it is what any\n"
        "new binned/EDF family would want.")

    # ---- 1c code cell: docstrings naming deleted classes ------------------
    sub(nb, "class ChannelState:",
        """    Owned by the driver, pushed once per step; any family needing this channel
    reads it off `ctx` instead of recomputing it (e.g. ContrastFamily/SlopeFamily
    both need the window running-sums `rs` that the channel bank also emits from).\"\"\"""",
        """    Owned by the driver, pushed once per step; any family needing this channel
    reads it off `ctx` instead of recomputing it, rather than each family keeping its
    own ring buffer over the same values.\"\"\"""")

    sub(nb, "class FeatureFamily:",
        """    IMPORTANT: REGISTRY holds one prototype *instance* per family (so a family
    with constructor args, like ChannelBankFamily(channel), only needs to be
    built once). init()/update() mutate `self` (e.g. AutocorrFamily's z1/z2,
    SlopeFamily's slope_B), so every StreamFeatures MUST work on its own""",
        """    IMPORTANT: REGISTRY holds one prototype *instance* per family (so a family
    with constructor args, like ChannelBankFamily(channel), only needs to be
    built once). init()/update() mutate `self` (e.g. TailFamily's ring buffers and
    running counts), so every StreamFeatures MUST work on its own""")

    # ---- 1 intro: the paragraph about contrast/slope ----------------------
    sub(nb, "## 1. The feature engine",
        """Several existing blocks (the recent-vs-previous contrast, the local-trend slope) read the
window running-sums that the per-channel moment bank also emits from -- so those trackers
(ring buffer, window sums, EWMA, CUSUM state) are pulled out into a shared `ChannelState`,
pushed once per step by the driver, and any family needing a channel reads it off a per-step
`Ctx` instead of recomputing it.""",
        """Per-channel trackers (ring buffer, window sums, EWMA, CUSUM state) live in a shared
`ChannelState`, pushed once per step by the driver; any family needing a channel reads it off
a per-step `Ctx` instead of maintaining its own copy. `Ctx.sorted_window(channel, W)` memoizes
the sorted trailing window for the rest of the step, which matters because most of the
surviving registry is two-sample tests that all want the same sorted view.""")

    # ---- diagnostic markdown: the h_* bullet -----------------------------
    sub(nb, "### Diagnostic run:",
        """- **`h_*` context features carry no break evidence by design** (design principle #5, section 1)
  -- they let the model learn how much to trust each evidence feature for this kind of process.
  They pay off only through interactions, so they are judged by ablation (section 3c), never by
  this table.
""",
        """- **A column with `std == 0` across the matrix is degenerate, not merely weak.** It gets
  TS-AUC exactly 0.5 and is indistinguishable in this table from a real feature with no
  signal, so `run_feature_diagnostic` now prints a warning listing any such column. An
  exact `0.500000` in the results table with no warning means "no signal"; with a warning
  it means "this family produced a constant and needs looking at".
""")

    # ---- cache cell comment: perm_ent example is archived now ------------
    sub(nb, "FEATURE_TO_FAMILY = {",
        """# Exact feature -> family-name map, built directly from REGISTRY (not guessed
# from the string, which breaks for any family whose .name itself contains an
# underscore, e.g. "perm_ent_z_w16" would wrongly split to "perm").""",
        """# Exact feature -> family-name map, built directly from REGISTRY (not guessed
# from the string, which breaks for any family whose .name itself contains an
# underscore -- the archived "perm_ent" family is the worked example: "perm_ent_z_w16"
# would wrongly split to "perm". No family name in the pruned registry contains an
# underscore, so the string-splitting shortcut would happen to work today; it is still
# wrong and the next family added could well break it.""")

    nbedit.save(nb)
    print("phase 1b applied (prose/comment fixes)")


if __name__ == "__main__":
    main()
