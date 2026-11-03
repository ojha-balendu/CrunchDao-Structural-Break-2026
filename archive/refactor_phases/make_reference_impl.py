"""Snapshot the CURRENT (pruned, pre-optimization) implementations into
tools/reference_impl.py, verbatim, as the equivalence baseline for Phases 2-3.

Run this BEFORE tools/phase2_fast.py and tools/phase3_metrics.py.

The snapshot is source text, not an import of the notebook: the whole point is
that it keeps working after the notebook's own classes are replaced.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, find_cell, extract_block

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_impl.py")

# (anchor to find the cell, [top-level blocks to lift verbatim])
PLAN = [
    ("class ChannelState:", ["class ChannelState:", "class Ctx:"]),
    ("class ChannelBankFamily(FeatureFamily):",
     ["class ChannelBankFamily(", "class TailFamily(", "class TimeFamily("]),
    ("class TwoSampleFamily(FeatureFamily):",
     ["def _mean_var(", "def _avg_ranks(", "def _guard(", "class RefSub:",
      "class TwoSampleFamily(", "class FTestFamily(", "class BartlettFamily(",
      "class LeveneFamily(", "class FlignerKilleenFamily("]),
    ("class LMomentsFamily(FeatureFamily):",
     ["class LMomentsFamily(", "class IQRRatioFamily(", "class AnsariBradleyFamily(",
      "class MoodFamily(", "class SiegelTukeyFamily("]),
    ("class WassersteinFamily(TwoSampleFamily):",
     ["class WassersteinFamily(", "class EnergyDistanceFamily(", "class MMDRbfFamily("]),
    ("class DiffEntropyFamily(TwoSampleFamily):", ["class DiffEntropyFamily("]),
    ("class StreamFeatures:", ["class StreamFeatures:"]),
]

METRIC_PLAN = ("def ts_auc(", ["def ts_auc(", "def _folded_ts_auc_table(",
                               "def permutation_null_ts_auc("])

HEADER = '''"""Frozen reference implementations -- EQUIVALENCE TESTING ONLY.

Snapshot of the notebook's own code as it stood after the Phase 1 registry cull and
BEFORE the Phase 2 (fast families) / Phase 3 (fast metrics) rewrites. Every block
below is byte-identical to what the notebook contained at that point.

Nothing on the crunch submission path imports this. It exists so
tools/test_phase2.py and tools/test_phase3.py can diff the optimized notebook code
against the arithmetic it replaced.

    from nb_exec import run_notebook
    from reference_impl import build_reference
    ns  = run_notebook()          # the live (optimized) notebook
    ref = build_reference(ns)     # the frozen reference, sharing ns's config
    ref["StreamFeatures"]         # the pre-Phase-2 driver
    ref["ts_auc"]                 # the pre-Phase-3 sklearn metric

build_reference() layers the snapshot over a COPY of the notebook namespace, so the
reference families reuse the notebook's HistoricalRef, config constants (WINDOWS,
ALPHAS, ...), FeatureFamily base, _safe, _l_moments and stride_for -- the things
Phases 2-3 do not touch -- while overriding everything they do touch.
"""

# flake8: noqa
import bisect
import copy
import math
from typing import List

import numpy as np
import pandas as pd
from scipy import special as _sp
from sklearn.metrics import roc_auc_score


def ndtri(x):
    return _sp.ndtri(x)


REFERENCE_SOURCE = r\'\'\'
'''

FOOTER = '''
\'\'\'


def build_reference(ns):
    """Exec the frozen snapshot over a copy of the notebook namespace `ns`."""
    refns = dict(ns)
    refns["REGISTRY"] = []
    refns["ndtri"] = ndtri
    refns["roc_auc_score"] = roc_auc_score
    refns["np"] = np
    refns["pd"] = pd
    refns["math"] = math
    refns["bisect"] = bisect
    refns["copy"] = copy
    refns["List"] = List
    exec(compile(REFERENCE_SOURCE, "reference_impl.REFERENCE_SOURCE", "exec"), refns)
    return refns
'''

REGISTRATION = '''

# ---------------------------------------------------------------------------
#  Reference REGISTRY -- same 17 families, same order as the notebook's
# ---------------------------------------------------------------------------
for _ch in ("z2", "e2"):
    REGISTRY.append(ChannelBankFamily(_ch))
REGISTRY.append(TailFamily())
REGISTRY.append(TimeFamily())
REGISTRY.extend([FTestFamily(), BartlettFamily(), LeveneFamily(), FlignerKilleenFamily()])
REGISTRY.extend([LMomentsFamily(), IQRRatioFamily(),
                 AnsariBradleyFamily(), MoodFamily(), SiegelTukeyFamily()])
REGISTRY.extend([WassersteinFamily(), EnergyDistanceFamily(), MMDRbfFamily()])
REGISTRY.extend([DiffEntropyFamily()])
FEATURES = [nm for f in REGISTRY for nm in f.names()]
'''


def main():
    nb = nbedit.load()
    chunks = []

    for anchor, blocks in PLAN:
        text = src(nb["cells"][find_cell(nb, anchor, "code")])
        for hp in blocks:
            block, _, _ = extract_block(text, hp)
            chunks.append(block)

    anchor, blocks = METRIC_PLAN
    text = src(nb["cells"][find_cell(nb, anchor, "code")])
    metric_chunks = []
    for hp in blocks:
        block, _, _ = extract_block(text, hp)
        metric_chunks.append(block)

    body = "\n\n\n".join(chunks) + REGISTRATION + "\n\n" + "\n\n\n".join(metric_chunks) + "\n"
    assert "'''" not in body, "snapshot contains a triple-quote that would break the literal"

    with open(OUT, "w") as fh:
        fh.write(HEADER)
        fh.write(body)
        fh.write(FOOTER)

    n_cls = body.count("\nclass ") + body.startswith("class ")
    print("wrote %s  (%d lines, %d top-level classes)" % (OUT, body.count("\n") + 1, n_cls))


if __name__ == "__main__":
    main()
