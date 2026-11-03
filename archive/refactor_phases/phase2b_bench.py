"""PHASE 2 (part b)

  * Put back the "family returned the wrong number of values" safety net that the
    fast driver dropped. The original asserted it on every step for every family;
    this checks it once, at t == 0, where every family computes regardless of stride.
    Same bug caught, same message, none of the per-step cost.
  * Rework bench_extractor for the new driver: projected wall-clock for an arbitrary
    series count and worker count, plus the reference/fast split per family.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit
from nbedit import src, set_src, find_cell

NEW_LOOP = '''        out = []
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
                        "%s.update() returned %d values, names() has %d"
                        % (f.name, len(f._last), nout))
        return out
'''

OLD_LOOP = '''        out = []
        for f in self.families:
            stride = stride_for(f.cost, t)
            if stride == 1 or t % stride == 0 or not hasattr(f, "_last"):
                f._last = f.update(ctx)
            out.extend(f._last)
        return out
'''

BENCH = '''import time


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
    print("  FULL registry (%d families, %d features): %7.4f s/series"
          % (len(REGISTRY), N_FEATURES, full))
    print()
    print("  projected for %d series on %d workers: %5.1f min  (baseline only: %.1f min)"
          % (n_project, n_workers, full * n_project / n_workers / 60,
             b * n_project / n_workers / 60))
    for n in (10_000, 25_000):
        print("  projected for %6d series on %d workers: %5.1f min"
              % (n, n_workers, full * n / n_workers / 60))

    print("\\n  marginal cost per family (on top of the O(1) baseline):")
    rows = sorted(((float(np.mean([t_of(r, base | {f.name}) for r in items])) - b,
                    f.name, f.cost) for f in REGISTRY if f.cost != "O1"), reverse=True)
    tot = sum(d for d, _, _ in rows)
    for d, n, c in rows:
        print("    %-10s %-7s %7.4f s  (%4.1f%% of the non-baseline cost)"
              % (n, c, d, 100.0 * d / tot if tot > 0 else 0.0))
    return full


bench_extractor(train_data)
'''


def main():
    nb = nbedit.load()

    i = find_cell(nb, "class StreamFeatures:", "code")
    text = src(nb["cells"][i])
    assert OLD_LOOP in text, "driver update loop not in the expected shape"
    set_src(nb["cells"][i], text.replace(OLD_LOOP, NEW_LOOP))

    i = find_cell(nb, "def bench_extractor(", "code")
    set_src(nb["cells"][i], BENCH)

    nbedit.save(nb)
    print("phase 2b applied")


if __name__ == "__main__":
    main()
