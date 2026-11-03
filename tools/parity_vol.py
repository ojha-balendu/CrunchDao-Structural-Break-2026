"""Parity check: does the SUBMISSION notebook reproduce the cached research extractions?

  python tools/parity_vol.py [--series 20] [--families garch,argarch,tgarch,gjr]

Converts submission.ipynb, imports it in this fresh process, runs StreamFeatures over the online
points of N series spread across cache/g_all.npy, and compares every produced column against the
cached column of the same name (cache/new_{garch,argarch,tgarch,gjr}.npz) on that series' rows.

Cached columns that the final families no longer emit are ignored and reported.
Tolerance is 1e-4: the cache is float32.
"""
import argparse, contextlib, importlib.util, io, json, os, subprocess, sys, tempfile, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FAMILIES = ("garch", "argarch", "tgarch", "gjr")
IGNORE = {"garch_persist", "gjr_gamma"}          # plus any gjr_asym_* (prefix match below)
TOL = 1e-4


def ignored(name):
    return name in IGNORE or name.startswith("gjr_asym")


def load_module():
    out = os.path.join(tempfile.mkdtemp(prefix="parity_"), "_parity_main.py")
    r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook",
                        os.path.join(ROOT, "submission.ipynb"), out, "--override"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    spec = importlib.util.spec_from_file_location("_parity_main", out)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=20)
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma list; each needs cache/new_<name>.npz")
    a = ap.parse_args()
    fam_list = tuple(x.strip() for x in a.families.split(",") if x.strip())

    g_all = np.load(os.path.join(CACHE, "g_all.npy"))
    t_all = np.load(os.path.join(CACHE, "t_all.npy"))
    uniq = np.unique(g_all)
    ids = uniq[np.linspace(0, uniq.size - 1, a.series).astype(int)]
    print("series under test (%d, spread across %d): %s" % (ids.size, uniq.size, list(ids[:6]) + ["..."]))

    m = load_module()
    print("registry: %d families, %d features" % (len(m.REGISTRY), m.N_FEATURES))
    fams = {f.name: f for f in m.REGISTRY if f.name in fam_list}
    assert set(fams) == set(fam_list), "missing families: %s" % (set(fam_list) - set(fams))

    os.chdir(ROOT)
    import crunch
    crunch_tools = crunch.load_notebook()
    with contextlib.redirect_stdout(io.StringIO()):
        train_data, _ = crunch_tools.load_data()
    want = set(int(i) for i in ids)
    series = {int(r[0]): (np.asarray(r[1], dtype=np.float64), np.asarray(list(r[2]), dtype=np.float64))
              for r in train_data if int(r[0]) in want}
    assert len(series) == ids.size, "found %d of %d series in train_data" % (len(series), ids.size)

    # ---- run the notebook's own families over each series
    t0 = time.time()
    produced = {}          # sid -> (names, matrix)
    for sid in ids:
        xh, xo = series[int(sid)]
        fx = m.StreamFeatures(xh, active_families=set(fam_list))
        rows = [fx.update(v) for v in xo]
        produced[int(sid)] = (list(fx.names), np.asarray(rows, dtype=np.float64))
    names_out = produced[int(ids[0])][0]
    print("extracted %d series in %.1fs | %d columns: %s"
          % (ids.size, time.time() - t0, len(names_out), names_out[:4] + ["..."]))

    # ---- row indices per series, in t order, matching the cache layout
    rowsel = {}
    for sid in ids:
        idx = np.flatnonzero(g_all == sid)
        rowsel[int(sid)] = idx[np.argsort(t_all[idx], kind="stable")]
    for sid in ids:
        n_cache, n_run = rowsel[int(sid)].size, produced[int(sid)][1].shape[0]
        assert n_cache == n_run, "series %d: cache has %d rows, run produced %d" % (sid, n_cache, n_run)

    # ---- compare, one cached family file at a time (each X is ~200-280 MB)
    table, failures, skipped = [], [], []
    for fam in fam_list:
        path = os.path.join(CACHE, "new_%s.npz" % fam)
        assert os.path.exists(path), "missing %s" % path
        Z = np.load(path)
        cached_names = [str(s) for s in Z["names"]]
        X = Z["X"]
        del Z
        skipped += [n for n in cached_names if ignored(n)]
        missing = [n for n in cached_names if not ignored(n) and n not in names_out]
        extra = [n for n in names_out if n.startswith(fam + "_") and n not in cached_names]
        worst, worst_col, worst_sid = 0.0, None, None
        for nm in cached_names:
            if ignored(nm) or nm not in names_out:
                continue
            ci, pi = cached_names.index(nm), names_out.index(nm)
            for sid in ids:
                c = X[rowsel[int(sid)], ci].astype(np.float64)
                p = produced[int(sid)][1][:, pi]
                d = float(np.max(np.abs(c - p))) if c.size else 0.0
                if d > worst:
                    worst, worst_col, worst_sid = d, nm, int(sid)
        del X
        ok = worst <= TOL and not missing
        table.append((fam, len(cached_names), worst_col, worst, worst_sid, ok, missing, extra))
        if not ok:
            failures.append(fam)
        print("  %-8s worst %-22s %.3e  (series %s)%s" % (fam, worst_col, worst, worst_sid,
              "" if ok else "   <-- FAIL"), flush=True)

    print("\n%-9s %8s %-24s %12s %10s  %s" % ("family", "cached", "worst column", "max abs diff", "series", "verdict"))
    for fam, ncol, col, worst, sid, ok, missing, extra in table:
        print("%-9s %8d %-24s %12.3e %10s  %s%s%s"
              % (fam, ncol, col, worst, sid, "OK" if ok else "FAIL",
                 "" if not missing else " | cached column not produced: %s" % missing,
                 "" if not extra else " | produced but not cached: %s" % extra))
    if skipped:
        print("ignored cached columns (no longer emitted): %s" % sorted(set(skipped)))
    print("\ntolerance %.0e | %s" % (TOL, "PARITY OK" if not failures else "PARITY FAILED: %s" % failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
