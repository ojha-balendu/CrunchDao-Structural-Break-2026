"""Verify the submission notebook actually works after Crunch's conversion.

  python tools/check_submission.py submission.ipynb

1. converts the notebook (fails on any error)
2. imports the converted script in a FRESH process (no notebook state)
3. asserts the registry/feature counts survived the conversion
4. trains on synthetic series and streams infer() the way Crunch does
5. checks determinism (Crunch re-runs 10% of inference, tolerance 1e-8)
"""
import json, os, subprocess, sys, tempfile, textwrap

NB = sys.argv[1] if len(sys.argv) > 1 else "submission.ipynb"
OUT = os.path.abspath("tools/_main.py")

print("== 1. convert")
r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", NB, OUT, "--override", "-v"],
                   capture_output=True, text=True)
sys.stdout.write(r.stdout[-2000:]); sys.stderr.write(r.stderr[-4000:])
assert r.returncode == 0, "conversion FAILED"
warn = sorted({l.split("`")[1] for l in (r.stdout + r.stderr).split("\n") if "global variable" in l and "`" in l})
print("commented-out globals:", warn)
ALLOWED = {"crunch_tools", "train_data", "test_data"}
bad = [w for w in warn if w not in ALLOWED]
assert not bad, "these are needed at runtime but get commented out: %s" % bad

# Crunch only accepts module-level imports: an indented `import x` / `from x import y` in the
# converted script means the notebook hid one inside a function, class or block.
def _is_nested_import(line):
    if not line[:1].isspace():
        return False
    s = line.lstrip()
    return s.startswith("import ") or (s.startswith("from ") and " import " in s)

nested = [(i, l.rstrip()) for i, l in enumerate(open(OUT), 1) if _is_nested_import(l)]
assert not nested, "nested (indented) imports in the converted script: %s" % nested[:5]
print("nested-import check: clean")

# What Crunch will install: let the converter detect the notebook's requirements. It writes
# requirements*.txt into the CURRENT directory, so run it in a scratch dir, never the repo root.
req_dir = tempfile.mkdtemp(prefix="reqs_")
r = subprocess.run([sys.executable, "-m", "crunch_convert", "notebook", os.path.abspath(NB),
                    "main.py", "--override", "--write-requirements"],
                   capture_output=True, text=True, cwd=req_dir)
assert r.returncode == 0, "requirements detection FAILED:\n" + r.stdout[-1500:] + r.stderr[-1500:]
print("detected requirements:")
for fn in ("requirements.original.txt", "requirements.txt"):
    p = os.path.join(req_dir, fn)
    assert os.path.exists(p), "converter did not write %s" % fn
    lines = [l.strip() for l in open(p) if l.strip() and not l.lstrip().startswith("#")]
    print("  %-26s %s" % (fn + ":", ", ".join(lines)))
    assert any(l.split("==")[0].split()[0] == "catboost" for l in lines), "catboost missing from %s" % fn
    assert any(l.split("==")[0].split()[0] == "torch" for l in lines), "torch missing from %s" % fn

CHILD = textwrap.dedent('''
    import os, sys, numpy as np, tempfile

    def main():
        sys.path.insert(0, os.path.dirname(%r))
        import _main as m
        m.EXTRACT_BACKEND = "serial"      # no loky/spawn from inside this child process
        m.NULLMODEL_STEPS = 30            # keeps the synthetic null-model training fast
        print("registry:", len(m.REGISTRY), "features:", m.N_FEATURES, "active:", m.ACTIVE_FAMILIES)
        names = [f.name for f in m.REGISTRY]
        assert len(m.REGISTRY) == 18, names
        assert m.N_FEATURES == 147
        assert m.ACTIVE_FAMILIES is None
        assert names == ["z2", "e2", "tail", "time", "lmom", "wass", "energy", "mmd", "entropy",
                         "bocpd", "garch", "argarch", "tgarch", "recres", "hmm",
                         "volser", "dbocpd", "nullmodel"], names
        assert names.count("bocpd") == 1 and "glr" not in names, names
        assert names.count("volser") == 1, names
        _w = {f.name: len(f.names()) for f in m.REGISTRY}
        _want = {"garch": 9, "argarch": 11, "tgarch": 11, "recres": 11, "hmm": 11, "lmom": 15}
        assert {k: _w[k] for k in _want} == _want, _w
        assert all(f.cost == "O1" for f in m.REGISTRY if f.name in ("recres", "hmm")), \
            "recres / hmm must be cost O1"
        _vs = [f for f in m.REGISTRY if f.name == "volser"][0]
        assert len(_vs.names()) == 14, _vs.names()
        assert _vs.cost == "O1", _vs.cost
        assert all(n.startswith("volser_") for n in _vs.names()), _vs.names()
        _db, _nm = m.REGISTRY[-2], m.REGISTRY[-1]
        assert _db.name == "dbocpd" and len(_db.names()) == 5 and _db.cost == "O1", (_db.name, _db.names())
        assert _nm.name == "nullmodel" and len(_nm.names()) == 17 and _nm.cost == "O1", (_nm.name, _nm.names())
        assert all(n.startswith("nm_") for n in _nm.names()), _nm.names()
        # pruned from the bank after CV -- defined in the notebook, never registered
        for _f in ("ftest", "bartlett", "levene", "sigtuk", "gjr", "wavvar", "ordpat", "evt"):
            assert _f not in names, ("%%s must not be registered" %% _f, names)
        _bad = [n for n in m.FEATURES if n.startswith("lmom_lscale_e_")]
        assert not _bad, ("lmom must not emit lscale on the e channel", _bad)
        for _n in ("lmom_lscale_z_w32", "lmom_lscale_z_w64", "lmom_lscale_z_w256"):
            assert _n in m.FEATURES, ("%%s missing from FEATURES" %% _n)

        rng = np.random.default_rng(0)
        def mk(i):
            nh, T = int(rng.integers(300, 900)), int(rng.integers(40, 160))
            xh = rng.standard_t(5, nh)
            tau = int(rng.integers(5, T)) if i %% 2 else None
            xo = rng.standard_t(5, T)
            if tau is not None:
                xo[tau:] *= 1.8
            return (i, xh.astype(np.float64), xo.astype(np.float64), tau)
        data = [mk(i) for i in range(60)]

        m.TRAIN_MAX_SERIES = 60; m.NUM_ROUNDS = 40; m.VALID_FRACTION = 0.3
        assert any(len(d[2]) <= 64 for d in data[:10]) and any(len(d[2]) > 64 for d in data[:10]), \
            "the inferred series must cover both regimes (t < 64 ranker, t >= 64 blend)"
        md = tempfile.mkdtemp()
        m.train(data, md)
        print("trained ->", os.listdir(md))

        def stream(n):
            gen = m.infer([(d[1], iter(d[2])) for d in data[:n]], md)
            next(gen)
            return [float(v) for v in gen]
        out1 = stream(10)
        assert len(out1) == sum(len(d[2]) for d in data[:10]), (len(out1),)
        assert all(0.0 <= v <= 1.0 for v in out1)
        out2 = stream(10)
        d = max(abs(a - b) for a, b in zip(out1, out2))
        print("infer rows:", len(out1), "| determinism max diff: %%.2e" %% d)
        assert d < 1e-8
        import joblib
        art = joblib.load(os.path.join(md, "model.joblib"))
        assert len(art["features"]) == 147 and art["active_families"] is None
        assert art["features"] == [nm for f in m.REGISTRY for nm in f.names()], art["features"][:5]
        assert art["features"] == m.FEATURES
        assert {"dbocpd", "nullmodel"} <= set(art["family_state"]), sorted(art["family_state"])
        assert len(art["rankers"]) == len(art["ranker_iters"]) == 3, (len(art["rankers"]), art["ranker_iters"])
        assert art["switch_t"] == 64, art["switch_t"]
        # the switch: ranker before step 64, blend from 64 on (recomputed straight from the artifact)
        m.apply_family_state(art["family_state"])
        ref = []
        for d in data[:10]:
            fx = m.StreamFeatures(d[1])
            F = np.array([fx.update(v) for v in d[2]], dtype=np.float64)
            r = np.mean([b.predict(F, num_iteration=it, raw_score=True) for b, it in zip(art["rankers"], art["ranker_iters"])], axis=0)
            pl = np.mean([b.predict(F, num_iteration=it) for b, it in zip(art["boosters"], art["best_iterations"])], axis=0)
            pb = (1.0 - art["cat_weight"]) * pl + art["cat_weight"] * art["catboost"].predict_proba(F)[:, 1]
            for t in range(len(d[2])):
                if t < 64:
                    ref.append(1.0 / (1.0 + np.exp(-r[t])))
                else:
                    s = t - t %% m._stride_for(t)                 # infer() holds the blend between strides
                    ref.append(pb[max(s, 64)])
        dsw = max(abs(a - b) for a, b in zip(out1, ref))
        print("switch check: infer() vs direct recomputation, max diff %%.2e" %% dsw)
        assert dsw < 1e-9, dsw
        from catboost import CatBoostClassifier
        assert isinstance(art.get("catboost"), CatBoostClassifier), type(art.get("catboost"))
        assert art["cat_weight"] == 0.3, art["cat_weight"]
        assert isinstance(art["cat_iters"], (int, np.integer)) and art["cat_iters"] > 0, art["cat_iters"]
        assert len(art["boosters"]) == 3 and len(art["best_iterations"]) == 3
        print("blend artifact: 3 lgb boosters + catboost | cat_weight %%.2f | cat_iters %%d"
              %% (art["cat_weight"], art["cat_iters"]))
        print("ALL CHECKS PASSED")

    if __name__ == "__main__":      # REQUIRED: spawn-based workers re-import this file
        main()
''') % OUT

print("== 2..5. fresh-process import + train + infer")
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(CHILD); child = f.name
r = subprocess.run([sys.executable, child], capture_output=True, text=True, cwd=tempfile.mkdtemp())
print(r.stdout[-4000:]); print(r.stderr[-4000:])
sys.exit(r.returncode)