"""Re-apply every phase, in order, on top of a post-Phase-1 notebook.

The phase scripts are one-shot transformations, so the way to change one is to restore
the base and replay the chain rather than patch the current file.

    python tools/rebuild_all.py <base.ipynb>

<base.ipynb> must be the notebook as it stood after phase1_prune.py + phase1b_docs.py
(that is what make_reference_impl.py snapshotted, so it is also what the Phase 2/3
equivalence tests diff against).
"""

import json
import os
import shutil
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(os.path.dirname(HERE), "new-structural (5).ipynb")

PHASES = [
    "phase2_fast.py",
    "phase2b_bench.py",
    "phase3_metrics.py",
    "phase4_cache.py",
    "phase5_research.py",
    "phase6_backend.py",
]


def ensure_cell_ids(path=NB):
    with open(path) as fh:
        nb = json.load(fh)
    seen = {c.get("id") for c in nb["cells"] if c.get("id")}
    n = 0
    for c in nb["cells"]:
        if not c.get("id"):
            while True:
                i = uuid.uuid4().hex[:8]
                if i not in seen:
                    break
            seen.add(i)
            c["id"] = i
            n += 1
    if n:
        with open(path, "w") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    print("cell ids: added %d" % n)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    base = sys.argv[1]
    shutil.copy(base, NB)
    print("base restored from %s\n" % base)
    for p in PHASES:
        print("--- %s" % p)
        subprocess.run([sys.executable, os.path.join(HERE, p)], check=True)
    ensure_cell_ids()
    print("\nrebuild complete")


if __name__ == "__main__":
    main()
