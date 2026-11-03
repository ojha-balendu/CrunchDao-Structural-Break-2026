"""
build_submission.py -- turn the working research notebook into a clean, converter-safe
submission notebook.

  python tools/build_submission.py new_structural.ipynb submission.ipynb

What it does:
  1. Drops the research/experiment cells (diagnostic run, KS scan, Part-2 harness,
     Optuna, feature trials) -- they are kept in research.ipynb instead.
  2. Wraps every module-level constant / REGISTRY statement that train() or infer()
     needs in `# @crunch/keep:on` ... `# @crunch/keep:off`, so Crunch's notebook
     converter does not comment them out.
  3. Leaves imports, defs and classes untouched (the converter keeps those anyway).

Ranges are matched by the FIRST LINE TEXT of each statement, not by line number, so
small edits above them don't break this script. It fails loudly if a marker is missing.
"""
import json
import sys

# (cell index, first line of the block, last line of the block) -- both matched literally
KEEP_BLOCKS = [
    (10, "WINDOWS = (32, 64, 256)", "REF_MAX = 256"),
    (10, 'CH = ("z", "z2", "e", "e2")', "INV_SQRT2 = 0.7071067811865476"),
    (12, "CH_IDX = {name: i for i, name in enumerate(CH)}", "CH_IDX = {name: i for i, name in enumerate(CH)}"),
    (12, "REGISTRY: List[FeatureFamily] = []", "REGISTRY: List[FeatureFamily] = []"),
    (14, 'BANK_CHANNELS = ("z2", "e2")', "REGISTRY.append(TimeFamily())"),
    (17, "_COEF_CACHE = {}", "_COEF_CACHE = {}"),
    (17, "REGISTRY.extend([", "])"),
    (19, "REGISTRY.extend([", "])"),
    (21, "REGISTRY.extend([", "])"),
    (23, "REGISTRY.extend([", "])"),
    (23, "FEATURES = [nm for f in REGISTRY for nm in f.names()]", "N_FEATURES = len(FEATURES)"),
    (23, "ACTIVE_FAMILIES = resolve_active_families(None)", "ACTIVE_FAMILIES = resolve_active_families(None)"),
    (24, "_LOG_2PI = math.log(2.0 * math.pi)", "_LOG_2PI = math.log(2.0 * math.pi)"),
    (27, "KEEP_EVERY = ((24, 1), (128, 2), (512, 4), (10 ** 9, 8))", "KEEP_EVERY = ((24, 1), (128, 2), (512, 4), (10 ** 9, 8))"),
    (27, "_USE_GLOBAL_ACTIVE = object()", "_USE_GLOBAL_ACTIVE = object()"),
    (27, 'EXTRACT_BACKEND = "auto"', "FORK_CHUNKSIZE = 8"),
    (29, 'CACHE_DIR = "cache"', 'CACHE_DIR = "cache"'),
    (29, "FEATURE_TO_FAMILY = feature_to_family()", "FEATURE_TO_FAMILY = feature_to_family()"),
    (43, "FAST_LGB = dict(", ")"),
    (43, "FINAL_LGB = dict(", ")"),
    (45, "LGB_PARAMS = dict(", "TRAIN_MAX_SERIES = None"),
    (47, "SMOOTH_ALPHA = 0.0", "RUNNING_MAX = False"),
]

# research / one-off cells: dropped entirely (they live in research.ipynb)
DROP_CELLS = [34, 35, 36, 41] + list(range(55, 85))
# single lines to drop (a call left inside an otherwise-needed cell)
DROP_LINES = [(33, "bench_extractor(train_data)")]


def _find(lines, text, occurrence=1, start=0):
    seen = 0
    for i in range(start, len(lines)):
        if lines[i].strip().startswith(text.strip()):
            seen += 1
            if seen == occurrence:
                return i
    raise SystemExit("NOT FOUND: %r" % text)


def main(src, dst):
    nb = json.load(open(src))
    cells = nb["cells"]

    for ci, text in DROP_LINES:
        lines = "".join(cells[ci]["source"]).split("\n")
        i = _find(lines, text)
        lines[i] = "# " + lines[i] + "    # (research only)"
        cells[ci]["source"] = "\n".join(lines)

    # the hold-out dump I added for research hardcodes a Kaggle path -> make it portable
    lines = "".join(cells[45]["source"]).split("\n")
    for i, l in enumerate(lines):
        if '/kaggle/working/holdout_scores.npz' in l:
            lines[i] = l.replace('"/kaggle/working/holdout_scores.npz"',
                                 'os.path.join(model_directory_path, "holdout_scores.npz")')
            lines[i] = lines[i].replace('-> /kaggle/working/holdout_scores.npz', '-> model dir')
    cells[45]["source"] = "\n".join(lines)

    by_cell = {}
    for block in KEEP_BLOCKS:
        ci, first, last = block[0], block[1], block[2]
        occ = block[3] if len(block) > 3 else 1
        by_cell.setdefault(ci, []).append((first, last, occ))

    for ci, blocks in by_cell.items():
        lines = "".join(cells[ci]["source"]).split("\n")
        spans = []
        for first, last, occ in blocks:
            a = _find(lines, first, occ)
            b = a if last == first else _find(lines, last, 1, a)
            spans.append((a, b))
        for a, b in sorted(spans, reverse=True):          # bottom-up so indices hold
            if "@crunch/keep:on" in lines[max(a - 1, 0)]:
                continue                                   # already wrapped
            lines.insert(b + 1, "# @crunch/keep:off")
            lines.insert(a, "# @crunch/keep:on")
        cells[ci]["source"] = "\n".join(lines)

    nb["cells"] = [c for i, c in enumerate(cells) if i not in set(DROP_CELLS)]
    for c in nb["cells"]:
        c["outputs"] = [] if c["cell_type"] == "code" else c.get("outputs", [])
        if c["cell_type"] == "code":
            c["execution_count"] = None
    json.dump(nb, open(dst, "w"), indent=1)
    print("wrote %s: %d cells (was %d)" % (dst, len(nb["cells"]), len(cells)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "new_structural.ipynb",
         sys.argv[2] if len(sys.argv) > 2 else "submission.ipynb")
