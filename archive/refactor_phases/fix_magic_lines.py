"""PROBLEM 1 -- stop crunch-convert from commenting out `%`-continuation lines.

crunch-convert treats any code line whose first non-blank character is `%` or `!` as an
IPython magic and rewrites it to `pass  #<line>`. A printf-style continuation like

        raise AssertionError("family %s returned %d values, expected %d"
                             % (f.name, len(f._last), nout))

therefore loses its second line, the call above it is never closed, and the whole
conversion aborts with a SyntaxError before anything runs.

Fix: join each such continuation onto the line above it. The format string and its
arguments are untouched, so every message is byte-for-byte what it was.

The `%pip install` / `!crunch setup-notebook` lines in the Setup cell are real magics
and are left alone.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbedit

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "new_structural.ipynb")

MAGIC_RE = re.compile(r"^\s*[%!]")
# the genuine magics, which must survive untouched
REAL_MAGICS = ("%pip install", "!crunch setup-notebook")


def is_real_magic(line):
    s = line.strip()
    return any(s.startswith(m) for m in REAL_MAGICS)


def fix_cell(text):
    """Join every `%`/`!`-leading continuation onto the preceding line."""
    lines = text.split("\n")
    n_fixed = 0
    i = 1
    while i < len(lines):
        line = lines[i]
        if MAGIC_RE.match(line) and not is_real_magic(line):
            prev = lines[i - 1]
            if not prev.strip():
                raise AssertionError(
                    "line %d starts with %r but the line above is blank -- this is a real "
                    "magic, not a continuation: %r" % (i + 1, line.strip()[0], line))
            lines[i - 1] = prev.rstrip() + " " + line.strip()
            del lines[i]
            n_fixed += 1
            # do not advance: the joined line may itself now be followed by another
            # `%` continuation
            continue
        i += 1
    return "\n".join(lines), n_fixed


def main():
    nb = nbedit.load(NB)
    total = 0
    report = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = nbedit.src(cell)
        new, n = fix_cell(text)
        if n:
            # the cell must still parse after the join
            try:
                compile(new, "<cell %d>" % idx, "exec")
            except SyntaxError as exc:
                raise AssertionError("cell %d no longer parses after joining: %s"
                                     % (idx, exc)) from exc
            nbedit.set_src(cell, new)
            report.append((idx, n))
            total += n

    nbedit.save(nb, NB)
    print("joined %d continuation line(s)" % total)
    for idx, n in report:
        print("   cell %-3d %d line(s)" % (idx, n))

    # nothing but the two real magics may match now
    left = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for ln, line in enumerate(nbedit.src(cell).split("\n"), 1):
            if MAGIC_RE.match(line):
                left.append((idx, ln, line.strip()))
    print("\nremaining magic-looking lines: {}".format(len(left)))
    for idx, ln, line in left:
        ok = is_real_magic(line)
        print("   cell %-3d line %-4d %-9s %s" % (idx, ln, "(real)" if ok else "UNEXPECTED",
                                                  line[:80]))
        assert ok, "unexpected magic-looking line survived"


if __name__ == "__main__":
    main()
