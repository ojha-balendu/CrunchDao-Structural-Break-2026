"""Small helpers for editing the notebook JSON in place.

The notebook is the deliverable (crunch submits it), so every structural change
is scripted here rather than hand-edited, which keeps the edits reviewable and
repeatable.
"""

import json
import os

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "new-structural (5).ipynb")


def load(nb_path=NB_PATH):
    with open(nb_path) as fh:
        return json.load(fh)


def save(nb, nb_path=NB_PATH):
    with open(nb_path, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    """Store `text` back as a notebook source list (line-terminated, like nbformat)."""
    lines = text.split("\n")
    out = [l + "\n" for l in lines[:-1]]
    if lines[-1] != "":
        out.append(lines[-1])
    cell["source"] = out
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def find_cell(nb, needle, cell_type=None, start=0):
    """Index of the first cell at/after `start` containing `needle`."""
    for i in range(start, len(nb["cells"])):
        c = nb["cells"][i]
        if cell_type and c["cell_type"] != cell_type:
            continue
        if needle in src(c):
            return i
    raise KeyError("no cell containing %r" % needle)


def new_code_cell(text):
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
    set_src(c, text)
    return c


def new_markdown_cell(text):
    c = {"cell_type": "markdown", "metadata": {}, "source": []}
    set_src(c, text)
    return c


# ---------------------------------------------------------------------------
#  Top-level block extraction (classes / functions), by source scanning
# ---------------------------------------------------------------------------
def extract_block(text, header_prefix, allow_multiple=False):
    """Return (block_text, start_line, end_line) for the top-level `class X`/`def X`
    block whose first line starts with `header_prefix`.

    The block runs from its header to the last line before the next line that is
    non-blank, non-comment and starts at column 0. Trailing blank lines are kept
    out of the block.

    Raises if the prefix matches more than one top-level header. That ambiguity is
    not theoretical: inserting a new definition and only THEN removing the old one
    leaves two, and this function would silently pick the new one and delete it,
    leaving the original in place. Remove first, insert second -- or be explicit with
    allow_multiple=True.
    """
    lines = text.split("\n")
    hits = [i for i, l in enumerate(lines) if l.startswith(header_prefix)]
    if not hits:
        raise KeyError("no top-level block starting with %r" % header_prefix)
    if len(hits) > 1 and not allow_multiple:
        raise KeyError(
            "%r matches %d top-level blocks (lines %s) -- ambiguous. Remove the old "
            "definition BEFORE inserting the new one, or pass allow_multiple=True."
            % (header_prefix, len(hits), [h + 1 for h in hits]))
    start = hits[0]

    end = len(lines)
    for j in range(start + 1, len(lines)):
        l = lines[j]
        if not l.strip():
            continue
        if not l[0].isspace():
            end = j
            break
    # trim trailing blank lines off the block itself
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end]), start, end


def replace_block(text, header_prefix, new_block):
    """Swap the named top-level block for `new_block`, keeping surrounding text."""
    _, start, end = extract_block(text, header_prefix)
    lines = text.split("\n")
    lines[start:end] = new_block.rstrip("\n").split("\n")
    return "\n".join(lines)


def insert_before(text, header_prefix, new_text):
    """Insert `new_text` immediately above the named top-level block."""
    _, start, _ = extract_block(text, header_prefix)
    lines = text.split("\n")
    lines[start:start] = new_text.rstrip("\n").split("\n") + ["", ""]
    return "\n".join(lines)


def remove_blocks(text, header_prefixes):
    """Delete each named top-level block; return (new_text, {prefix: block_text})."""
    grabbed = {}
    for hp in header_prefixes:
        block, start, end = extract_block(text, hp)
        grabbed[hp] = block
        lines = text.split("\n")
        # also swallow the blank separator lines that followed the block
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start:end]
        text = "\n".join(lines)
    return text, grabbed
