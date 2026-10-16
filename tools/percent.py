"""Reader for the percent-format notebook sources in src/.

Cell headers (one per cell):

    # %%                      code cell
    # %% [markdown]           markdown cell (each line prefixed with "# ")
    # %% [hide]               code cell run in the notebook but not shown in the book
    # %% [markdown] [hide]    markdown cell shown in the notebook only
    # %% [part] Title         starts a new part; the book places part k after the
                              k-th group of slide frames that cite this notebook
"""
import re

HEADER = re.compile(r"^# %%(.*)$")


def parse(text):
    """Return a list of cells: dict(type, source, hide, part)."""
    cells, cur = [], None
    part = 0
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            flags = m.group(1)
            if cur:
                cells.append(cur)
            if "[part]" in flags:
                part += 1
                cur = None
                cells.append(dict(type="part", source=flags.split("[part]", 1)[1].strip(),
                                  hide=False, part=part))
                continue
            cur = dict(type="markdown" if "[markdown]" in flags else "code",
                       lines=[], hide="[hide]" in flags, part=part)
            continue
        if cur is None:
            continue
        cur["lines"].append(line)
    if cur:
        cells.append(cur)
    out = []
    for c in cells:
        if c["type"] == "part":
            out.append(c)
            continue
        lines = c.pop("lines")
        if c["type"] == "markdown":
            lines = [re.sub(r"^# ?", "", l) if l.startswith("#") else l for l in lines]
        src = "\n".join(lines).strip("\n")
        if src.strip():
            c["source"] = src
            out.append(c)
    return out
