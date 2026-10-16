"""One-time import of the original notebooks (../Python) into editable sources.

Each notebook becomes a "percent-format" Python file in src/ :

    # %% [markdown]
    # Some *markdown* text
    # %%
    x = 1                      # a code cell

The Google-Colab boilerplate (drive mount, path checks, sys.path tweaks) is
removed and replaced by a single portable setup cell. Data/ and Support/ are
copied next to the generated notebooks so they run from notebooks/.

Usage:  python3 tools/import_notebooks.py [--force]
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT.parent / "Python"
SRC = ROOT / "src"
NB = ROOT / "notebooks"

BOILERPLATE = [
    r"from google\.colab import drive",
    r"drive\.mount\(",
    r"^\s*#+\s*Start (here|from here)",
    r"os\.path\.isdir\(path\)",
    r"sys\.path\.append\(path",
    r"^\s*# Specify the path of this file",
]
# a cell that only sets the Colab path (other cells just lose that line)
PATH_ONLY = r"^\s*(#[^\n]*\n)?\s*path\s*=\s*'/content/drive[^']*'\s*(\n\s*print\(path\))?\s*$"

SETUP = """# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''
"""


def is_boilerplate(src: str) -> bool:
    s = src.strip()
    if not s or s == "---":
        return True
    return bool(re.fullmatch(PATH_ONLY, s)) or any(re.search(p, s, re.M) for p in BOILERPLATE)


def clean_code(src: str) -> str:
    # inline re-definitions of the Colab path inside code cells
    src = re.sub(r"^\s*path\s*=\s*'/content/drive[^']*'\s*\n", "", src, flags=re.M)
    src = src.replace("path+'/Data/", "path+'Data/")
    return src.rstrip()


def to_percent(nb: dict) -> str:
    out = [SETUP]
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if is_boilerplate(src):
            continue
        if cell["cell_type"] == "markdown":
            body = "\n".join("# " + l if l else "#" for l in src.rstrip().splitlines())
            out.append("# %% [markdown]\n" + body + "\n")
        elif cell["cell_type"] == "code":
            out.append("# %%\n" + clean_code(src) + "\n")
    return "\n".join(out)


def main():
    force = "--force" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    SRC.mkdir(exist_ok=True)
    NB.mkdir(exist_ok=True)
    for f in sorted(ORIG.glob("*.ipynb")):
        dst = SRC / (f.stem + ".py")
        if only and f.stem not in only:
            continue
        if dst.exists() and not (force or only):
            continue
        dst.write_text(to_percent(json.loads(f.read_text())))
        print("imported", f.name)
    for d in ("Data", "Support"):
        if not (NB / d).exists():
            shutil.copytree(ORIG / d, NB / d, ignore=shutil.ignore_patterns("__pycache__"))


if __name__ == "__main__":
    main()
