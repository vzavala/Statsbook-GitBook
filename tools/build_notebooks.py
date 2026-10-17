"""Build executed Jupyter notebooks (notebooks/*.ipynb) from src/*.py.

Requires: nbformat, nbclient, ipykernel and the scientific stack.

Usage:
    python tools/build_notebooks.py                 # all sources that changed
    python tools/build_notebooks.py ch2_example_poisson ch1_gibbs_example
    python tools/build_notebooks.py --all --no-exec # rebuild without running

Built notebooks are also copied to ../Python and ../GitRepo/Python (the course's
Python folder and the StatsBook GitHub repository) when they exist; pass --no-export to skip that.
"""
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).parent))
import book_config as cfg  # noqa: E402
import percent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC, NB = ROOT / "src", ROOT / "notebooks"
# The course's Python folder (next to this repository) gets the same notebooks,
# so students see one version of the code everywhere.
PYTHON_FOLDERS = [ROOT.parent / "Python", ROOT.parent / "GitRepo" / "Python"]
TIMEOUT = 1800


def build(src: Path, execute=True):
    cells = []
    for c in percent.parse(src.read_text().replace("__GITHUB_REPO__", cfg.GITHUB_REPO)):
        if c["type"] == "part":
            if c["source"]:
                cells.append(nbformat.v4.new_markdown_cell("## " + c["source"].split(" @ ")[0],
                                                           metadata={"part": c["part"], "part_title": c["source"]}))
            continue
        meta = {"part": c["part"]}
        if c["hide"]:
            meta["tags"] = ["book-hide"]
        make = nbformat.v4.new_markdown_cell if c["type"] == "markdown" else nbformat.v4.new_code_cell
        cells.append(make(c["source"], metadata=meta))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}})
    if execute:
        NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3",
                       resources={"metadata": {"path": str(NB)}}).execute()
    nbformat.write(nb, NB / (src.stem + ".ipynb"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    execute = "--no-exec" not in sys.argv
    if args:
        srcs = [SRC / (a.removesuffix(".py") + ".py") for a in args]
    else:
        srcs = sorted(SRC.glob("*.py"))
        if "--all" not in sys.argv:
            srcs = [s for s in srcs if not (NB / (s.stem + ".ipynb")).exists()
                    or (NB / (s.stem + ".ipynb")).stat().st_mtime < s.stat().st_mtime]
    failed = []
    for s in srcs:
        t = time.time()
        try:
            build(s, execute)
            print(f"ok    {s.stem}  ({time.time() - t:.1f}s)")
        except Exception as e:  # keep going; report at the end
            failed.append(s.stem)
            msg = str(e).strip().splitlines()
            print(f"FAIL  {s.stem}: {msg[-1] if msg else e}")
    if failed:
        print("\nfailed:", " ".join(failed))
        sys.exit(1)
    if "--no-export" not in sys.argv:
        export([s.stem for s in srcs])


def export(names):
    """Copy built notebooks to the course's Python folder (Data/ and Support/ are shared)."""
    for folder in PYTHON_FOLDERS:
        if not folder.is_dir():
            continue
        for name in names:
            shutil.copy2(NB / f"{name}.ipynb", folder / f"{name}.ipynb")
        print(f"copied {len(names)} notebook(s) to {folder}")


if __name__ == "__main__":
    main()
