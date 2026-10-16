# How to use this book

## Reading

Each page follows the slides. After an example on the slides, the **Python implementation** that
follows starts with a table of symbols and their Python names, then alternates equations and code.
The [Nomenclature](nomenclature.md) page collects all of these tables and the general naming rules
(for example, $$\hat{\theta}$$ is `theta_hat`, $$x_\omega$$ is `x_omega`, and $$\lambda$$ is `lam`).

## Running the notebooks

### In Google Colab

Use the **Open in Colab** link at the top of each Python section. Notebooks that read data (from
`Data/`) or helper code (from `Support/`) need those folders: copy the repository's `notebooks/`
folder to your Google Drive, mount the drive in the notebook, and set `path` to the folder, for example

```python
from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/StatsBook/notebooks/'
```

### On your computer

```bash
pip install -r requirements.txt
```

```bash
cd notebooks && jupyter lab
```

The notebooks are set up to run from the `notebooks/` folder (`path = ''`). The CNN notebooks of
Chapter 6 also need PyTorch (`torch`).

## For maintainers: how the book is built

The pages are generated; edit the sources, not the generated files.

| What | Where | Edit? |
|---|---|---|
| Slide content | `../Slides/Slides-New/*.tex` | yes (source of the concept text) |
| Python examples | `src/*.py` (Jupyter "percent" format) | yes (source of the notebooks) |
| Book layout (chapters, page splits, notebook placement) | `tools/book_config.py` | yes |
| Notebooks with outputs | `notebooks/*.ipynb` | generated |
| Pages and table of contents | `chapters/**.md`, `SUMMARY.md`, `nomenclature.md` | generated |
| Figures | `.gitbook/assets/` | generated |

In `src/*.py`, cells start with `# %%` (code) or `# %% [markdown]`. `# %% [hide]` cells run in the notebook
but are not shown in the book, and `# %% [part] Title` splits a notebook so that its parts follow
successive slide examples (`# %% [part] Title @ Frame title #k` places a part after the k-th group of
frames with that title).

To rebuild after editing:

```bash
python tools/build_notebooks.py
```

```bash
python3 tools/build_book.py
```

`build_notebooks.py` executes the notebooks whose sources changed (pass names to rebuild specific ones,
or `--all`). `build_book.py` converts the slides, inserts the notebooks, converts the slide figures to PNG
(it needs Ghostscript), and rewrites `SUMMARY.md` and `nomenclature.md`. `python tools/preview.py <folder>`
writes an approximate HTML preview of the pages.

`REVIEW.md` lists issues found in the slides and in the original notebooks while aligning them.
