# How to use this book

## Reading

Each page follows the slides. After an example on the slides, the **Python implementation** that
follows starts with a table of symbols and their Python names, then alternates equations and code.
The [Nomenclature](nomenclature.md) page collects all of these tables and the general naming rules
(for example, $$\hat{\theta}$$ is `theta_hat`, $$x_\omega$$ is `x_omega`, and $$\lambda$$ is `lam`).

## Running the notebooks

### In Google Colab

1. Click the **Open in Colab** link at the top of a Python section.
2. Choose **Runtime → Run all** (or run the cells one by one with the ▶ button or **Shift+Enter**).
   If Colab warns that the notebook was not authored by Google, click **Run anyway**.
3. Notebooks that read data start with a **Setup** cell. On Colab it downloads the `Data/` and
   `Support/` folders from the book's GitHub repository automatically (a few seconds); nothing
   else is needed.

To keep your own edits, use **File → Save a copy in Drive**.

### On your computer

```bash
pip install -r requirements.txt
```

```bash
cd notebooks && jupyter lab
```

The notebooks are set up to run from the `notebooks/` folder, where they find `Data/` and `Support/`. The CNN notebooks of
Chapter 6 also need PyTorch (`torch`).


