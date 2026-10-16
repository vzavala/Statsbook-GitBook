# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Inspecting the machine at random (Victor Z and David Cole, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $N_t$ | `N_t` | weeks in the record (population) |
# | $N_y$ | `N_y` | weeks in failure mode |
# | $p=N_y/N_t$ | `p` | fraction of failure weeks |
# | $n$ | `n` | number of inspections |
# | $f(x)=\mathbb{P}(X=x)$ | `f(x, n)` | hypergeometric pdf |

# %%
import numpy as np
from scipy.stats import hypergeom

N_t = 156   # weeks in 3 years
N_y = 79    # weeks the machine was in failure mode
p = N_y / N_t

def f(x, n):
    """Hypergeometric pdf f(x) for n inspections (scipy order: N_t, N_y, n)."""
    return hypergeom.pmf(x, N_t, N_y, n)

print(f"p = N_y/N_t = {p:.3f}")

# %% [markdown]
# Probability that the machine is in failure mode in all $n$ inspections,
# $\mathbb{P}(X=n)=f(n)$, for $n=5$ and $n=10$ (and, for contrast, in none of them).

# %%
for n in (5, 10):
    print(f"n = {n:2d}:  P(X={n}) = f({n}) = {f(n, n):.4g},   P(X=0) = f(0) = {f(0, n):.4g}")
