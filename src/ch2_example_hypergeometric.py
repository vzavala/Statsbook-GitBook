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

# %% [markdown]
# **Note:** This notebook reproduces Example 2.6 (Section 2.1.3) of *Statistics for Chemical Engineers*: inspecting the machine of Example 2.2 at 5 or 10 weeks chosen at random.

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

# %% [markdown]
# **Note:** As in the book, observing a failure in all 10 inspections has probability 8e-4: it is highly unlikely.
