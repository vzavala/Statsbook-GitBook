# %% [markdown]
# Failures in a sequence of $n=10$ weeks (Victor Z and J E Umaña, UW-Madison, 2024).
# The weekly status is $\mathcal{B}(p)$ with $p=0.5$, so the number of failures in
# $n$ weeks is $X\sim\mathrm{Bi}(n,p)$ with pdf $f(x)=\binom{n}{x}p^x(1-p)^{n-x}$.
# Python names: `n`, `p`, pdf `f(x)`, `E_X`.

# %% [markdown]
# **Note:** This notebook reproduces Example 2.8 (Section 2.1.4) of *Statistics for Chemical Engineers*: number of failures in a sequence of n weeks, with p = 0.5.

# %%
from scipy.stats import binom

n, p = 10, 0.5

def f(x):
    """Binomial pdf f(x) = C(n, x) p^x (1-p)^(n-x)."""
    return binom.pmf(x, n, p)

print(f"P(X=2)  = f(2)  = {f(2):.4f}")
print(f"P(X=10) = f(10) = {f(10):.3g}")
E_X = n * p
print(f"E_X = n p = {E_X:g}")

# %% [markdown]
# **Note:** As in the book, the machine is equally likely to be in failure or normal mode, so it is unlikely to be in either mode all n times: P(X = 0) = P(X = 10) = 9.76e-4.
