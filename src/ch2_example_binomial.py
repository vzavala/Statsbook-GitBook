# %% [markdown]
# Failures in a sequence of $n=10$ weeks (Victor Z and J E Umaña, UW-Madison, 2024).
# The weekly status is $\mathcal{B}(p)$ with $p=0.5$, so the number of failures in
# $n$ weeks is $X\sim\mathrm{Bi}(n,p)$ with pdf $f(x)=\binom{n}{x}p^x(1-p)^{n-x}$.
# Python names: `n`, `p`, pdf `f(x)`, `E_X`.

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
