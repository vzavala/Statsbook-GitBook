# %% [markdown]
# Risk measures as norms (written for the book to accompany the slide example). For
# $\mathbf{y}=(100,70,50,20,10)$ the slide's $\mathrm{CVaR}$ at $\alpha=k/S$ is the average of the $k$ largest
# entries: $k=S$ gives the mean ($\ell_1$-like) and $k=1$ the worst case ($\ell_\infty$-like).
# Python names: `y`, `S`, `CVaR(y, k)`.

# %%
import numpy as np

y = np.array([100, 70, 50, 20, 10])
S = len(y)

def CVaR(y, k):
    """Average of the k largest entries of y."""
    return np.sort(y)[::-1][:k].mean()

print(f"mean = {y.mean():.0f},  worst case = {y.max():.0f}")
for k in range(S, 0, -1):
    print(f"alpha = {k}/{S}:  CVaR = {CVaR(y, k):.1f}")
