# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Verifying the extreme value theorem (Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $X\sim\mathcal{N}(2,1)$ | `mu`, `sigma` | underlying RV |
# | $X^S_{max}=\max\{X_1,\dots,X_S\}$ | `x_max` | sample maxima (one per repetition) |
# | $\mathrm{GEV}(a,b,c)$ | `a`, `b`, `c` | location, scale, shape |
#
# `scipy.stats.genextreme` uses the opposite sign for the shape: `c_scipy = -c`.
# `R` is the number of repetitions.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, genextreme

mu, sigma = 2, 1
R = 1000
np.random.seed(0)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
x = np.linspace(-3, 7, 400)
axs[0, 0].plot(x, norm.pdf(x, mu, sigma), 'black', linewidth=1.5)
axs[0, 0].set_xlabel('$x$')
axs[0, 0].set_ylabel('$f(x)$')
for ax, S in zip((axs[0, 1], axs[1, 0], axs[1, 1]), (10, 100, 1000)):
    x_max = np.random.normal(mu, sigma, (R, S)).max(axis=1)
    ax.hist(x_max, bins=30, density=True, edgecolor='black', facecolor='none')
    c_scipy, a, b = genextreme.fit(x_max)
    ax.plot(x, genextreme.pdf(x, c_scipy, loc=a, scale=b), 'k', linewidth=1.5)
    ax.set_xlim(2, 7)
    ax.set_title(f'$S={S}$:  a={a:.2f}, b={b:.2f}, c={-c_scipy:.2f}')
    ax.set_xlabel(r'$x_{max}$')
    ax.set_ylabel(r'$f(x_{max})$')
for ax in axs.ravel():
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Probabilities of rare events from the GEV fit for $S=1000$.

# %%
for x_k in (5, 7):
    print(f"P(X_max > {x_k}) = {genextreme.sf(x_k, c_scipy, loc=a, scale=b):.3g}")
