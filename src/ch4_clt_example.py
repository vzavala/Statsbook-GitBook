# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Verifying the central limit theorem (Victor Z, UW-Madison, 2024).
# $X$ is Weibull with scale $\beta=2$ and shape $\xi=1$ (as in the slide figure).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $S$ | `S` | sample size of each average |
# | $\bar{X}=\hat{\mathbb{E}}^S_X$ | `x_bar` | sample averages (one per repetition) |
# | $e=\mathbb{E}[X]$, $v^2=\mathbb{V}[X]$ | `e`, `v2` | true mean and variance |
# | $\mathcal{N}(e, v^2/S)$ | `norm(e, sqrt(v2/S))` | CLT limit |
#
# `R` is the number of repetitions used to draw the histogram of $\bar{X}$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, norm

xi, beta = 1, 2
X = weibull_min(xi, scale=beta)
e, v2 = X.mean(), X.var()
R = 1000

np.random.seed(0)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
x = np.linspace(0, 10, 400)
axs[0, 0].plot(x, X.pdf(x), 'black', linewidth=1.5)
axs[0, 0].set_xlabel('$x$')
axs[0, 0].set_ylabel('$f(x)$')
z = np.linspace(0, 4, 400)
for ax, S in zip((axs[0, 1], axs[1, 0], axs[1, 1]), (10, 100, 1000)):
    x_bar = X.rvs(size=(R, S)).mean(axis=1)               # R independent sample averages
    ax.hist(x_bar, bins=30, density=True, edgecolor='black', facecolor='none')
    ax.plot(z, norm.pdf(z, e, np.sqrt(v2 / S)), 'k', linewidth=1.5)  # CLT: N(e, v^2/S)
    ax.set_xlim(0, 4)
    ax.set_title(f'$S={S}$')
    ax.set_xlabel(r'$\bar{x}$')
    ax.set_ylabel(r'$f(\bar{x})$')
    print(f"S = {S:4d}:  mean(x_bar) = {x_bar.mean():.3f} (e = {e:.3f}),  "
          f"S * var(x_bar) = {S * x_bar.var():.3f} (v^2 = {v2:.3f})")
for ax in axs.ravel():
    ax.grid(True)
plt.tight_layout()
plt.show()
