# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Pdf and cdf of a Bernoulli RV $X\sim\mathcal{B}(p)$ with $p=0.3$
# (Victor Z and J E Umaña, UW-Madison, 2024).
# Python names: `p`, domain `D`, pdf `f`, cdf `F`, realizations `x_omega`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

p = 0.3
D = np.array([0, 1])
f = np.array([1 - p, p])        # f(x) = p^1[x=1] (1-p)^1[x=0]

x = np.linspace(-1, 2, 1000)
F = np.array([np.sum(f[D <= x_k]) for x_k in x])

np.random.seed(0)
S = 100
x_omega = bernoulli.rvs(p, size=S)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, (1, 2))
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.grid(True)
plt.yticks([-0.5, 0, 0.5, 1, 1.5])
plt.axis([0, S, -0.5, 1.5])

plt.subplot(2, 2, 3)
plt.vlines(D, 0, f, colors='k', linestyles='--')
plt.scatter(D, f, marker='o', facecolors='w', edgecolors='k', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.axis([-0.5, 1.5, -0.02, 1.02])

plt.subplot(2, 2, 4)
plt.step(x, F, 'k-', where='post')
plt.scatter(D, np.cumsum(f), marker='o', facecolors='w', edgecolors='k', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$F(x)$', fontsize=14)
plt.axis([-0.5, 1.5, -0.02, 1.02])
plt.tight_layout()
plt.show()
