# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Pdf and cdf of a Poisson RV $X\sim\mathcal{P}(\lambda)$ for several $\lambda$
# (Victor Z and David Cole, UW-Madison, 2024).
# Python names: `lam` for $\lambda$ (`lambda` is reserved in Python), realizations `x_omega`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

lam_values = [1/2, 5, 10]
x = np.arange(0, 21)

np.random.seed(0)
S = 1000
x_omega = poisson.rvs(1/2, size=S)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, (1, 2))
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.grid(True)
plt.axis([0, S, 0, 10])

for sub, fun, label in ((3, poisson.pmf, '$f(x)$'), (4, poisson.cdf, '$F(x)$')):
    plt.subplot(2, 2, sub)
    for lam, mk in zip(lam_values, ["o-", "x-", "d-"]):
        plt.plot(x, fun(x, lam), mk, color='k', markerfacecolor='w', label=rf'$\lambda={lam}$')
    plt.grid(True)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(label, fontsize=14)
    plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
