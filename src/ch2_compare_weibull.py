# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Weibull RVs $X\sim\mathrm{Weibull}(\xi,\beta)$ (Victor Z and David Cole, UW-Madison, 2024).
# `scipy.stats.weibull_min` takes the **shape** $\xi$ first and the **scale** $\beta$ as `scale`.
# The dashed line marks $F(\beta)=1-e^{-1}=0.63$ for any $\xi$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

beta = [1, 1, 2, 2]     # scale
xi = [1, 2, 1, 2]       # shape
lw = [0.5, 1, 1.5, 2]
x = np.linspace(0, 4, 1000)

plt.figure(figsize=(10, 4))
for sub, fun, label in ((1, weibull_min.pdf, '$f(x)$'), (2, weibull_min.cdf, '$F(x)$')):
    plt.subplot(1, 2, sub)
    for k in range(len(beta)):
        plt.plot(x, fun(x, xi[k], scale=beta[k]), color='black', linewidth=lw[k],
                 label=rf'$\beta={beta[k]},\ \xi={xi[k]}$')
    plt.grid(True)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(label, fontsize=14)
plt.axhline(1 - np.exp(-1), color='k', ls='--')
plt.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.show()
