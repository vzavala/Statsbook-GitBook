# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Pdf and cdf of a continuous uniform RV $X\sim\mathcal{U}_c(a,b)$ with $a=-1,b=1$
# (Victor Z and David Cole, UW-Madison, 2024).
# Python names: `a`, `b`, pdf `f`, cdf `F`, realizations `x_omega`.
# In `scipy.stats.uniform`, `loc` $=a$ and `scale` $=b-a$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

a, b = -1, 1
x = np.linspace(-2, 2, 1000)
f = uniform.pdf(x, loc=a, scale=b - a)    # 1/(b-a) on [a, b]
F = uniform.cdf(x, loc=a, scale=b - a)    # (x-a)/(b-a) on [a, b]

np.random.seed(0)
S = 1000
x_omega = np.random.uniform(a, b, S)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, (1, 2))
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.grid(True)
plt.axis([0, S, -2, 2])
plt.subplot(2, 2, 3)
plt.plot(x, f, color='black', linewidth=1.5)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.subplot(2, 2, 4)
plt.plot(x, F, color='black', linewidth=1.5)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$F(x)$', fontsize=14)
plt.tight_layout()
plt.show()
