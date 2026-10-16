# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# The logistic function $m(x,\theta)=1/(1+e^{-\theta^Tx})$ for a scalar input and
# $\theta=-1,0,+1$ (written for the book to accompany the slide figure). The evidence is $\theta x$.

# %%
import numpy as np
import matplotlib.pyplot as plt

def m(x, theta):
    return 1 / (1 + np.exp(-theta * x))

x = np.linspace(-10, 10, 400)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, theta in zip(axs, (-1, 0, 1)):
    ax.plot(x, m(x, theta), 'k', linewidth=1.5)
    ax.set_title(rf'$\theta={theta}$')
    ax.set_xlabel('$x$'); ax.set_ylabel(r'$m(x,\theta)$')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
plt.tight_layout()
plt.show()
