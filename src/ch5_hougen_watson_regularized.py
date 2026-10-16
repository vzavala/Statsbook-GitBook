# %% [markdown] [hide]
# **Setup.** Run this cell first: it locates the data files this notebook reads. On Google
# Colab it downloads them (the `Data/` and `Support/` folders) from the book's GitHub
# repository; on your computer, run the notebook from the `notebooks/` folder.

# %% [hide]
import os
import sys
path = ''
if 'google.colab' in sys.modules and not os.path.isdir('Data'):
    if not os.path.isdir('statsbook'):
        os.system('git clone -q --depth 1 --filter=blob:none --sparse '
                  'https://github.com/__GITHUB_REPO__.git statsbook')
        os.system('git -C statsbook sparse-checkout set notebooks/Data notebooks/Support')
    path = 'statsbook/notebooks/'

# %% [markdown]
# Ridge-regularized SSE for the Hougen-Watson model (Victor Z and Leo Gonzalez, UW-Madison, 2024):
#
# $$SSE(\theta)+\kappa\,\rho(\theta),\qquad \rho(\theta)=\frac{1}{2}\lVert\theta-\bar{\theta}\rVert_2^2,\quad \kappa=1,\ \bar{\theta}=0.$$
#
# Python names: `kappa`, `theta_bar`, `rho(theta)`, `SSE(theta)` (model `m` as in `ch5_hougen_watson`).

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(path+'Data/hougenwatson.dat')
X, y = data[:, :3], data[:, -1]

def m(theta, X):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (theta[0]*x2 - x3/theta[4]) / (1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3)

def SSE(theta):
    return 0.5 * np.sum((y - m(theta, X))**2)

kappa = 1.0
theta_bar = np.zeros(5)

def rho(theta):
    return 0.5 * np.sum((theta - theta_bar)**2)

theta_ref = np.array([1.0, 0.05, 0.02, 0.1, 2.0])
grid = np.linspace(-2, 2, 201)
maps = {'SSE': np.zeros((201, 201)), 'regularized': np.zeros((201, 201))}
for i, t2 in enumerate(grid):
    for j, t1 in enumerate(grid):
        th = theta_ref.copy()
        th[1], th[2] = t1, t2
        maps['SSE'][i, j] = SSE(th)
        maps['regularized'][i, j] = SSE(th) + kappa * rho(th)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, (title, Z) in zip(axs, maps.items()):
    im = ax.imshow(np.log10(Z), extent=[-2, 2, -2, 2], origin='lower', cmap='gray_r', aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_title(r'$\log_{10}$ ' + ('$SSE(\\theta)$' if title == 'SSE' else r'$[SSE(\theta)+\kappa\rho(\theta)]$'))
    ax.set_xlabel(r'$\theta_1$'); ax.set_ylabel(r'$\theta_2$')
plt.tight_layout()
plt.show()
