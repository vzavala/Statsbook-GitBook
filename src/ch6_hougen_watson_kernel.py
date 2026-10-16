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
# Kernel model for the Hougen-Watson data (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $x_\omega$, $y_\omega$ | `X`, `y` | partial pressures and reaction rates |
# | $k(x_\omega,x_{\omega'})=\exp(-\gamma\lVert x_\omega-x_{\omega'}\rVert^2)$ | `rbf(xi, xj, gamma)` | RBF kernel |
# | $\mathbf{K}$, $\lambda$ | `K`, `lam` | kernel matrix and regularization |
# | $\hat{\mathbf{y}}=\mathbf{K}(\mathbf{K}+\lambda I)^{-1}\mathbf{y}$ | `y_hat` | kernel prediction |

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(path+'Data/hougenwatson.dat')
X, y = data[:, :3], data[:, -1]
S = len(y)
lam = 0.01

def rbf(xi, xj, gamma):
    return np.exp(-gamma * np.sum((xi - xj)**2))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, gamma in zip(axs, (0.1, 0.001, 0.0001)):
    K = np.array([[rbf(X[i], X[j], gamma) for j in range(S)] for i in range(S)])
    y_hat = K @ np.linalg.solve(K + lam * np.eye(S), y)
    ax.plot(y_hat, y, 'ko', markerfacecolor='w')
    ax.plot([0, 15], [0, 15], 'k--')
    ax.set_title(rf'$\gamma={gamma}$:  $\|y-\hat{{y}}\|$ = {np.linalg.norm(y - y_hat):.2f}')
    ax.set_xlabel(r'$\hat{y}$'); ax.set_ylabel('$y$')
    ax.grid(True)
plt.tight_layout()
plt.show()
