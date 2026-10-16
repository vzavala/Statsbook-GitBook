# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Using the GP uncertainty to choose experiments (Victor Z, UW-Madison, 2024). The true response is
# $y=x^2$ (plus small noise). With 3, 4, 5, and 6 experiments $(x_\omega,y_\omega)$ (`x_omega`, `y_omega`)
# we plot the mean $m(x)$ and the band $m(x)\pm1.96\,\sigma(x)$; uncertainty is zero at the
# experiments and large far from them, which tells us where to sample next.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

np.random.seed(0)
x = np.linspace(-1, 1, 400).reshape(-1, 1)
designs = [[-1, 0, 0.5], [-1, 0, 0.5, 1], [-1, 0, 0.5, 1, 0.1], [-1, 0, 0.5, 1, 0.1, -0.5]]

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for ax, design in zip(axs.ravel(), designs):
    x_omega = np.array(design).reshape(-1, 1)
    y_omega = x_omega.ravel()**2 + 0.01 * np.random.randn(len(design))
    kernel = ConstantKernel(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01**2, random_state=0)
    gp.fit(x_omega, y_omega)
    m_x, sigma_x = gp.predict(x, return_std=True)
    ax.fill_between(x.ravel(), m_x - 1.96*sigma_x, m_x + 1.96*sigma_x, color='k', alpha=0.1)
    ax.plot(x, m_x, '--k', linewidth=1.5)
    ax.scatter(x_omega, y_omega, edgecolor='black', facecolor='white', s=60, zorder=3)
    ax.set_title(f'{len(design)} experiments')
    ax.set_xlim(-1, 1); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.grid(True)
plt.tight_layout()
plt.show()
