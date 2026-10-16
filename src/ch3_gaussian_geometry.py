# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Geometry of a 2D Gaussian (Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mu$, $\Sigma$ | `mu`, `Sigma` | mean and covariance |
# | $1-\alpha$ | `1 - alpha` | probability of the region (95%) |
# | $\mathbb{Q}(1-\alpha)$ of $\chi^2(n)$ | `Q_n` | quantile defining the ellipse $\mathcal{E}$ and its bounding box $\mathcal{B}$ |
# | $\mathbb{Q}(1-\alpha)$ of $\chi^2(1)$ | `Q_1` | quantile defining the marginal box |
#
# The `ellipse` helper (`Support/ellipse.py`) returns the ellipse
# $\{(x-\mu)^T\Sigma^{-1}(x-\mu)= \mathbb{Q}\}$ and its tightest bounding box.

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, multivariate_normal
sys.path.append(path + 'Support')
from ellipse import ellipse

mu = np.array([0, 0])
Sigma = np.array([[1, -0.5], [-0.5, 1]])
alpha, n = 0.05, 2

Q_n = chi2.ppf(1 - alpha, n)
x_e, y_e, _, _, bbox = ellipse(np.linalg.inv(Sigma), Q_n, 200, mu)     # ellipse E and box B

Q_1 = chi2.ppf(1 - alpha, 1)
half = np.sqrt(Q_1 * np.diag(Sigma))                                   # marginal box half-widths
mbox = mu + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * half

# %%
x1 = np.linspace(-4, 4, 200)
x2 = np.linspace(-4, 4, 200)
X1, X2 = np.meshgrid(x1, x2)
f = multivariate_normal(mu, Sigma).pdf(np.dstack([X1, X2]))   # joint pdf (captures the correlation)
f_1 = norm.pdf(x1, mu[0], np.sqrt(Sigma[0, 0]))
f_2 = norm.pdf(x2, mu[1], np.sqrt(Sigma[1, 1]))

fig, axs = plt.subplots(2, 2, figsize=(10, 9))
axs[0, 0].plot(x1, f_1, 'k')
for v in (mbox[0, 0], mbox[1, 0]):
    axs[0, 0].axvline(v, color='k', ls='--')
axs[0, 0].set_xlabel('$x_1$'); axs[0, 0].set_ylabel('$f_1(x_1)$')
im = axs[0, 1].imshow(f, extent=[-4, 4, -4, 4], cmap='gray_r', origin='lower')
fig.colorbar(im, ax=axs[0, 1])
axs[0, 1].set_xlabel('$x_1$'); axs[0, 1].set_ylabel('$x_2$')
axs[1, 0].plot(x_e, y_e, 'k', label=r'ellipse $\mathcal{E}$')
axs[1, 0].plot(bbox[:, 0], bbox[:, 1], 'k--', label=r'bounding box $\mathcal{B}$')
axs[1, 0].plot(mbox[:, 0], mbox[:, 1], 'k:', label='marginal box')
axs[1, 0].set_xlabel('$x_1$'); axs[1, 0].set_ylabel('$x_2$')
axs[1, 0].legend(fontsize=9)
axs[1, 0].set_aspect('equal')
axs[1, 1].plot(f_2, x2, 'k')
for v in (mbox[0, 1], mbox[2, 1]):
    axs[1, 1].axhline(v, color='k', ls='--')
axs[1, 1].set_xlabel('$f_2(x_2)$'); axs[1, 1].set_ylabel('$x_2$')
plt.tight_layout()
plt.show()
print(f"chi2(2) quantile Q_n = {Q_n:.2f},  chi2(1) quantile Q_1 = {Q_1:.2f}")
