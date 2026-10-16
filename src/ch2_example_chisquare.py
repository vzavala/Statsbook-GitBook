# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Confidence intervals and regions from chi-squared quantiles (Victor Z and J E Umaña, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $F_3\sim\mathcal{N}(\mu,\sigma^2)$ | `mu`, `sigma` | output flow of the mixer |
# | $\alpha$ | `alpha` | probability outside the interval |
# | $Y\sim\chi^2(1)$, $\mathbb{Q}_Y(1-\alpha)$ | `Q_Y` | chi-squared quantile |
# | $z_{\alpha/2}=\sqrt{\mathbb{Q}_Y(1-\alpha)}$ | `z_alpha_2` | critical value |

# %% [part] 90% interval for the mixer output

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

mu, sigma = 30, 1
alpha = 0.1
Q_Y = chi2.ppf(1 - alpha, df=1)            # Q_Y(1 - alpha), Y ~ chi2(1)
z_alpha_2 = np.sqrt(Q_Y)                   # equals Q_Z(1 - alpha/2)
print(f"Q_Y(0.9) = {Q_Y:.2f},  z_alpha/2 = {z_alpha_2:.2f}  (normal: {norm.ppf(1 - alpha/2):.2f})")
print(f"90% interval: [{mu - z_alpha_2*sigma:.2f}, {mu + z_alpha_2*sigma:.2f}] gpm")

# %% [part] Going further: a 2D confidence ellipse from chi2(2)

# %% [markdown]
# For a 2D standard normal $Z=(Z_1,Z_2)$, $Z^T\Sigma^{-1}Z\sim\chi^2(2)$, so the region
# $\{z: z^T\Sigma^{-1}z\leq \mathbb{Q}_{\chi^2(2)}(1-\alpha)\}$ holds $Z$ with probability $1-\alpha$.
# (Multivariate Gaussians are covered in Chapter 3.) The `ellipse` helper is in `Support/ellipse.py`.

# %%
sys.path.append(path + 'Support')
from ellipse import ellipse

mu_vec = np.array([0, 0])
Sigma = np.array([[1, 0], [0, 1]])
level = chi2.ppf(1 - alpha, 2)                       # chi2(2) quantile
xe, ye, _, _, _ = ellipse(np.linalg.inv(Sigma), level, 100, mu_vec)
r1 = np.sqrt(chi2.ppf(1 - alpha, 1))                 # 1D interval half-width

plt.figure(figsize=(6, 6))
plt.plot(xe, ye, 'k', linewidth=1.5, label=r'$\chi^2(2)$ region')
plt.plot([-r1, r1], [0, 0], 'k--', linewidth=1.0, label=r'$\chi^2(1)$ intervals')
plt.plot([0, 0], [-r1, r1], 'k--', linewidth=1.0)
plt.xlabel('$z_1$', fontsize=14)
plt.ylabel('$z_2$', fontsize=14)
plt.axis([-2.5, 2.5, -2.5, 2.5])
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

np.random.seed(1)
S = 1000
z_omega = np.random.multivariate_normal(mu_vec, Sigma, S)
inside = np.einsum('ij,jk,ik->i', z_omega - mu_vec, np.linalg.inv(Sigma), z_omega - mu_vec) <= level
print(f"{inside.sum()} of {S} samples fall in the ellipse (expected about {S*(1-alpha):.0f})")
