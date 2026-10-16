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
# Covariance and correlation for the Gibbs reactor (Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $x_\omega\in\mathbb{R}^5$, $S$ | `x_omega` (S x 5), `S` | pressure, conversion, outflows of CO, H$_2$, CH$_3$OH |
# | $\hat{m}_i=\hat{\mathbb{E}}[X_i]$ | `m_hat` | empirical means |
# | $\hat{\mathrm{Cov}}[X]$ | `Cov_hat` | sample covariance (with $1/S$) |
# | $\hat{\mathrm{PCorr}}[X]=D^{-1}\hat{\mathrm{Cov}}[X]D^{-1}$ | `PCorr_hat`, `D` | Pearson correlation |

# %% [part] Sample covariance and correlation

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

names = ['P', 'conversion', 'F_CO', 'F_H2', 'F_CH3OH']
x_omega = np.loadtxt(path+'Data/gibbs_covariance.dat')
S, n = x_omega.shape

np.random.seed(1)   # measurement noise (results differ slightly from the Matlab version)
x_omega = x_omega + np.random.normal(0, 1, (S, n)) * np.array([10, 0.1, 10, 10, 10])

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for ax, (i, j) in zip(axs.ravel(), [(0, 1), (0, 2), (2, 3), (3, 4)]):
    ax.plot(x_omega[:, i], x_omega[:, j], 'ko', markerfacecolor='w', markersize=4)
    ax.set_xlabel(names[i])
    ax.set_ylabel(names[j])
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# $\hat{\mathrm{Cov}}[X]_{ij}=\frac{1}{S}\sum_{\omega}(x_{\omega,i}-\hat{m}_i)(x_{\omega,j}-\hat{m}_j)$.
# (`np.cov` divides by $S-1$ unless `bias=True`.)

# %%
m_hat = x_omega.mean(axis=0)
Cov_hat = (x_omega - m_hat).T @ (x_omega - m_hat) / S
assert np.allclose(Cov_hat, np.cov(x_omega, rowvar=False, bias=True))

D = np.diag(np.sqrt(np.diag(Cov_hat)))
PCorr_hat = np.linalg.inv(D) @ Cov_hat @ np.linalg.inv(D)
assert np.allclose(PCorr_hat, np.corrcoef(x_omega, rowvar=False))

pd.options.display.float_format = '{:.2f}'.format
print("Cov_hat[X] =")
print(pd.DataFrame(Cov_hat, index=names, columns=names))
print("\nPCorr_hat[X] =")
print(pd.DataFrame(PCorr_hat, index=names, columns=names))
print("\neigenvalues of Cov_hat (all >= 0):", np.round(np.linalg.eigvalsh(Cov_hat), 3))

# %% [part] Conditional variance of pressure and conversion @ Example:Uncertainty Reduction in Gibbs Reactor

# %% [markdown]
# For a Gaussian $(X_1,X_2)$ (pressure, conversion):
# $\Sigma_{2|1}=\Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}$ and
# $\Sigma_{1|2}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$.
# We use the covariance values quoted on the slide. Note that the relative reduction
# $1-\Sigma_{2|1}/\Sigma_{22}=1-\Sigma_{1|2}/\Sigma_{11}=\rho^2$ is the same in both directions.

# %%
Sigma = np.array([[776.94, 1.83],
                  [1.83, 0.02]])     # Cov[X] for (pressure, conversion), from the slide
Sigma_2g1 = Sigma[1, 1] - Sigma[1, 0] / Sigma[0, 0] * Sigma[0, 1]
Sigma_1g2 = Sigma[0, 0] - Sigma[0, 1] / Sigma[1, 1] * Sigma[1, 0]
rho = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
print(f"Sigma_2|1 = {Sigma_2g1:.4f}  (reduction {1 - Sigma_2g1/Sigma[1, 1]:.0%})")
print(f"Sigma_1|2 = {Sigma_1g2:.2f}  (reduction {1 - Sigma_1g2/Sigma[0, 0]:.0%})")
print(f"rho = {rho:.2f},  rho^2 = {rho**2:.2f}")

# %% [markdown]
# The same computation with the full data set: the variance of conversion given the
# other four variables, $\Sigma_{1|2}$ with block $1=$ conversion and block $2=$ the rest.

# %%
order = [1, 0, 2, 3, 4]               # conversion first
C = Cov_hat[np.ix_(order, order)]
Sigma_11, Sigma_12, Sigma_22 = C[0, 0], C[0, 1:], C[1:, 1:]
Sigma_1g2 = Sigma_11 - Sigma_12 @ np.linalg.solve(Sigma_22, Sigma_12)
print(f"V[conversion] = {Sigma_11:.4f},  V[conversion | other 4 variables] = {Sigma_1g2:.4f}")
