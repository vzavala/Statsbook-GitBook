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
# Uncertainty quantification for the heat-capacity model (Victor Z and Jaron T, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\hat{\theta}$, $\mathrm{Cov}[\hat{\theta}]$ | `theta_hat`, `Cov_theta` | estimate and its covariance |
# | $q_{n+1}$ | `q_np1` | 0.95 quantile of $\chi^2(n+1)$ |
# | $m_i=\sqrt{q_{n+1}\mathrm{Cov}[\hat{\theta}]_{ii}}$ | `m_theta` | parameter CI half-widths |
# | $\mathrm{Cov}[\hat{\mathbf{y}}]=\mathbf{X}\mathrm{Cov}[\hat{\theta}]\mathbf{X}^T$ | `Cov_y` | prediction covariance |
# | $q_S$, $m_\omega$ | `q_S`, `m_y` | 0.95 quantile of $\chi^2(S)$ and prediction CI half-widths |

# %% [part] Confidence intervals for the parameters

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
sys.path.append(path + 'Support')
from ellipse import ellipse

theta = np.array([6.190, 2.923e-3, -7.052e-7])
x = np.repeat([300.0, 600.0, 900.0], 10)
S = len(x)
X = np.column_stack([np.ones(S), x, x**2])
np.random.seed(0)
y = X @ theta + np.random.normal(0, 0.1, S)

theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
y_hat = X @ theta_hat
sigma_hat = norm.fit(y - y_hat)[1]
Cov_theta = np.linalg.inv(X.T @ X) * sigma_hat**2

n_par = len(theta_hat)                         # n + 1 parameters
q_np1 = chi2.ppf(0.95, n_par)
m_theta = np.sqrt(q_np1 * np.diag(Cov_theta))
print(f"q_(n+1) = {q_np1:.2f}")
for i in range(n_par):
    print(f"theta_{i} = {theta_hat[i]: .3e} +/- {m_theta[i]:.2e}   (true {theta[i]: .3e})")

# %% [markdown]
# 95% confidence ellipses for pairs of parameters (deviations $\Delta\theta=\theta-\hat{\theta}$).

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (i, j) in zip(axs, [(0, 1), (1, 2), (0, 2)]):
    C_ij = Cov_theta[np.ix_([i, j], [i, j])]
    xe, ye, _, _, _ = ellipse(np.linalg.inv(C_ij), chi2.ppf(0.95, 2), 200, [0, 0])
    ax.plot(xe, ye, 'k')
    ax.plot(0, 0, 'k+')
    ax.set_xlabel(rf'$\Delta\theta_{i}$')
    ax.set_ylabel(rf'$\Delta\theta_{j}$')
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [part] Confidence intervals for the predictions

# %% [markdown]
# $\hat{\mathbf{y}}\sim\mathcal{N}(\mathbf{y},\mathbf{X}\mathrm{Cov}[\hat{\theta}]\mathbf{X}^T)$, so
# $y_\omega=\hat{y}_\omega\pm m_\omega$ with $m_\omega=\sqrt{q_S\,\mathrm{Cov}[\hat{\mathbf{y}}]_{\omega\omega}}$.

# %%
Cov_y = X @ Cov_theta @ X.T
q_S = chi2.ppf(0.95, S)
m_y = np.sqrt(q_S * np.diag(Cov_y))
print(f"q_S = {q_S:.2f}  (S = {S})")

plt.plot(y_hat, 'ko', markerfacecolor='black', label=r'$\hat{y}_\omega$')
plt.plot(y, 'ko', markerfacecolor='white', label=r'$y_\omega$ (observed)')
plt.plot(y_hat - m_y, 'k--', label='95% CI')
plt.plot(y_hat + m_y, 'k--')
plt.xlabel(r'$\omega$', fontsize=14)
plt.ylabel(r'$y_\omega$', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# For comparison, `statsmodels` reports the usual $t$-based (marginal) 95% intervals, which are
# narrower than the joint $\chi^2(n+1)$ intervals above.

# %%
import statsmodels.api as sm
print(sm.OLS(y, X).fit().conf_int(0.05))
