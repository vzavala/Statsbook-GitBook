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
# **Additional example: lasso trade-off for the Gibbs reactor** (Victor Z and Jaron T, UW-Madison, 2024).
# Predict conversion $y$ from pressure, the three outlet flows, and temperature (scaled), solving
#
# $$\min_\theta\ MSE(\theta)+\kappa\,\rho(\theta),\qquad \rho(\theta)=\lVert\theta\rVert_1,$$
#
# for a range of $\kappa$ and drawing the trade-off curve $MSE(\hat{\theta}(\kappa))$ vs $\rho(\hat{\theta}(\kappa))$.
# Python names: `X` (with intercept column), `y`, `kappa`, `theta_hat`, `MSE`, `rho`, `NNZ`
# (number of nonzero parameters).

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data_cov = np.loadtxt(path+'Data/gibbs_covariance.dat')
data_low = np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')
data = np.hstack((data_cov, data_low[:, :1]))
y = data[:, 1].copy()                       # conversion
X = data[:, [0, 2, 3, 4]]                   # pressure and outlet flows
np.random.seed(1)
y += np.random.normal(0, 0.05, y.shape)
X = (X - X.mean(axis=0)) / X.std(axis=0)    # scale inputs
X = np.hstack((np.ones((len(y), 1)), X))    # intercept theta_0

def MSE_of(theta):
    return np.mean((X @ theta - y)**2)

def rho(theta):
    return np.sum(np.abs(theta))

theta_ls = np.linalg.solve(X.T @ X, X.T @ y)     # kappa = 0 (least squares)
print("least-squares theta_hat =", np.round(theta_ls, 4))
print("eigenvalues of X^T X:", np.round(np.linalg.eigvalsh(X.T @ X), 2))

# %%
kappa_values = np.logspace(-6, 0, 40)
results = []
theta0 = theta_ls.copy()
for kappa in kappa_values:
    sol = minimize(lambda th: MSE_of(th) + kappa * rho(th), theta0, method='Powell',
                   options={'xtol': 1e-8, 'ftol': 1e-12, 'maxiter': 20000})
    theta0 = sol.x
    results.append((MSE_of(sol.x), rho(sol.x), np.sum(np.abs(sol.x) > 1e-4)))
MSE, rho_values, NNZ = map(np.array, zip(*results))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogx(rho_values, MSE, 'ko-', markersize=4)
axs[0].set_xlabel(r'$\rho(\hat{\theta}(\kappa))$'); axs[0].set_ylabel(r'$MSE(\hat{\theta}(\kappa))$')
axs[1].plot(NNZ, MSE, 'ko', markersize=4)
axs[1].set_xlabel('NNZ'); axs[1].set_ylabel('MSE')
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()
