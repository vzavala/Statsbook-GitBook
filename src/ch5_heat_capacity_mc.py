# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# **Monte Carlo estimate of the parameter covariance** (heat-capacity data). Perturb the outputs,
# $\mathbf{y}_k=\mathbf{y}+\mathbf{e}_k$ with $\mathbf{e}_k\sim\mathcal{N}(0,\hat{\sigma}^2 I)$, re-estimate
# $\hat{\theta}_k$ for $k=1,\dots,K$, and compute
# $\hat{\mathrm{Cov}}[\hat{\theta}]=\frac{1}{K-1}\sum_k(\hat{\theta}_k-\hat{\mathbb{E}}[\hat{\theta}])(\hat{\theta}_k-\hat{\mathbb{E}}[\hat{\theta}])^T$.
# For this linear model we can compare with the exact $(\mathbf{X}^T\mathbf{X})^{-1}\hat{\sigma}^2$.
# Python names: `K`, `theta_k` (K x 3), `Cov_mc`, `Cov_exact`.

# %%
import numpy as np

theta = np.array([6.190, 2.923e-3, -7.052e-7])
x = np.repeat([300.0, 600.0, 900.0], 10)
S = len(x)
X = np.column_stack([np.ones(S), x, x**2])
np.random.seed(0)
y = X @ theta + np.random.normal(0, 0.1, S)

theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
sigma2_hat = np.var(y - X @ theta_hat)
Cov_exact = np.linalg.inv(X.T @ X) * sigma2_hat

K = 1000
theta_k = np.array([np.linalg.solve(X.T @ X, X.T @ (y + np.random.normal(0, np.sqrt(sigma2_hat), S)))
                    for _ in range(K)])
E_theta = theta_k.mean(axis=0)
Cov_mc = (theta_k - E_theta).T @ (theta_k - E_theta) / (K - 1)

np.set_printoptions(precision=3)
print("exact Cov[theta_hat]:\n", Cov_exact)
print("Monte Carlo estimate:\n", Cov_mc)
print("ratio of variances (MC / exact):", np.diag(Cov_mc) / np.diag(Cov_exact))
