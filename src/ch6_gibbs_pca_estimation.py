# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Principal component estimation for the Gibbs reactor (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mathbf{X}=[1, X_1,\dots,X_5]$ | `X` | intercept, $P$, $CO$, $H_2$, $CH_3OH$ flows, $T$ ($S=250$) |
# | $\mathbf{y}$ | `y` | conversion |
# | $\hat{\theta}$, $\mathrm{Cov}(\hat{\theta})$ | `theta_hat`, `Cov_theta` | standard estimate |
# | $\mathbf{W}_1$, $\boldsymbol{\Lambda}_1$, $n_1$ | `W1`, `Lam1`, `n1` | retained eigenvectors/eigenvalues |
# | $\mathbf{T}_1=\mathbf{X}\mathbf{W}_1$ | `T1` | reduced inputs |
# | $\hat{\gamma}$, $\hat{\theta}=\mathbf{W}_1\hat{\gamma}$ | `gamma_hat`, `theta_hat_pca` | PCA estimate |

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.hstack((np.loadtxt(path+'Data/gibbs_covariance.dat'),
                  np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')[:, [0]]))
S = len(data)
np.random.seed(2024)
data[:, 1] += np.random.normal(0, 0.05, S)
y = data[:, 1]
X = np.hstack((np.ones((S, 1)), data[:, [0, 2, 3, 4, 5]]))
print(np.round(X[:6], 2))

# %% [markdown]
# Standard MLE: $\hat{\theta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. The kernel has three
# (numerically) zero eigenvalues, so $\mathrm{Cov}(\hat{\theta})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$ is singular.

# %%
theta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
SSE = np.mean((X @ theta_hat - y)**2)
lam, W = np.linalg.eigh(X.T @ X)
order = np.argsort(lam)[::-1]
lam, W = lam[order], W[:, order]
print(f"SSE (mean squared error) = {SSE:.3e}")
print("eigenvalues of X^T X:", np.array2string(lam, precision=2))

# %% [markdown]
# Keep the $n_1=3$ informative eigenvectors, project $\mathbf{T}_1=\mathbf{X}\mathbf{W}_1$, estimate
# $\hat{\gamma}=\arg\min\frac{1}{2}\lVert\mathbf{y}-\mathbf{T}_1\gamma\rVert^2$, and map back $\hat{\theta}=\mathbf{W}_1\hat{\gamma}$ with
# $\mathrm{Cov}(\hat{\theta})=\sigma^2\mathbf{W}_1\boldsymbol{\Lambda}_1^{-1}\mathbf{W}_1^T$.

# %%
n1 = 3
W1, Lam1 = W[:, :n1], np.diag(lam[:n1])
T1 = X @ W1
gamma_hat = np.linalg.solve(T1.T @ T1, T1.T @ y)
theta_hat_pca = W1 @ gamma_hat
e = X @ theta_hat_pca - y
sigma2 = np.var(e)
Cov_theta_pca = sigma2 * W1 @ np.linalg.inv(Lam1) @ W1.T
print(f"SSE (PCA) = {np.mean(e**2):.3e}")
print("V(theta_hat) with PCA:", np.array2string(np.diag(Cov_theta_pca), precision=2))

fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, th, title in ((axs[0], theta_hat, 'standard'), (axs[1], theta_hat_pca, 'PCA')):
    ax.plot(y, X @ th, 'ko', markerfacecolor='w', markersize=4)
    ax.plot([0, 1], [0, 1], 'k')
    ax.set_title(title)
    ax.set_xlabel('$y$'); ax.set_ylabel(r'$\hat{y}$')
    ax.axis([0.2, 1, 0.2, 1])
    ax.grid(True)
plt.tight_layout()
plt.show()
