# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# PCA of the Gibbs reactor data (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mathbf{X}$ ($S\times n$) | `X` | data matrix with columns $(T, P, C)$, centered and scaled |
# | $\boldsymbol{\Sigma}=\mathbf{X}^T\mathbf{X}$ | `Sigma` | kernel matrix |
# | $\mathbf{W}$, $\lambda_k$ | `W`, `lam` | eigenvectors and eigenvalues (descending) |
# | $\alpha_k$ | `alpha` | spectrum (% of information) |
# | $\mathbf{T}=\mathbf{X}\mathbf{W}$ | `T_pc` | principal components (scores) |
#
# (`T_pc` avoids a clash with the temperature `T`.)

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.vstack((np.loadtxt(path+'Data/gibbs_hightemp_class.dat'),
                  np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')))
S = len(data)
np.random.seed(1)
T = data[:, 0] + np.random.randn(S) * 0.1    # temperature
P = data[:, 1] + np.random.randn(S) * 0.1    # pressure
C = data[:, 2] + np.random.randn(S) * 0.1    # conversion

X_raw = np.column_stack((T, P, C))
X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)     # zero-mean, unit-variance columns

# %% [markdown]
# Eigendecomposition $\boldsymbol{\Sigma}=\mathbf{W}\boldsymbol{\Lambda}\mathbf{W}^T$, spectrum
# $\alpha_k=100\,\lambda_k/\sum_j\lambda_j$, and projection $\mathbf{T}=\mathbf{X}\mathbf{W}$.

# %%
Sigma = X.T @ X
lam, W = np.linalg.eigh(Sigma)
order = np.argsort(lam)[::-1]
lam, W = lam[order], W[:, order]
alpha = 100 * lam / lam.sum()
T_pc = X @ W
print("eigenvalues:", np.round(lam, 1), "  spectrum alpha_k [%]:", np.round(alpha, 1))
print("Cov(T) = T^T T is diagonal:", np.allclose(T_pc.T @ T_pc, np.diag(lam)))

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(221, projection='3d')
ax.scatter(T, P, C, c='black', s=4)
ax.set_xlabel('$T$'); ax.set_ylabel('$P$'); ax.set_zlabel('$C$')
ax = fig.add_subplot(222)
ax.scatter(P, C, c='black', s=4)
ax.set_xlabel('$P$'); ax.set_ylabel('$C$')
ax = fig.add_subplot(223)
ax.scatter(T_pc[:, 0], T_pc[:, 1], c='black', s=4)
ax.set_xlabel('$T_1$'); ax.set_ylabel('$T_2$')
ax = fig.add_subplot(224)
ax.scatter(T_pc[:, 1], T_pc[:, 2], c='black', s=4)
ax.set_xlabel('$T_2$'); ax.set_ylabel('$T_3$')
for a in fig.axes[1:]:
    a.grid(True)
plt.tight_layout()
plt.show()
