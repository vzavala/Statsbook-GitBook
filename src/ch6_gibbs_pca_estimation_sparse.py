# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Sparse eigenvectors to find redundant variables in the Gibbs reactor data
# (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2019).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mathbf{W}$ | `W` | eigenvectors of $\mathbf{X}^T\mathbf{X}$ (rows: 1, $X_1,\dots,X_5$) |
# | $\tilde{\mathbf{W}}$, $\kappa$ | `W_sparse`, `kappa` | sparsified eigenvectors and sparsity weight |
#
# Here each eigenvector is sparsified by solving
# $\tilde{w}_j=\arg\min_w \frac{1}{2}\lVert w_j-w\rVert_2^2+\kappa\lVert w\rVert_1$ (soft thresholding),
# with $\kappa$ large enough to keep one entry. (For eigenvectors with zero eigenvalue
# $\mathbf{X}w_j=0$, so the data-space version on the slide returns $\tilde{w}_j=0$.)

# %%
import numpy as np
import matplotlib.pyplot as plt

names = ['1', 'X1 (P)', 'X2 (CO)', 'X3 (H2)', 'X4 (CH3OH)', 'X5 (T)']
data = np.hstack((np.loadtxt(path+'Data/gibbs_covariance.dat'),
                  np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')[:, [0]]))
S = len(data)
np.random.seed(2024)
data[:, 1] += np.random.normal(0, 0.05, S)
y = data[:, 1]
X = np.hstack((np.ones((S, 1)), data[:, [0, 2, 3, 4, 5]]))

lam, W = np.linalg.eigh(X.T @ X)
order = np.argsort(lam)[::-1]
lam, W = lam[order], W[:, order]
np.set_printoptions(precision=2, suppress=True)
print("W =\n", W)

def soft_threshold(w, kappa):
    return np.sign(w) * np.maximum(np.abs(w) - kappa, 0)

W_sparse = np.zeros_like(W)
for j in range(W.shape[1]):
    kappa = np.sort(np.abs(W[:, j]))[-2]          # keeps only the largest entry
    w = soft_threshold(W[:, j], kappa)
    W_sparse[:, j] = w / np.linalg.norm(w)
print("W_sparse =\n", np.abs(W_sparse))
for j in range(3, 6):
    print(f"eigenvector {j+1} (lambda = {lam[j]:.1e}) is driven by {names[np.argmax(np.abs(W_sparse[:, j]))]}")

# %% [markdown]
# The three non-informative eigenvectors point to CO, CH$_3$OH and to the constant column
# (the intercept and $X_5=T$ are identical up to scale because $T$ is constant). Dropping
# $X_2$, $X_4$, $X_5$ keeps the fit and gives precise estimates.

# %%
keep = [0, 1, 3]                                  # intercept, P, H2
X_red = X[:, keep]
theta_red = np.linalg.solve(X_red.T @ X_red, X_red.T @ y)
e = X_red @ theta_red - y
Cov_red = np.var(e) * np.linalg.inv(X_red.T @ X_red)
print("kept:", [names[k] for k in keep])
print("theta_hat =", np.array2string(theta_red, formatter={'float': '{:.3e}'.format}))
print("V(theta_hat) =", np.array2string(np.diag(Cov_red), formatter={'float': '{:.3e}'.format}))
print(f"SSE = {np.mean(e**2):.3e}")

plt.plot(y, X_red @ theta_red, 'ko', markerfacecolor='w', markersize=4)
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('$y$'); plt.ylabel(r'$\hat{y}$')
plt.axis([0.2, 1, 0.2, 1]); plt.grid(True)
plt.show()
