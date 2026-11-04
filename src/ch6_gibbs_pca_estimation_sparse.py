# %% [markdown] [hide]
# **Setup.** Run this cell first: it locates the data files this notebook reads. On Google
# Colab it downloads them (the `Data/` and `Support/` folders) from the book's GitHub
# repository; on your computer, run the notebook from the folder that contains `Data/`.

# %% [hide]
import os
import sys
path = ''
if 'google.colab' in sys.modules and not os.path.isdir('Data'):
    if not os.path.isdir('statsbook'):
        os.system('git clone -q --depth 1 --filter=blob:none --sparse '
                  'https://github.com/__GITHUB_REPO__.git statsbook')
        os.system('git -C statsbook sparse-checkout set __NOTEBOOK_DIR__/Data __NOTEBOOK_DIR__/Support')
    path = 'statsbook/__NOTEBOOK_DIR__/'

# %% [markdown]
# Sparse eigenvectors to find redundant variables in the Gibbs reactor data
# (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2019).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mathbf{W}$ | `W` | eigenvectors of $\mathbf{X}^T\mathbf{X}$ (rows: 1, $X_1,\dots,X_5$) |
# | $\tilde{\mathbf{W}}$ | `W_sparse` | sparse eigenvectors with one nonzero entry each |
#
# The sparse eigenvectors are computed with the same method used in the book (the sparse PCA
# method of Hein and Bühler, used in the MATLAB script with one nonzero entry per component):
# each component selects the input column with the largest norm, and the data are then
# deflated along that component before computing the next one.
#
# Note that $\mathbf{X}$ has rank 3, so the last three eigenvalues of $\mathbf{X}^T\mathbf{X}$ are zero
# (up to round-off). Any rotation of the corresponding eigenvectors is also a valid set of
# eigenvectors, so the last three columns of `W` (and of `W_sparse`) can differ from those
# reported in the book, and can change with the numerical library used.

# %% [markdown]
# **Note:** This notebook reproduces Example 6.3 (Section 6.1.2) of *Statistics for Chemical Engineers*: sparse PCA to identify which input variables carry redundant information.

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
print("rank of X =", np.linalg.matrix_rank(X))
print("W =\n", W)

def sparse_pca_card1(X, num_comp):
    # sparse PCA with one nonzero entry per component and deflation of the data
    X_cur = X.copy()
    F = np.zeros((X.shape[1], num_comp))
    for l in range(num_comp):
        i = np.argmax(np.linalg.norm(X_cur, axis=0))   # best one-entry component
        F[i, l] = 1.0
        z = X_cur[:, i] / np.linalg.norm(X_cur[:, i])
        X_cur = X_cur - np.outer(z, z @ X_cur)        # deflate along the component
    return F

with np.errstate(all='ignore'):
    W_sparse = sparse_pca_card1(X, 6)
print("W_sparse =\n", W_sparse)
for j in range(3):
    print(f"component {j+1} selects {names[np.argmax(W_sparse[:, j])]}")

# %% [markdown]
# The first three (informative) sparse components select $X_5=T$, $X_1=P$ and $X_3=H_2$, which
# reproduces the first columns of $\tilde{\mathbf{W}}$ in the book. The input data matrix only contains three
# independent columns: $T$ is constant in this dataset (so its column is a multiple of the intercept
# column), and the flows of CO, H$_2$, and CH$_3$OH are linked by the mass balances. We thus keep the
# intercept, $X_1$, and $X_3$ and drop $X_2$, $X_4$, $X_5$; this keeps the fit and gives precise estimates.

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

# %% [markdown]
# **Note:** As in the book, keeping only X5, X2, and X4 gives a poor fit and a singular kernel matrix: these input variables are the problematic ones.
