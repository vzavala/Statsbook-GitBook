# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Finding data redundancies with the SVD (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mathbf{X}=\mathbf{U}\mathbf{S}\mathbf{V}^T$ | `U`, `s`, `Vt` | SVD (`s` holds the singular values $s_j$; `S` is the sample size) |
# | $\lambda_i=s_i^2$ | `lam` | eigenvalues of $\mathbf{X}^T\mathbf{X}$ |
# | $\mathrm{rank}(\mathbf{X}[:,1..k])$ | `np.linalg.matrix_rank` | rank as columns are added |

# %%
import numpy as np

data = np.hstack((np.loadtxt(path+'Data/gibbs_covariance.dat'),
                  np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')[:, [0]]))
S = len(data)
X = np.hstack((np.ones((S, 1)), data[:, [0, 2, 3, 4, 5]]))
np.set_printoptions(precision=2, suppress=True)
print("first rows of X:\n", X[:6])

U, s, Vt = np.linalg.svd(X, full_matrices=False)
print("diag(S) =", s)
print("lambda = s^2 =", np.array2string(s**2, precision=3))
for k in range(1, X.shape[1] + 1):
    print(f"rank(X[:, 1..{k}]) = {np.linalg.matrix_rank(X[:, :k])}")
