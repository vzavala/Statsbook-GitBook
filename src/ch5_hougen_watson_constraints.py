# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Hougen-Watson estimation with bounds $\underline{\theta}\leq\theta\leq\overline{\theta}$
# (Victor Z and Leo Gonzalez, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\underline{\theta}$, $\overline{\theta}$ | `theta_lb`, `theta_ub` | bounds $(0,\dots,0)$ and $(2,\dots,2)$ |
# | $\nabla_\theta\mathbf{m}(\hat{\theta})$ | `J` | Jacobian returned by the solver |
# | $\mathcal{H}(\hat{\theta})\approx\nabla_\theta\mathbf{m}^T\nabla_\theta\mathbf{m}$ | `H` | Gauss-Newton Hessian |
# | $\lambda$ | `lam` | eigenvalues of $\mathcal{H}$ |
# | $\mathrm{Cov}[\hat{\theta}]$, $\mathbb{V}[\hat{\theta}]$ | `Cov_theta`, `V_theta` | covariance and variances |

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

data = np.loadtxt(path+'Data/hougenwatson.dat')
X, y = data[:, :3], data[:, -1]
S = len(y)

def m(theta, X):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (theta[0]*x2 - x3/theta[4]) / (1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3)

def residual(theta):
    return m(theta, X) - y

# %% [markdown]
# Unconstrained problem from the bad initial guess $\theta=(-1,\dots,-1)$:

# %%
sol = least_squares(residual, -np.ones(5), method='trf')
print("unconstrained theta_hat =", np.round(sol.x, 2))

# %% [markdown]
# Constrained problem with $\underline{\theta}=0$ and $\overline{\theta}=2$:

# %%
theta_lb, theta_ub = np.zeros(5), 2 * np.ones(5)
sol = least_squares(residual, np.ones(5), bounds=(theta_lb, theta_ub), method='trf')
theta_hat = sol.x
print("constrained theta_hat =", np.round(theta_hat, 2))

y_hat = m(theta_hat, X)
plt.plot([0, 15], [0, 15], 'k-', linewidth=1)
plt.scatter(y_hat, y, marker='o', color='w', edgecolor='k', s=50)
plt.xlabel(r'$\hat{y}$', fontsize=14); plt.ylabel('$y$', fontsize=14)
plt.axis([0, 15, 0, 15]); plt.grid(True)
plt.show()

# %% [markdown]
# Gauss-Newton Hessian, its eigenvalues, and $\mathrm{Cov}[\hat{\theta}]=\mathcal{H}(\hat{\theta})^{-1}\hat{\sigma}^2$.

# %%
J = sol.jac
H = J.T @ J
lam = np.linalg.eigvalsh(H)[::-1]
sigma2_hat = np.var(y_hat - y, ddof=1)
Cov_theta = np.linalg.inv(H) * sigma2_hat
V_theta = np.diag(Cov_theta)
np.set_printoptions(precision=3)
print("H =\n", H)
print("eigenvalues of H:", lam)
print("V[theta_hat] =", V_theta)
