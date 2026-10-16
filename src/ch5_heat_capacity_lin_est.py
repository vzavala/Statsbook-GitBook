# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Estimating heat capacity with a model that is linear in the parameters
# (Victor Z and Jaron T, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $X$ (temperature), $Y$ (heat capacity) | `x`, `y` | inputs [K] and outputs [cal/mol-K] |
# | $\mathbf{X}=[1,\ x,\ x^2]$, $\mathbf{y}$ | `X`, `y` | data matrix and output vector |
# | $\theta=(\theta_0,\theta_1,\theta_2)$ | `theta` (true), `theta_hat` | parameters |
# | $\hat{\mathbf{y}}$, $\hat{\epsilon}$ | `y_hat`, `eps_hat` | predictions and residuals |
# | $\hat{\sigma}$ | `sigma_hat` | estimated noise SD |
# | $\mathrm{Cov}[\hat{\theta}]$, $\mathcal{I}(\theta)$ | `Cov_theta`, `FI` | parameter covariance and Fisher information |

# %% [part] Least-squares estimate

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

# true model  C_p = theta_0 + theta_1 T + theta_2 T^2  (cal/mol-K, T in K)
theta = np.array([6.190, 2.923e-3, -7.052e-7])

# S = 30 experiments: 10 replicates at each of T = 300, 600, 900 K
x = np.repeat([300.0, 600.0, 900.0], 10)
S = len(x)
X = np.column_stack([np.ones(S), x, x**2])
y_true = X @ theta
np.random.seed(0)        # results differ slightly from Matlab (different random numbers)
sigma = 0.1
y = y_true + np.random.normal(0, sigma, S)

plt.scatter(x, y, facecolors='none', edgecolors='black', label='observations')
plt.scatter(x, y_true, color='black', label='true values')
plt.xlabel('$T$ [K]', fontsize=14)
plt.ylabel('Heat capacity [cal/mol-K]', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# With $\Sigma=\sigma^2 I$, MLE is SSE minimization:
# $\hat{\theta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. The minimizer is unique if the Hessian
# $\nabla^2_\theta SSE=\mathbf{X}^T\mathbf{X}$ is positive definite.

# %%
theta_hat = np.linalg.solve(X.T @ X, X.T @ y)
y_hat = X @ theta_hat
eps_hat = y - y_hat
print("theta_hat =", np.array2string(theta_hat, precision=3))
print("eigenvalues of X^T X:", np.array2string(np.linalg.eigvalsh(X.T @ X), precision=3))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot([6.5, 8.5], [6.5, 8.5], 'k')
axs[0].scatter(y, y_hat, facecolors='none', edgecolors='black', label='observations')
axs[0].scatter(y_true, y_hat, color='black', label='true values')
axs[0].set_xlabel('$y$'); axs[0].set_ylabel(r'$\hat{y}$'); axs[0].legend()
e = np.linspace(-0.75, 0.75, 200)
axs[1].hist(eps_hat, bins=np.arange(-0.75, 0.85, 0.1), density=True, edgecolor='black', facecolor='none')
axs[1].plot(e, norm.pdf(e, *norm.fit(eps_hat)), 'k-', linewidth=1.5)
axs[1].set_xlabel(r'$\epsilon$'); axs[1].set_ylabel(r'$f(\epsilon)$')
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [part] Parameter covariance

# %% [markdown]
# Fit $\epsilon\sim\mathcal{N}(\hat{\mu},\hat{\sigma}^2)$ to the residuals and use
# $\mathrm{Cov}[\hat{\theta}]=(\mathbf{X}^T\Sigma^{-1}\mathbf{X})^{-1}=(\mathbf{X}^T\mathbf{X})^{-1}\hat{\sigma}^2$.

# %%
mu_hat, sigma_hat = norm.fit(eps_hat)
Cov_theta = np.linalg.inv(X.T @ X) * sigma_hat**2
print(f"mu_hat = {mu_hat:.2e},  sigma_hat = {sigma_hat:.3f}")
print("Cov[theta_hat] =\n", np.array2string(Cov_theta, precision=3))

# %% [part] Fisher information

# %% [markdown]
# For the linear model, $\mathcal{I}(\theta)=\mathbf{X}^T\Sigma^{-1}\mathbf{X}=\mathrm{Cov}[\hat{\theta}]^{-1}$;
# large eigenvalues mean the data carry a lot of information.

# %%
FI = X.T @ X / sigma_hat**2
print("I(theta) =\n", np.array2string(FI, precision=3))
print("eigenvalues of I:", np.array2string(np.linalg.eigvalsh(FI)[::-1], precision=3))

# %% [part] R^2 and Q-Q plot of the residuals @ Residual Analysis

# %% [markdown]
# $S_y=\sum_\omega(y_\omega-\bar{y})^2$, $S_m=\sum_\omega(\hat{y}_\omega-\bar{y})^2$, $S_e=\sum_\omega(y_\omega-\hat{y}_\omega)^2$ and
# $R^2=S_m/S_y=1-S_e/S_y$. A Q-Q plot checks whether the residuals are Gaussian.

# %%
y_bar = np.mean(y)
S_y = np.sum((y - y_bar)**2)
S_m = np.sum((y_hat - y_bar)**2)
S_e = np.sum((y - y_hat)**2)
print(f"R^2 = S_m/S_y = {S_m/S_y:.4f}   (1 - S_e/S_y = {1 - S_e/S_y:.4f})")

probplot(eps_hat, dist="norm", plot=plt)
plt.title('')
plt.xlabel(r'Theoretical $\mathbb{Q}(\alpha)$', fontsize=14)
plt.ylabel(r'Empirical $\hat{\mathbb{Q}}(\alpha)$', fontsize=14)
plt.grid(True)
plt.show()
