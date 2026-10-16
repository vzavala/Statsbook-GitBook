# %% [markdown]
# **Additional example: recursive Bayesian estimation of the heat-capacity model.**
# With a Gaussian prior $\theta\sim\mathcal{N}(\bar{\theta},\Sigma_\theta)$ the Bayes estimate solves
#
# $$\min_\theta\ \frac{1}{2}(\mathbf{y}-\mathbf{X}\theta)^T\Sigma^{-1}(\mathbf{y}-\mathbf{X}\theta)+\frac{1}{2}(\theta-\bar{\theta})^T\Sigma_\theta^{-1}(\theta-\bar{\theta}),$$
#
# and the posterior covariance is $(\mathbf{X}^T\Sigma^{-1}\mathbf{X}+\Sigma_\theta^{-1})^{-1}$. Processing the data
# in batches, the posterior of one batch becomes the prior of the next.
# Python names: `theta_bar`, `Sigma_theta`, `Sigma` (noise covariance), `theta_hat`.

# %%
import numpy as np

theta = np.array([6.190, 2.923e-3, -7.052e-7])
x = np.repeat([300.0, 600.0, 900.0], 20)
S = len(x)
X = np.column_stack([np.ones(S), x, x**2])
np.random.seed(0)
sigma = 0.1
y = X @ theta + np.random.normal(0, sigma, S)
idx = np.random.permutation(S)
X, y = X[idx], y[idx]

def bayes_estimate(X, y, Sigma, theta_bar, Sigma_theta):
    """Minimizer of the Bayes objective (closed form for a linear model) and posterior covariance."""
    A = X.T @ np.linalg.solve(Sigma, X) + np.linalg.inv(Sigma_theta)
    b = X.T @ np.linalg.solve(Sigma, y) + np.linalg.solve(Sigma_theta, theta_bar)
    return np.linalg.solve(A, b), np.linalg.inv(A)

theta_bar, Sigma_theta = np.ones(3), np.eye(3)
theta_hat, Sigma_post = bayes_estimate(X, y, sigma**2 * np.eye(S), theta_bar, Sigma_theta)
print("all data at once:  theta_hat =", np.array2string(theta_hat, precision=4))

# %% [markdown]
# Recursive estimation with batches of 20 experiments:

# %%
batch = 20
for k in range(S // batch):
    rows = slice(k * batch, (k + 1) * batch)
    theta_bar, Sigma_theta = bayes_estimate(X[rows], y[rows], sigma**2 * np.eye(batch), theta_bar, Sigma_theta)
    print(f"after batch {k+1}: theta_hat = {np.array2string(theta_bar, precision=4)},  "
          f"SD = {np.array2string(np.sqrt(np.diag(Sigma_theta)), precision=2)}")
