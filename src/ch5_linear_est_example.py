# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Scalar linear estimation with the correct model (Victor Z and Jaron T, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $(x_\omega,y_\omega)$, $S$ | `x_omega`, `y_omega`, `S` | observation pairs |
# | $\theta$, $\sigma$ | `theta`, `sigma` | true slope and noise SD ($\sigma^2=0.0625$) |
# | $\hat{\theta}=\sum_\omega x_\omega y_\omega/\sum_\omega x_\omega^2$ | `theta_hat` | MLE (least squares) |
# | $\hat{y}_\omega=\hat{\theta}x_\omega$, $\hat{\epsilon}_\omega$ | `y_hat`, `eps_hat` | predictions and residuals |
# | $SSE(\theta)$ | `SSE` | sum of squared errors |

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

theta, sigma = 2, 0.25

def generate(S, seed=0):
    """S observations of y = theta x + eps with eps ~ N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    x_omega = rng.random(S)
    y_omega = theta * x_omega + rng.normal(0, sigma, S)
    return x_omega, y_omega

S = 100
x_omega, y_omega = generate(S)
theta_hat = np.sum(x_omega * y_omega) / np.sum(x_omega**2)
y_hat = theta_hat * x_omega
eps_hat = y_omega - y_hat
mu_eps, sigma_eps = norm.fit(eps_hat)
print(f"theta_hat = {theta_hat:.3f};  residuals ~ N({mu_eps:.3f}, {sigma_eps:.3f}^2)")

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
order = np.argsort(x_omega)
axs[0, 0].plot(x_omega[order], y_hat[order], 'k-', linewidth=1.5, label='model')
axs[0, 0].plot(x_omega, y_omega, 'ko', markerfacecolor='w', label='data')
axs[0, 0].set_xlabel(r'$x_\omega$'); axs[0, 0].set_ylabel(r'$y_\omega$'); axs[0, 0].legend(loc='lower right')
lims = [y_omega.min(), y_omega.max()]
axs[0, 1].plot(lims, lims, 'k-')
axs[0, 1].plot(y_omega, y_hat, 'ko', markerfacecolor='w')
axs[0, 1].set_xlabel(r'$y_\omega$'); axs[0, 1].set_ylabel(r'$\hat{y}_\omega$')
axs[1, 0].axhline(0, color='k')
axs[1, 0].plot(eps_hat, 'ko', markerfacecolor='w')
axs[1, 0].set_ylim(-3*sigma, 3*sigma)
axs[1, 0].set_xlabel(r'$\omega$'); axs[1, 0].set_ylabel(r'$\hat{\epsilon}_\omega$')
e = np.linspace(-0.75, 0.75, 200)
axs[1, 1].hist(eps_hat, bins=np.arange(-0.75, 0.75, 0.1), density=True, edgecolor='black', facecolor='none')
axs[1, 1].plot(e, norm.pdf(e, mu_eps, sigma_eps), 'k-', linewidth=1.5)
axs[1, 1].set_xlabel(r'$\epsilon$'); axs[1, 1].set_ylabel(r'$f(\epsilon)$')
for ax in axs.ravel():
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# $SSE(\theta)=\frac{1}{2}\sum_\omega(y_\omega-\theta x_\omega)^2$ (scaled by $1/S$) becomes sharper as $S$ grows.

# %%
theta_grid = np.linspace(0, 4, 200)
for S, style in ((10, 'k-'), (100, 'k--'), (1000, 'k:')):
    x_omega, y_omega = generate(S)
    SSE = np.array([0.5 * np.sum((y_omega - th * x_omega)**2) for th in theta_grid])
    theta_hat = np.sum(x_omega * y_omega) / np.sum(x_omega**2)
    plt.plot(theta_grid, SSE / S, style, linewidth=1.5, label=rf'$S={S}$, $\hat\theta={theta_hat:.2f}$')
plt.grid(True)
plt.xlabel(r'$\theta$', fontsize=14)
plt.ylabel(r'$SSE(\theta)/S$', fontsize=14)
plt.legend()
plt.show()
