# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# **Additional example: residuals of a wrong model** (Victor Z and Leo Gonzalez, UW-Madison, 2024).
# Data come from $Y=\theta X^2+\epsilon$ but we fit $Y=\theta X+\epsilon$. The residuals
# $\hat{\epsilon}_\omega=y_\omega-\hat{\theta}x_\omega$ show a systematic trend and are not
# $\mathcal{N}(0,\sigma^2)$: part of the structure has leaked into the "noise".
# Python names as in `ch5_linear_est_example`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(0)
S, theta, sigma = 100, 2, 0.25
x_omega = rng.random(S)
y_omega = theta * x_omega**2 + rng.normal(0, sigma, S)      # true structure is quadratic

theta_hat = np.sum(x_omega * y_omega) / np.sum(x_omega**2)   # fit of the (wrong) linear model
y_hat = theta_hat * x_omega
eps_hat = y_omega - y_hat
print(f"theta_hat = {theta_hat:.3f}")

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
order = np.argsort(x_omega)
axs[0, 0].plot(x_omega[order], y_hat[order], 'k-', linewidth=1.5)
axs[0, 0].plot(x_omega, y_omega, 'ko', markerfacecolor='w')
axs[0, 0].set_xlabel('$x$'); axs[0, 0].set_ylabel('$y$')
lims = [y_omega.min(), y_omega.max()]
axs[0, 1].plot(lims, lims, 'k-')
axs[0, 1].plot(y_omega, y_hat, 'ko', markerfacecolor='w')
axs[0, 1].set_xlabel('$y$'); axs[0, 1].set_ylabel(r'$\hat{y}$')
axs[1, 0].axhline(0, color='k')
axs[1, 0].plot(x_omega, eps_hat, 'ko', markerfacecolor='w')
axs[1, 0].set_xlabel(r'$x_\omega$'); axs[1, 0].set_ylabel(r'$\hat{\epsilon}_\omega$')
e = np.linspace(-1, 1, 200)
axs[1, 1].hist(eps_hat, bins=20, density=True, edgecolor='black', facecolor='none')
axs[1, 1].plot(e, norm.pdf(e, *norm.fit(eps_hat)), 'k-', linewidth=1.5)
axs[1, 1].set_xlabel(r'$\epsilon$'); axs[1, 1].set_ylabel(r'$f(\epsilon)$')
for ax in axs.ravel():
    ax.grid(True)
plt.tight_layout()
plt.show()
