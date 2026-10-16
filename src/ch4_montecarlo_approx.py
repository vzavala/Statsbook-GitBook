# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Convergence of Monte Carlo approximations (Victor Z, UW-Madison, 2024).
# Samples come from a Weibull RV with scale $\beta=2$ and shape $\xi=1$ (as in the slide figure;
# the slide text writes "Weibull(2,1)" in Matlab's (scale, shape) order).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $S$ | `S_values` | sample sizes |
# | $\hat{\mathbb{E}}^S_{\phi(X)}=\frac{1}{S}\sum_\omega\phi(x_\omega)$ | `E_hat`, `E_log_hat` | MC estimates of $\mathbb{E}[X]$, $\mathbb{E}[\log X]$ |
# | $\hat{\mathbb{V}}^S$ | `V_hat` | MC estimate of $\mathbb{V}[e^{-X}+X^2]$ |
# | $\hat{\mathbb{P}}(X\in\mathcal{A})$ | `P_hat` | MC estimate of $\mathbb{P}(X\leq 1)$ |
#
# We use $e^{-X}$ instead of $e^{X}$: with this RV, $\mathbb{V}[e^{X}]$ is infinite.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

xi, beta = 1, 2
X = weibull_min(xi, scale=beta)

np.random.seed(1)
S_values = np.arange(1, 5001, 10)
E_hat, E_log_hat, V_hat, P_hat = [], [], [], []
for S in S_values:
    x_omega = X.rvs(size=S)
    E_hat.append(np.mean(x_omega))
    E_log_hat.append(np.mean(np.log(x_omega)))
    V_hat.append(np.var(np.exp(-x_omega) + x_omega**2))
    P_hat.append(np.mean(x_omega <= 1))

# %%
exact = {'E': X.mean(), 'P': X.cdf(1)}
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
panels = [(E_hat, r'$\hat{\mathbb{E}}[X]$', exact['E']), (E_log_hat, r'$\hat{\mathbb{E}}[\log X]$', None),
          (V_hat, r'$\hat{\mathbb{V}}[e^{-X}+X^2]$', None), (P_hat, r'$\hat{\mathbb{P}}(X\leq 1)$', exact['P'])]
for ax, (vals, label, ref) in zip(axs.ravel(), panels):
    ax.plot(S_values, vals, 'o', markerfacecolor='w', markeredgecolor='black', markersize=4)
    if ref is not None:
        ax.axhline(ref, color='k')
    ax.set_xlabel('$S$')
    ax.set_ylabel(label)
    ax.grid(True)
plt.tight_layout()
plt.show()
print(f"exact E[X] = {exact['E']:.3f},  exact P(X <= 1) = {exact['P']:.3f}")
