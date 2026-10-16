# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Method of moments for a Weibull RV (Victor Z, UW-Madison, 2023).
# The matching equations
#
# $$\hat{\mathbb{E}}_X=\beta\,\Gamma(1+1/\xi),\qquad \hat{\mathbb{V}}_X=\beta^2\left(\Gamma(1+2/\xi)-\Gamma(1+1/\xi)^2\right)$$
#
# have no closed-form solution, so we solve them numerically for $\theta=(\xi,\beta)$.
# Python names: `x_omega`, `E_hat`, `V_hat`, `xi_hat`, `beta_hat`; data come from
# $\mathrm{Weibull}(\xi=2,\beta=1)$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import gamma as Gamma
from scipy.stats import weibull_min

def matching_error(theta, E_hat, V_hat):
    """Moment-matching residuals for theta = (xi, beta)."""
    xi, beta = theta
    E = beta * Gamma(1 + 1/xi)
    V = beta**2 * (Gamma(1 + 2/xi) - Gamma(1 + 1/xi)**2)
    return [E - E_hat, V - V_hat]

xi_true, beta_true = 2, 1
x = np.linspace(0, 5, 200)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for row, S in enumerate((10, 100)):
    np.random.seed(1)
    x_omega = weibull_min.rvs(xi_true, scale=beta_true, size=S)
    E_hat, V_hat = np.mean(x_omega), np.var(x_omega)
    xi_hat, beta_hat = fsolve(matching_error, [1, 1], args=(E_hat, V_hat))
    print(f"S = {S:3d}:  E_hat = {E_hat:.3f}, V_hat = {V_hat:.3f}  ->  xi_hat = {xi_hat:.2f}, beta_hat = {beta_hat:.2f}")

    axs[row, 0].hist(x_omega, bins=np.arange(0, 5, 0.5), density=True, edgecolor='black', facecolor='none')
    axs[row, 0].plot(x, weibull_min.pdf(x, xi_hat, scale=beta_hat), 'k', linewidth=1.5)
    axs[row, 0].set_ylabel('$f(x)$')
    axs[row, 1].step(np.sort(x_omega), np.arange(1, S + 1) / S, where='post', color='black')
    axs[row, 1].plot(x, weibull_min.cdf(x, xi_hat, scale=beta_hat), 'k', linewidth=1.5)
    axs[row, 1].set_ylabel('$F(x)$')
    for ax in axs[row]:
        ax.set_xlabel('$x$')
        ax.set_title(f'$S={S}$')
        ax.grid(True)
plt.tight_layout()
plt.show()
