# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Generator sizing: deterministic versus stochastic decision (Victor Z and David Cole, UW-Madison, 2024).
# Same model as `ch7_generator_det`; the stochastic decision solves
# $\min_{u\in\mathcal{U}}\ \mathrm{CVaR}_\alpha[Y(u)]$ with $\alpha=0.9$, evaluated with the $S$ load samples.
# Python names: `rho(u)` (the risk measure), `u_D`, `u_star`; other risk measures are tabulated for comparison.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

alpha_R, alpha_u = 5, 15

def cost(u, L):
    R = np.maximum(L - u, 0)
    return alpha_R * R**2 + alpha_u * u**2

np.random.seed(0)
S = 1000
L_omega = weibull_min.rvs(2, scale=10, size=S)
U = np.arange(0, 11)
u_D = U[np.argmin([cost(u, L_omega.max()) for u in U])]

def CVaR(y, alpha=0.9):
    return y[y >= np.quantile(y, alpha)].mean()

rho = {u: CVaR(cost(u, L_omega)) for u in U}
u_star = min(rho, key=rho.get)
print(f"deterministic u_D* = {u_D},  stochastic (CVaR) u* = {u_star}")
print(f"{'u':>3s}{'E':>9s}{'SD':>9s}{'VaR':>9s}{'CVaR':>9s}{'max':>9s}{'P(y>2000)':>11s}")
for u in U:
    y_u = cost(u, L_omega)
    print(f"{u:3d}{y_u.mean():9.1f}{y_u.std():9.1f}{np.quantile(y_u, 0.9):9.1f}{rho[u]:9.1f}"
          f"{y_u.max():9.1f}{np.mean(y_u > 2000):11.1%}")

# %%
y_D, y_s = cost(u_D, L_omega), cost(u_star, L_omega)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
axs[0].plot(U, [rho[u] for u in U], 'k-o', markerfacecolor='w')
axs[0].set_xlabel('$u$'); axs[0].set_ylabel(r'$\mathrm{CVaR}_{0.9}[Y(u)]$')
axs[1].hist(y_D, bins=np.arange(0, 3100, 100), edgecolor='black', facecolor='white', label=f'deterministic $u={u_D}$')
axs[1].hist(y_s, bins=np.arange(0, 3100, 100), edgecolor='black', facecolor='black', alpha=0.5, label=f'stochastic $u={u_star}$')
axs[1].set_xlabel('$y$'); axs[1].set_ylabel('Frequency'); axs[1].legend()
for y_u, style, label in ((y_D, 'k--', 'deterministic'), (y_s, 'k-', 'stochastic')):
    axs[2].step(np.sort(y_u), np.arange(1, S + 1) / S, style, where='post', label=label)
axs[2].set_xlabel('$y$'); axs[2].set_ylabel(r'$\hat{F}(y)$'); axs[2].legend(loc='lower right')
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()
