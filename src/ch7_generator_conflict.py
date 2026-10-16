# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Conflict resolution in generator sizing (Victor Z and David Cole, UW-Madison, 2024). We trace trade-off
# curves by solving
# $\min_u\ \mathbb{E}[Y(u)]+\kappa\,\mathbb{SD}[Y(u)]$ for $\kappa\in[0,20]$ and
# $\min_u\ \mathrm{CVaR}_\alpha[Y(u)]$ for $\alpha\in[0,1]$ (here $u\in[0,10]$ is treated as continuous).
# Python names: `kappa`, `alpha`, `E(u)`, `SD(u)`, `CVaR(u, alpha)`, `u_star`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

alpha_R, alpha_u = 5, 15
np.random.seed(0)
S = 1000
L_omega = np.random.weibull(2, S) * 10

def Y(u):
    return alpha_R * np.maximum(L_omega - u, 0)**2 + alpha_u * u**2

def E(u):
    return Y(u).mean()

def SD(u):
    return Y(u).std()

def CVaR(u, alpha):
    y = Y(u)
    return y[y >= np.quantile(y, alpha)].mean()

def solve(objective):
    return minimize_scalar(objective, bounds=(0, 10), method='bounded').x

kappa = np.arange(0, 21)
u_kappa = np.array([solve(lambda u: E(u) + k * SD(u)) for k in kappa])
alpha = np.linspace(0, 0.99, 12)
u_alpha = np.array([solve(lambda u: CVaR(u, a)) for a in alpha])

fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for row, (u_opt, par, name) in enumerate(((u_kappa, kappa, r'\kappa'), (u_alpha, alpha, r'\alpha'))):
    axs[row, 0].plot([E(u) for u in u_opt], [SD(u) for u in u_opt], 'k-o', markerfacecolor='w')
    axs[row, 0].set_xlabel(r'$\mathbb{E}[Y(u^*)]$'); axs[row, 0].set_ylabel(r'$\mathbb{SD}[Y(u^*)]$')
    axs[row, 1].plot(par, u_opt, 'k-o', markerfacecolor='w')
    axs[row, 1].set_xlabel(f'${name}$'); axs[row, 1].set_ylabel('$u^*$')
for ax in axs.ravel():
    ax.grid(True)
plt.tight_layout()
plt.show()
