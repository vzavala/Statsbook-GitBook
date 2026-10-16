# %% [markdown]
# **Additional example: sample average approximation (SAA)** (Victor Zavala and Leo Gonzalez, UW-Madison, 2024).
# Stochastic problems are solved with samples: $\min_u \hat{\mathbb{E}}^S[\varphi(X,u)]=\frac{1}{S}\sum_\omega\varphi(x_\omega,u)$.
# For $\varphi(X,u)=(u-X)^2$ with $X\sim\mathcal{N}(1,10^2)$ the exact solution is $u^*=\mathbb{E}[X]=1$ with
# optimal value $\mathbb{V}[X]=100$. Repeating the SAA over $B$ batches gives the CLT-based 95% intervals
# (Chapter 4) of the optimal value and solution, which shrink as $S$ grows.
# Python names: `S_values`, `B`, `u_hat` (SAA solutions), `E_hat` (SAA optimal values).

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)
mu, sigma = 1, 10
B = 10
z = norm.ppf(0.975)
S_values = np.array([1, 10, 50, 500, 1000, 5000, 10000])
u_mean, u_half, E_mean, E_half = [], [], [], []
for S in S_values:
    u_hat, E_hat = np.zeros(B), np.zeros(B)
    for b in range(B):
        x_omega = np.random.normal(mu, sigma, S)
        u_hat[b] = x_omega.mean()                       # argmin of (1/S) sum (u - x_omega)^2
        E_hat[b] = np.mean((u_hat[b] - x_omega)**2)
    u_mean.append(u_hat.mean()); u_half.append(z * u_hat.std(ddof=1) / np.sqrt(B))
    E_mean.append(E_hat.mean()); E_half.append(z * E_hat.std(ddof=1) / np.sqrt(B))
u_mean, u_half, E_mean, E_half = map(np.array, (u_mean, u_half, E_mean, E_half))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, m, h, ref, label in ((axs[0], E_mean, E_half, sigma**2, r'$\hat{\mathbb{E}}^S$ (optimal value)'),
                             (axs[1], u_mean, u_half, mu, r'$\hat{u}^S$ (solution)')):
    ax.fill_between(S_values, m - h, m + h, color='k', alpha=0.2)
    ax.semilogx(S_values, m, 'k-o', markerfacecolor='w')
    ax.axhline(ref, color='k', ls=':')
    ax.set_xlabel('$S$'); ax.set_ylabel(label); ax.grid(True)
plt.tight_layout()
plt.show()
