# %% [markdown]
# Gaussian RVs $X\sim\mathcal{N}(\mu,\sigma^2)$ for different $\mu,\sigma$
# (Victor Z and J E Umaña, UW-Madison, 2024).
# Python names: `mu`, `sigma`, realizations `x_omega`. Note that `scipy.stats.norm`
# takes the standard deviation $\sigma$ (not $\sigma^2$) as `scale`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = [0, 0, 1]
sigma = [np.sqrt(1/5), np.sqrt(1), np.sqrt(1/10)]
lw = [0.5, 1.5, 2.5]
x = np.linspace(-5, 5, 1000)

np.random.seed(0)
S = 1000
x_omega = np.random.normal(0, 1, S)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, (1, 2))
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.axis([0, S, -4, 4])
plt.grid(True)
plt.subplot(2, 2, 3)
for k in range(len(mu)):
    plt.plot(x, norm.pdf(x, mu[k], sigma[k]), 'k-', linewidth=lw[k],
             label=rf'$\mu={mu[k]},\ \sigma^2={sigma[k]**2:.3g}$')
plt.grid(True)
plt.legend(loc='upper left', fontsize=10)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.axis([-5, 5, 0, 1.5])
plt.subplot(2, 2, 4)
for k in range(len(mu)):
    plt.plot(x, norm.cdf(x, mu[k], sigma[k]), 'k-', linewidth=lw[k])
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$F(x)$', fontsize=14)
plt.axis([-5, 5, 0, 1.02])
plt.show()

# %% [markdown]
# Levels $\mu+k\sigma$, $k=0,1,2,3$, and their probabilities
# $\mathbb{P}(X\leq\mu+k\sigma)=F_Z(k)$ for the standard normal $Z$.

# %%
mu, sigma = 0, 1
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(x, norm.pdf(x, mu, sigma), 'k', linewidth=1.5)
axs[1].plot(x, norm.cdf(x, mu, sigma), 'k', linewidth=1.5)
for k in range(4):
    q = mu + k * sigma
    axs[0].axvline(q, color='k', ls='--', lw=1)
    axs[1].axvline(q, color='k', ls='--', lw=1)
    axs[1].axhline(norm.cdf(q), color='k', ls='--', lw=1)
    axs[1].scatter(q, norm.cdf(q), facecolors='w', edgecolors='k', zorder=3)
    print(f"P(Z <= {k}) = F_Z({k}) = {norm.cdf(k):.3f}")
axs[0].set_ylabel('$f(x)$', fontsize=14)
axs[1].set_ylabel('$F(x)$', fontsize=14)
axs[1].set_xlabel('$x$', fontsize=14)
for ax in axs:
    ax.grid(True)
plt.show()
