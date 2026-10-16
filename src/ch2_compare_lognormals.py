# %% [markdown]
# Log-normal RVs for several $(\mu,\sigma)$ (Victor Z and David Cole, UW-Madison, 2024).
# In `scipy.stats.lognorm`, the shape `s` is $\sigma$ and `scale` is $e^{\mu}$ (the median $\mathbb{M}_X$).

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

mu = [0, 0, 1]
sigma = [np.sqrt(1/5), np.sqrt(1), np.sqrt(1/10)]
lw = [0.5, 1.5, 2.5]
x = np.linspace(0.001, 5, 1000)

np.random.seed(0)
S = 1000
x_omega = np.random.lognormal(mean=0, sigma=2, size=S)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.grid(True)
plt.subplot(2, 2, 2)
plt.semilogy(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$ (log scale)', fontsize=14)
plt.grid(True)
for sub, fun, label in ((3, lognorm.pdf, '$f(x)$'), (4, lognorm.cdf, '$F(x)$')):
    plt.subplot(2, 2, sub)
    for k in range(len(mu)):
        plt.plot(x, fun(x, sigma[k], scale=np.exp(mu[k])), 'k-', linewidth=lw[k],
                 label=rf'$\mu={mu[k]},\ \sigma^2={sigma[k]**2:.2g}$')
    plt.grid(True)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(label, fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.show()
