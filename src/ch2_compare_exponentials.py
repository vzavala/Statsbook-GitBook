# %% [markdown]
# Exponential RVs $X\sim\mathrm{Exp}(\beta)$ for $\beta=1,2,4$ (Victor Z and J E Umaña, UW-Madison, 2024).
# `scipy.stats.expon` uses `scale` $=\beta$ (the intensity is $\eta=1/\beta$).
# The dashed line marks $F(\beta)=1-e^{-1}=0.63$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

beta = [1, 2, 4]
lw = [0.5, 1, 1.5]
x = np.linspace(0, 4, 1000)

np.random.seed(0)
S = 1000
x_omega = np.random.exponential(scale=4, size=S)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, (1, 2))
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.axis([0, S, 0, 36])
plt.grid(True)
plt.subplot(2, 2, 3)
for k in range(len(beta)):
    plt.plot(x, expon.pdf(x, scale=beta[k]), 'k', linewidth=lw[k])
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.axis([0, 4, 0, 1])
plt.subplot(2, 2, 4)
for k in range(len(beta)):
    plt.plot(x, expon.cdf(x, scale=beta[k]), 'k', linewidth=lw[k], label=rf'$\beta={beta[k]}$')
plt.axhline(1 - np.exp(-1), color='k', ls='--')
plt.grid(True)
plt.legend(loc='lower right', fontsize=10)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$F(x)$', fontsize=14)
plt.axis([0, 4, 0, 1])
plt.show()
