# %% [markdown]
# Gamma RVs $X\sim\mathrm{Gamma}(\alpha,\beta)$ (Victor Z and J E Umaña, UW-Madison, 2024).
# `scipy.stats.gamma` uses shape `a` $=\alpha$ and `scale` $=\beta$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

alpha = [1, 2, 1, 2]
beta = [1, 1, 0.5, 0.5]
lw = [0.5, 1, 1.5, 2]
x = np.linspace(0, 4, 1000)

np.random.seed(0)
S = 1000
plt.figure(figsize=(10, 8))
for sub, a_k in ((1, 1), (2, 2)):
    x_omega = gamma.rvs(a_k, scale=1, size=S)
    plt.subplot(2, 2, sub)
    plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
    plt.xlabel(r'Outcome $\omega$', fontsize=14)
    plt.ylabel(r'$x_\omega$', fontsize=14)
    plt.title(rf'$\alpha={a_k},\ \beta=1$')
    plt.grid(True)
    plt.axis([0, S, 0, 10])
for sub, fun, label in ((3, gamma.pdf, '$f(x)$'), (4, gamma.cdf, '$F(x)$')):
    plt.subplot(2, 2, sub)
    for k in range(len(alpha)):
        plt.plot(x, fun(x, a=alpha[k], scale=beta[k]), 'k', linewidth=lw[k],
                 label=rf'$\alpha={alpha[k]},\ \beta={beta[k]}$')
    plt.grid(True)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel(label, fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()
