# %% [markdown]
# Chi-squared RVs $X\sim\chi^2(r)$ as Gamma RVs with $\beta=2$, $\alpha=r/2$
# (Victor Z and J E Umaña, UW-Madison, 2024).

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, chi2, norm

x = np.linspace(0, 4, 1000)
plt.figure(figsize=(10, 4))
for sub, fun, label in ((1, gamma.pdf, '$f(x)$'), (2, gamma.cdf, '$F(x)$')):
    plt.subplot(1, 2, sub)
    for r in (1, 2, 3, 4):
        plt.plot(x, fun(x, r/2, scale=2), linewidth=1.5, label=f'$r={r}$')   # = chi2(r)
    plt.grid(True)
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.axis([0, 4, 0, 1])
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# $\mathbb{Q}_Y(1-\alpha)$ for $Y\sim\chi^2(1)$ equals $\mathbb{Q}_Z(1-\alpha/2)^2$ for $Z\sim\mathcal{N}(0,1)$.

# %%
alpha = np.linspace(0.01, 1, 100)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(alpha, chi2.ppf(1 - alpha, df=1), 'b-', linewidth=1.5)
plt.ylabel(r'$\mathbb{Q}_Y(1-\alpha)$', fontsize=12)
plt.subplot(1, 2, 2)
plt.plot(alpha, norm.ppf(1 - alpha/2)**2, 'b-', linewidth=1.5)
plt.ylabel(r'$\mathbb{Q}_Z(1-\alpha/2)^2$', fontsize=12)
for k in (1, 2):
    plt.subplot(1, 2, k)
    plt.grid(True)
    plt.xlabel(r'$\alpha$', fontsize=12)
    plt.axis([0, 1, 0, 8])
plt.tight_layout()
plt.show()
