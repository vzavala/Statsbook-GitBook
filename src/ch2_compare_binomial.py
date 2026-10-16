# %% [markdown]
# Pdf and cdf of a binomial RV $X\sim\mathrm{Bi}(n,p)$, and convergence of
# $\mathcal{H}(N_y,N_t,n,p)$ to $\mathrm{Bi}(n,p)$ for large $N_t$
# (Victor Z and J E Umaña, UW-Madison, 2024).
# Python names: `n`, `p`, domain `D`, pdf `f`, cdf `F`, population `N_t`, `N_y`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, hypergeom

n, p = 10, 0.3
D = np.arange(0, n + 1)
f = binom.pmf(D, n, p)     # C(n,x) p^x (1-p)^(n-x)
F = binom.cdf(D, n, p)

plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.vlines(D, 0, f, colors='k', linestyles='--')
plt.scatter(D, f, marker='o', facecolors='w', edgecolors='black', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)

plt.subplot(222)
plt.step(D, F, where='post', color='black')
plt.scatter(D, F, marker='o', facecolors='w', edgecolors='black', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$F(x)$', fontsize=14)
plt.axis([-1, 10, 0, 1.02])

plt.subplot(223)
for p_k, mk in zip([0.95, 0.75, 0.5], ["o-", "x-", "d-"]):
    plt.plot(D, binom.pmf(D, n, p_k), mk, color='k', markerfacecolor='w', label=f'$p={p_k}$')
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.axis([0, 10, 0, 0.62])
plt.legend(loc='upper left', fontsize=10)

# hypergeometric H(N_y, N_t, n, p) approaches Bi(n, p) when N_t is large
plt.subplot(224)
p_H, N_t = 0.25, 1000
N_y = round(p_H * N_t)
D_H = np.arange(max(n + N_y - N_t, 0), min(n, N_y) + 1)
plt.plot(D_H, hypergeom.pmf(D_H, N_t, N_y, n), 'k--', linewidth=0.5, label=r'$\mathcal{H}(N_y,N_t=1000,n,p)$')
plt.plot(D, binom.pmf(D, n, p_H), 'ko', markerfacecolor='w', label=r'$\mathrm{Bi}(n,p)$')
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.axis([0, 10, 0, 0.3])
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()
