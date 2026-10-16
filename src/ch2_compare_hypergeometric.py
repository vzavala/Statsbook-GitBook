# %% [markdown]
# Pdf and cdf of a hypergeometric RV $X\sim\mathcal{H}(N_y,N_t,n,p)$ with $N_t=500$,
# $N_y=50$, $n=50$ (Victor Z and David Cole, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $N_t$, $N_y$ | `N_t`, `N_y` | population size and number of type-$Y$ items |
# | $n$ | `n` | sample size |
# | $p=N_y/N_t$ | `p` | fraction of type-$Y$ items |
# | $\mathcal{D}$, $f(x)$, $F(x)$ | `D`, `f`, `F` | domain, pdf, cdf |

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import hypergeom

N_t, N_y, n = 500, 50, 50
p = N_y / N_t
D = np.arange(max(0, n + N_y - N_t), min(n, N_y) + 1)    # domain

# f(x) = C(N_y, x) C(N_t - N_y, n - x) / C(N_t, n)
f = comb(N_y, D) * comb(N_t - N_y, n - D) / comb(N_t, n)
F = np.cumsum(f)

np.random.seed(0)
S = 1000
x_omega = hypergeom.rvs(N_t, N_y, n, size=S)   # scipy order: (population, type-Y, sample)

print(f"mode = {D[np.argmax(f)]},  E_X = n p = {n*p}")

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, (1, 2))
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.grid(True)
plt.axis([0, S, 0, 15])

plt.subplot(2, 2, 3)
plt.vlines(D, 0, f, colors='k', linestyles='--')
plt.scatter(D, f, facecolor='w', edgecolor='k', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.axis([-1, 20, 0, max(f) * 1.05])

plt.subplot(2, 2, 4)
plt.step(D, F, 'k-', where='post')
plt.scatter(D, F, facecolor='w', edgecolor='k', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$F(x)$', fontsize=14)
plt.axis([-1, 20, -0.02, 1.02])
plt.tight_layout()
plt.show()
