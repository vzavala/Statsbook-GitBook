# %% [markdown]
# Pdf and cdf of a discrete uniform RV $X\sim\mathcal{U}(a,b)$ with $a=-1,b=1$
# (Victor Z and David Cole, UW-Madison, 2024).
# Python names: `a`, `b`, `n` $=b-a+1$, domain `D`, pdf `f`, cdf `F`, realizations `x_omega`.

# %%
import numpy as np
import matplotlib.pyplot as plt

a, b = -1, 1
D = np.arange(a, b + 1)         # domain {a, ..., b}
n = b - a + 1
f = np.ones(n) / n              # f(x) = 1/n

np.random.seed(0)
S = 100
x_omega = np.random.randint(a, b + 1, S)   # realizations

x = np.linspace(-2, 2, 1000)
F = np.array([np.sum(f[D <= x_k]) for x_k in x])   # F(x) = (x - a + 1)/n on D

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, (1, 2))
plt.plot(x_omega, 'ko', markerfacecolor='w', markersize=4)
plt.xlabel(r'Outcome $\omega$', fontsize=14)
plt.ylabel(r'$x_\omega$', fontsize=14)
plt.grid(True)
plt.yticks([-1, 0, 1])
plt.axis([0, S, -1.5, 1.5])

plt.subplot(2, 2, 3)
plt.vlines(D, 0, f, colors='k', linestyles='--')
plt.scatter(D, f, marker='o', facecolors='w', edgecolors='black', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$f(x)$', fontsize=14)
plt.axis([-1.5, 1.5, -0.02, 0.52])

plt.subplot(2, 2, 4)
plt.step(x, F, where='post', color='black')
plt.scatter(D, np.cumsum(f), marker='o', facecolors='w', edgecolors='black', zorder=3)
plt.grid(True)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$F(x)$', fontsize=14)
plt.axis([-1.5, 1.5, -0.02, 1.02])
plt.tight_layout()
plt.show()
