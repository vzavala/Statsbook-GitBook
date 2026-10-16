# %% [markdown]
# Random and non-random samples (Victor Z, UW-Madison, 2024).
# A random sequence $X_\omega$ from $\mathcal{N}(0,1)$ (`x_random`) and a non-random one that
# discards draws above one (`x_biased`). The bias is the average deviation from
# $\mathbb{E}[X]=0$ (`E_X`).

# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
S = 100
E_X = 0
x_random = np.random.normal(0, 1, S)
draws = np.random.normal(0, 1, 2 * S)
x_biased = draws[draws <= 1][:S]           # discrimination mechanism: discard x > 1

print(f"bias (random)     = {np.mean(x_random - E_X):+.3f}")
print(f"bias (non-random) = {np.mean(x_biased - E_X):+.3f}")

fig, axs = plt.subplots(2, 1, figsize=(10, 7))
for ax, x_omega, title in ((axs[0], x_random, 'random'), (axs[1], x_biased, 'non-random')):
    ax.plot(np.arange(1, S + 1), x_omega, 'o', markerfacecolor='w', markeredgecolor='black')
    ax.axhline(E_X, color='k', linewidth=1.5)
    ax.set_title(title)
    ax.axis([0, S, -4, 4])
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$x_\omega$')
    ax.grid(True)
plt.tight_layout()
plt.show()
