# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Flow cytometry data treated as a discrete 2D RV (Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $X_1$ (FSC), $X_2$ (SSC) | `x1`, `x2` | log front and side scatter of each droplet |
# | $S$ | `S` | number of droplets |
# | $\mathcal{D}_1=\mathcal{D}_2=\{L,M,H\}$ | `edges_1`, `edges_2` | bin edges of the three categories |
# | $\hat{f}(x_1,x_2)$ | `f_hat` | empirical joint pdf (rows: SSC, columns: FSC) |
# | $\hat{f}_1$, $\hat{f}_2$ | `f1_hat`, `f2_hat` | empirical marginal pdfs |
# | $\hat{f}(x_1\mid x_2)$, $\hat{f}(x_2\mid x_1)$ | `f_hat_1g2`, `f_hat_2g1` | empirical conditional pdfs |

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(path+'Data/flow_cytometer.dat')
S = len(data)
x1 = data[:, 0]   # FSC
x2 = data[:, 1]   # SSC
print(f"S = {S} droplets;  FSC in [{x1.min():.4f}, {x1.max():.4f}],  SSC in [{x2.min():.4f}, {x2.max():.4f}]")

# %% [markdown]
# Count droplets in each of the 9 categories $(x_1,x_2)\in\mathcal{D}_1\times\mathcal{D}_2$ and
# divide by $S$.

# %%
edges_1 = [4.2, 4.9, 5.6, 6.3]   # FSC: L, M, H
edges_2 = [3.5, 4.3, 5.1, 5.9]   # SSC: L, M, H
counts, _, _ = np.histogram2d(x2, x1, bins=(edges_2, edges_1))   # rows: SSC, cols: FSC
f_hat = counts / S

labels = ['L', 'M', 'H']
print("f_hat(x1, x2)  (rows SSC = H, M, L; columns FSC = L, M, H)")
for i in (2, 1, 0):
    print(f"  SSC={labels[i]}: " + "  ".join(f"{f_hat[i, j]:.4f}" for j in range(3)))
print("sum =", f_hat.sum())
i, j = np.unravel_index(np.argmax(f_hat), f_hat.shape)
print(f"Most likely category: (FSC, SSC) = ({labels[j]}, {labels[i]}),  f_hat = {f_hat[i, j]:.4f}")

plt.plot(x1, x2, 'ko', markerfacecolor='w', markersize=1)
for e in edges_2:
    plt.axhline(e, color='k', ls='--')
for e in edges_1:
    plt.axvline(e, color='k', ls='--')
plt.xlabel(r'$x_1=\log \mathrm{FSC}$', fontsize=14)
plt.ylabel(r'$x_2=\log \mathrm{SSC}$', fontsize=14)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.show()

# %% [markdown]
# Marginals add the joint pdf over the other variable; conditionals divide the joint pdf by a marginal:
# $\hat{f}(x_1|x_2)=\hat{f}(x_1,x_2)/\hat{f}_2(x_2)$.

# %%
f1_hat = f_hat.sum(axis=0)          # marginal of FSC (sum over SSC rows)
f2_hat = f_hat.sum(axis=1)          # marginal of SSC
with np.errstate(invalid='ignore', divide='ignore'):
    f_hat_1g2 = f_hat / f2_hat[:, None]     # each row sums to one
    f_hat_2g1 = f_hat / f1_hat[None, :]     # each column sums to one
print("f1_hat (FSC: L, M, H):", np.round(f1_hat, 4))
print("f2_hat (SSC: L, M, H):", np.round(f2_hat, 4))
print("f_hat(x1 | x2):\n", np.round(f_hat_1g2, 4))
print("f_hat(x2 | x1):\n", np.round(f_hat_2g1, 4))
