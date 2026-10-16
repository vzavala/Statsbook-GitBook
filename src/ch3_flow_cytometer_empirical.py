# %% [markdown] [hide]
# **Setup.** Run this cell first: it locates the data files this notebook reads. On Google
# Colab it downloads them (the `Data/` and `Support/` folders) from the book's GitHub
# repository; on your computer, run the notebook from the `notebooks/` folder.

# %% [hide]
import os
import sys
path = ''
if 'google.colab' in sys.modules and not os.path.isdir('Data'):
    if not os.path.isdir('statsbook'):
        os.system('git clone -q --depth 1 --filter=blob:none --sparse '
                  'https://github.com/__GITHUB_REPO__.git statsbook')
        os.system('git -C statsbook sparse-checkout set notebooks/Data notebooks/Support')
    path = 'statsbook/notebooks/'

# %% [markdown]
# Empirical joint pdf of flow cytometry data (Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $x_{\omega,1}$ (FSC), $x_{\omega,2}$ (SSC) | `x1`, `x2` | realizations |
# | $\hat{\mathcal{D}}$ | `x1.min()`, ... | empirical domain (bounding box of the data) |
# | $\Delta x_1=\Delta x_2$ | `dx` | bin size (0.05) |
# | $\hat{f}(x_1,x_2)$ | `f_hat` | empirical joint pdf |

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(path+'Data/flow_cytometer.dat')
S = len(data)
x1, x2 = data[:, 0], data[:, 1]      # FSC, SSC
print(f"D_hat = {{{x1.min():.4f} <= X1 <= {x1.max():.4f}  &  {x2.min():.4f} <= X2 <= {x2.max():.4f}}}")

# %% [markdown]
# $\hat{f}(x_1,x_2)\approx\frac{1}{\Delta x_1\Delta x_2}\hat{\mathbb{P}}(x_1\leq X_1\leq x_1+\Delta x_1\ \&\ x_2\leq X_2\leq x_2+\Delta x_2)$

# %%
dx = 0.05
edges_1 = np.arange(x1.min(), x1.max() + dx, dx)
edges_2 = np.arange(x2.min(), x2.max() + dx, dx)
counts, _, _ = np.histogram2d(x1, x2, bins=[edges_1, edges_2])
f_hat = counts / (S * dx * dx)
print(f"{len(edges_1)-1} x {len(edges_2)-1} bins;  integral of f_hat = {f_hat.sum() * dx * dx:.3f}")

fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(221)
ax1.plot(x1, x2, 'ko', markerfacecolor='w', markersize=1)
ax1.set_title('observations')
ax2 = fig.add_subplot(222)
im = ax2.imshow(f_hat.T, extent=[edges_1[0], edges_1[-1], edges_2[0], edges_2[-1]],
                cmap='gray_r', origin='lower', aspect='auto')
fig.colorbar(im, ax=ax2, label=r'$\hat{f}(x_1,x_2)$')
ax2.set_title('density of observations in bins')
for ax in (ax1, ax2):
    ax.set_xlabel(r'$x_1=\log \mathrm{FSC}$')
    ax.set_ylabel(r'$x_2=\log \mathrm{SSC}$')
ax3 = fig.add_subplot(212, projection='3d')
X1, X2 = np.meshgrid(edges_1[:-1], edges_2[:-1], indexing='ij')
ax3.bar3d(X1.ravel(), X2.ravel(), 0, dx, dx, f_hat.ravel(), color='gray', shade=True)
ax3.set_xlabel(r'$x_1$')
ax3.set_ylabel(r'$x_2$')
ax3.set_zlabel(r'$\hat{f}(x_1,x_2)$')
plt.tight_layout()
plt.show()

# %% [markdown]
# Empirical marginal pdfs $\hat{f}_1(x_1)$ and $\hat{f}_2(x_2)$.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(x1, bins=edges_1, density=True, edgecolor='black', facecolor='none')
axs[0].set_xlabel(r'$x_1=\log \mathrm{FSC}$')
axs[0].set_ylabel(r'$\hat{f}_1(x_1)$')
axs[1].hist(x2, bins=edges_2, density=True, edgecolor='black', facecolor='none')
axs[1].set_xlabel(r'$x_2=\log \mathrm{SSC}$')
axs[1].set_ylabel(r'$\hat{f}_2(x_2)$')
plt.tight_layout()
plt.show()
