# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Examples of multivariate data sets (Victor Z, UW-Madison, 2024): realizations
# $x_\omega=(x_{\omega,1},x_{\omega,2})$ (`x_omega[:, 0]`, `x_omega[:, 1]`) that are
# anticorrelated, uncorrelated, and nonlinearly related, and a 3D data set.

# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
S = 1000
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, Sigma, title in ((axs[0], [[1, -0.9], [-0.9, 1]], 'anticorrelated'),
                         (axs[1], [[1, 0], [0, 1]], 'uncorrelated')):
    x_omega = np.random.multivariate_normal([0, 0], Sigma, S)
    ax.plot(x_omega[:, 0], x_omega[:, 1], 'ko', markerfacecolor='w', markersize=3)
    ax.set_title(title)
x1 = np.linspace(0, 5, 200)
x2 = x1**2 + np.random.normal(0, 1, 200)
axs[2].plot(x2, x1, 'ko', markerfacecolor='w', markersize=3)
axs[2].set_title('nonlinear relationship')
for ax in axs:
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.grid(True)
plt.tight_layout()
plt.show()

# %%
x1, x2 = np.meshgrid(np.arange(0, 11, 0.5), np.arange(0, 11, 0.5))
x3 = 2 * x1**2 + 2 * x2**2 + 20 * np.random.normal(0, 1, x1.shape)
ax = plt.figure(figsize=(6, 5)).add_subplot(projection='3d')
ax.scatter(x1, x2, x3, c='w', edgecolors='k', marker='o')
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_zlabel('$x_3$', fontsize=14)
plt.show()
