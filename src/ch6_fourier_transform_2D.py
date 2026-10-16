# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# A 2D signal as a superposition of 2D waves (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024):
# $f(x_1,x_2)=\sin 2\pi\omega_1(x_1+x_2)-\sin 2\pi\omega_2(x_1+x_2)$ with $\omega_1=20$, $\omega_2=60$
# (`omega_1`, `omega_2`). The grid is fine enough (8 samples per period of the 60 Hz wave) to avoid aliasing.

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 0.25, 125)
X1, X2 = np.meshgrid(x, x)
omega_1, omega_2 = 20, 60
f1 = np.sin(2*np.pi*omega_1*(X1 + X2))
f2 = np.sin(2*np.pi*omega_2*(X1 + X2))
f = f1 - f2

fig = plt.figure(figsize=(11, 13))
for row, (Z, lab) in enumerate(((f, 'f'), (f1, 'f_1'), (f2, 'f_2'))):
    ax = fig.add_subplot(3, 2, 2*row + 1, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='gray', linewidth=0)
    ax.view_init(elev=30., azim=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel(f'${lab}(x_1,x_2)$')
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax2 = fig.add_subplot(3, 2, 2*row + 2)
    ax2.imshow(Z, cmap='gray', origin='lower', extent=[0, 0.25, 0, 0.25])
    ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
plt.tight_layout()
plt.show()
