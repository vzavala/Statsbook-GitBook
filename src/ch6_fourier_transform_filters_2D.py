# %% [markdown]
# 2D Fourier transforms of a uniform (box) filter and of a Gaussian filter
# (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024). We show each filter $g$ (`g`), the magnitude
# $\log(1+|\hat{g}(\omega_1,\omega_2)|)$ (centered with `fftshift`), and the same as a 3D surface.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter

L = 50
g_box = np.zeros((L, L))
g_box[19:30, 19:30] = 1                     # 2D step (uniform) filter
g_gauss = np.zeros((L, L))
g_gauss[L // 2, L // 2] = 1
g_gauss = gaussian_filter(g_gauss, sigma=1)  # 2D Gaussian filter

fig = plt.figure(figsize=(11, 13))
for col, (g, title) in enumerate(((g_box, 'uniform filter'), (g_gauss, 'Gaussian filter'))):
    g_hat = fftshift(np.log(1 + np.abs(fft2(g))))
    ax = fig.add_subplot(3, 2, col + 1)
    ax.imshow(g[10:40, 10:40], cmap='gray_r')
    ax.set_title(title); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax = fig.add_subplot(3, 2, col + 3)
    ax.imshow(g_hat, cmap='gray_r')
    ax.set_xlabel(r'$\omega_1$'); ax.set_ylabel(r'$\omega_2$')
    ax = fig.add_subplot(3, 2, col + 5, projection='3d')
    W1, W2 = np.meshgrid(np.arange(L), np.arange(L))
    ax.plot_surface(W1, W2, g_hat, cmap='gray_r', linewidth=0)
    ax.set_xlabel(r'$\omega_1$'); ax.set_ylabel(r'$\omega_2$'); ax.set_zlabel(r'$|\hat{g}|$')
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
plt.tight_layout()
plt.show()
