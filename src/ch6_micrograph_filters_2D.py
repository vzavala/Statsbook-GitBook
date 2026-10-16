# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Applying 2D filters to a micrograph (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024).
# The image is a 2D signal $f$ (`f`); $f_1=f*g$ with a Sobel filter $g$ reveals intensity gradients and
# $f_2$ with a Gaussian filter ($\sigma=10$ pixels) keeps the coarse features.

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

f = 255 * color.rgb2gray(io.imread(path+'Data/droplet1.jpg'))

g = np.array([[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]]).T          # Sobel filter (horizontal gradient)
f_1 = convolve2d(f, g, mode='same')
f_2 = gaussian_filter(f, sigma=10)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, img, title in ((axs[0], f, '$f$'), (axs[1], np.abs(f_1), r'$|f*g|$ (Sobel)'),
                       (axs[2], f_2, 'Gaussian filter')):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
