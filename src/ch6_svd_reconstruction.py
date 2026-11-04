# %% [markdown] [hide]
# **Setup.** Run this cell first: it locates the data files this notebook reads. On Google
# Colab it downloads them (the `Data/` and `Support/` folders) from the book's GitHub
# repository; on your computer, run the notebook from the folder that contains `Data/`.

# %% [hide]
import os
import sys
path = ''
if 'google.colab' in sys.modules and not os.path.isdir('Data'):
    if not os.path.isdir('statsbook'):
        os.system('git clone -q --depth 1 --filter=blob:none --sparse '
                  'https://github.com/__GITHUB_REPO__.git statsbook')
        os.system('git -C statsbook sparse-checkout set __NOTEBOOK_DIR__/Data __NOTEBOOK_DIR__/Support')
    path = 'statsbook/__NOTEBOOK_DIR__/'

# %% [markdown]
# Image compression with the SVD (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2020).
# The grayscale image is a matrix $\mathbf{X}$ (`X`); we keep the first $n$ terms of
# $\mathbf{X}=\sum_k s_k\mathbf{u}_k\mathbf{v}_k^T$ for $n=1,3,10,100$.

# %% [markdown]
# **Note:** This notebook reproduces Example 6.6 (Section 6.2) of *Statistics for Chemical Engineers*: compressing a micrograph by keeping only some of its singular values.

# %%
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io

X = 255 * color.rgb2gray(io.imread(path+'Data/droplet1.jpg'))
print("image size:", X.shape, "  rank:", np.linalg.matrix_rank(X))
U, s, Vt = np.linalg.svd(X, full_matrices=False)

n_values = [1, 3, 10, 100]
errors = []
fig, axs = plt.subplots(1, 5, figsize=(16, 4))
axs[0].imshow(X, cmap='gray'); axs[0].set_title('original')
for ax, n in zip(axs[1:], n_values):
    X_n = (U[:, :n] * s[:n]) @ Vt[:n]            # sum of the first n terms
    errors.append(np.sum((X - X_n)**2))
    ax.imshow(X_n, cmap='gray'); ax.set_title(f'$n={n}$')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()

plt.semilogy(n_values, errors, 'ko-')
plt.xlabel('$n$'); plt.ylabel(r'$\Vert\mathbf{X}-\mathbf{X}_n\Vert_F^2$')
plt.grid(True)
plt.show()

# %% [markdown]
# **Note:** As in the book, a few singular values reproduce the image well: the remaining ones carry details our eyes cannot perceive.
