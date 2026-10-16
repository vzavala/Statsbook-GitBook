# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Image compression with the SVD (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2020).
# The grayscale image is a matrix $\mathbf{X}$ (`X`); we keep the first $n$ terms of
# $\mathbf{X}=\sum_k s_k\mathbf{u}_k\mathbf{v}_k^T$ for $n=1,3,10,100$.

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
plt.xlabel('$n$'); plt.ylabel(r'$\lVert\mathbf{X}-\mathbf{X}_n\rVert_F^2$')
plt.grid(True)
plt.show()
