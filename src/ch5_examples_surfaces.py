# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Quadratic surfaces $SSE(\theta)=\theta^T\mathcal{H}\theta$ and the eigenvalues of the Hessian
# $\mathcal{H}$ (`H`) (Victor Z and Leo Gonzalez, UW-Madison, 2024): large positive eigenvalues give a
# sharp minimum, a zero eigenvalue a flat (non-unique) minimum, negative eigenvalues a maximum,
# and mixed signs a saddle point.

# %%
import numpy as np
import matplotlib.pyplot as plt

cases = [("sharp minimum", np.array([[2, 0], [0, 2]]), 1.0),
         ("flat minimum", np.array([[2, 2], [2, 2]]), 0.5),
         ("maximum", -np.array([[2, 0], [0, 2]]), 0.5),
         ("saddle point", np.array([[1, 2], [2, 1]]), 0.5)]
fig = plt.figure(figsize=(12, 10))
for k, (title, H, r) in enumerate(cases):
    theta_1, theta_2 = np.meshgrid(np.linspace(-r, r, 101), np.linspace(-r, r, 101))
    SSE = H[0, 0]*theta_1**2 + 2*H[0, 1]*theta_1*theta_2 + H[1, 1]*theta_2**2
    ax = fig.add_subplot(2, 2, k + 1, projection='3d')
    ax.plot_surface(theta_1, theta_2, SSE, cmap='Greys', alpha=0.9)
    ax.set_title(f"{title}: eig(H) = {np.round(np.linalg.eigvalsh(H), 2)}")
    ax.set_xlabel(r'$\theta_1$'); ax.set_ylabel(r'$\theta_2$'); ax.set_zlabel(r'$SSE(\theta)$')
plt.tight_layout()
plt.show()
