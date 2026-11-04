# %% [markdown]
# Weight matrix of a 2D Gaussian filter (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024):
# the kernel $g(x,y)$ is the pdf of $\mathcal{N}(0,\Sigma)$ with $\Sigma=\sigma^2 I$, $\sigma=\sqrt{2}$,
# evaluated on a $10\times 10$ window (`g`, scaled by 100).

# %% [markdown]
# **Note:** This notebook reproduces Example 6.18 (Section 6.3.5) of *Statistics for Chemical Engineers*: a 2D Gaussian filter on a 10 x 10 window with sigma = sqrt(2).

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

sigma2 = 2.0
x = np.linspace(-3, 3, 10)
X1, X2 = np.meshgrid(x, x)
g = multivariate_normal([0, 0], sigma2 * np.eye(2)).pdf(np.dstack((X1, X2)))
print(np.round(100 * g).astype(int))

plt.figure(figsize=(5, 5))
plt.imshow(100 * g, cmap='gray_r', interpolation='nearest', origin='lower')
plt.xticks([]); plt.yticks([])
plt.colorbar(fraction=0.046)
plt.show()

# %% [markdown]
# **Note:** As in the book, the filter is symmetric around its center, where it is largest, and decays quickly away from it.
