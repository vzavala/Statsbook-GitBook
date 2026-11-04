# %% [markdown]
# Convolution of a $5\times 5$ matrix $f$ with a $3\times 3$ filter $g$ (written for the book to
# accompany the slide example). `mode='valid'` keeps the window inside $f$ (a $3\times 3$ result);
# `mode='same'` lets the window stick out and ignores the outside entries (a $5\times 5$ result).
# The filter is symmetric, so convolution and correlation coincide.

# %% [markdown]
# **Note:** This notebook reproduces Example 6.17 (Section 6.3.5) of *Statistics for Chemical Engineers*: the convolution of a 5 x 5 matrix with a 3 x 3 filter.

# %%
import numpy as np
from scipy.signal import convolve2d

f = np.array([[1, 0, 1, 0, 0],
              [0, 1, 0, 1, 0],
              [1, 0, 1, 1, 1],
              [0, 0, 1, 1, 0],
              [0, 1, 1, 0, 0]])
g = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]])
print("phi = f*g (valid):\n", convolve2d(f, g, mode='valid'))
print("phi = f*g (same):\n", convolve2d(f, g, mode='same'))

# %% [markdown]
# **Note:** The 'valid' mode keeps the filter inside the matrix (3 x 3 result) and the 'same' mode keeps the size of the matrix (5 x 5 result).
