# %% [markdown]
# Convolution of a $5\times 5$ matrix $f$ with a $3\times 3$ filter $g$ (written for the book to
# accompany the slide example). `mode='valid'` keeps the window inside $f$ (a $3\times 3$ result);
# `mode='same'` lets the window stick out and ignores the outside entries (a $5\times 5$ result).
# The filter is symmetric, so convolution and correlation coincide.

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
