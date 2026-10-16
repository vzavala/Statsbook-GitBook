# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# 1D convolutions with moving-average and Gaussian filters (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $f_j$ | `f` | signal |
# | $g_j$ | `g` | filter (weights over a window of $N$ entries) |
# | $\varphi=f*g$ | `phi` | convolution $\varphi_k=\sum_j g_j f_{k+j}$ |
# | $N$, $\alpha$, $\sigma=(N-1)/(2\alpha)$ | `N`, `alpha`, `sigma` | window length, width factor, Gaussian SD |
#
# The Gaussian weights are $g_j=e^{-j^2/(2\sigma^2)}$, $j=-(N-1)/2,\dots,(N-1)/2$
# (this reproduces the slide values, e.g. $g=(0.13,0.6,1,0.6,0.13)$ for $N=5$, $\alpha=2$).

# %% [part] Moving average filter for a pulse

# %%
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter_1d(N, alpha):
    sigma = (N - 1) / (2 * alpha)
    j = np.arange(N) - (N - 1) / 2
    return np.exp(-j**2 / (2 * sigma**2)), sigma

print("N=5, alpha=2:", np.round(gaussian_filter_1d(5, 2)[0], 2))

def show(t, f, g_padded, phi):
    fig, axs = plt.subplots(3, 1, figsize=(8, 7))
    axs[0].step(t, f, 'k'); axs[0].set_ylabel('$f$')
    axs[1].plot(t, g_padded, 'k'); axs[1].set_ylabel('$g$')
    axs[2].plot(t, phi, 'k'); axs[2].set_ylabel(r'$\varphi=f*g$')
    for ax in axs:
        ax.grid(True)
    axs[2].set_xlabel('$t$')
    plt.tight_layout()
    plt.show()

N, pad = 20, 40
f = np.concatenate([np.zeros(pad), np.ones(N), np.zeros(pad)])     # rectangular pulse (100 entries)
t = np.arange(len(f))
g = np.ones(N) / N                                                 # moving average, weights 1/N
phi = np.convolve(f, g, mode='same')
show(t, f, np.concatenate([np.zeros(pad), g, np.zeros(pad)]), phi)

# %% [part] Gaussian filter for a pulse

# %%
g, sigma = gaussian_filter_1d(N=20, alpha=4)
print(f"sigma = {sigma:.2f}")
phi = np.convolve(f, g, mode='same')
show(t, f, np.concatenate([np.zeros(pad), g, np.zeros(pad)]), phi)

# %% [part] Gaussian filters for a noisy signal

# %%
Fs = 1000                                  # sampling frequency [Hz]
t = np.arange(500) / Fs
np.random.seed(0)
f = np.sin(2*np.pi*30*t) + np.sin(2*np.pi*60*t) + np.random.normal(0, 1, len(t))

fig, axs = plt.subplots(3, 1, figsize=(8, 7))
axs[0].plot(t, f, 'k'); axs[0].set_ylabel('$f$')
for ax, alpha in ((axs[1], 1), (axs[2], 4)):
    g, sigma = gaussian_filter_1d(N=4, alpha=alpha)
    ax.plot(t, np.convolve(f, g, mode='same'), 'k')
    ax.set_ylabel(r'$f*g$')
    ax.set_title(f"g = {np.round(g, 4)}  (sigma = {sigma:.3f})", fontsize=10)
for ax in axs:
    ax.grid(True)
axs[2].set_xlabel('$t$')
plt.tight_layout()
plt.show()
