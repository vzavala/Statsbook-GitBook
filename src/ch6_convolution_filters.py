# %% [markdown]
# **Additional example: a Gaussian filter as a low-pass filter** (Shiyi (Amy) Qin and Victor Z,
# UW-Madison, 2024). The signal $f$ mixes 30 Hz and 100 Hz sines. The FT of the Gaussian filter
# $g$ (window $N=100$, width factor $\alpha=8$) is large only at low frequencies, so $f*g$ keeps the
# 30 Hz component and removes the 100 Hz one. Python names: `f`, `g`, `phi`, `omega`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

L, Fs = 1000, 1000
t = np.arange(L) / Fs
f = np.sin(2*np.pi*30*t) + np.sin(2*np.pi*100*t)

N, alpha = 100, 8
sigma = (N - 1) / (2 * alpha)
j = np.arange(N) - (N - 1) / 2
g = np.exp(-j**2 / (2 * sigma**2))
g /= g.sum()                                  # unit area: keeps the amplitude of what passes

def power(z, n):
    m = np.abs(np.fft.fft(z, n)[:n // 2 + 1])
    m[1:-1] *= 2
    return m, Fs * np.arange(n // 2 + 1) / n

phi = fftconvolve(f, g, mode='same')
fig, axs = plt.subplots(3, 2, figsize=(11, 8))
for row, (s, lab) in enumerate(((f, 'f'), (g, 'g'), (phi, r'\varphi'))):
    tt = j / Fs if lab == 'g' else t
    axs[row, 0].plot(tt, s, 'k'); axs[row, 0].set_ylabel(f'${lab}(t)$')
    m, om = power(s, L)
    axs[row, 1].plot(om, m, 'k'); axs[row, 1].set_xlim(0, 150)
    axs[row, 1].set_ylabel(rf'$|\hat{{{lab}}}(\omega)|$')
for ax in axs.ravel():
    ax.grid(True)
axs[2, 0].set_xlabel('$t$'); axs[2, 1].set_xlabel(r'$\omega$')
plt.tight_layout()
plt.show()
