# %% [markdown]
# Fourier transform of a periodic signal (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $f(t)=\sin(2\pi\omega_1t)+\sin(2\pi\omega_2t)$ | `f`, `omega_1`, `omega_2` | signal (30 and 60 Hz) |
# | $\hat{f}_k=\sum_j f_je^{-i2\pi kj/N}$ | `f_hat = np.fft.fft(f)` | discrete FT |
# | $\omega_k$, $\lvert\hat{f}_k\rvert$ | `omega`, `power(f_hat)` | frequencies and one-sided magnitude |
# | $\mathcal{F}\{f*g\}=\mathcal{F}\{f\}\cdot\mathcal{F}\{g\}$ | `phi_hat = f_hat * g_hat` | convolution theorem |

# %%
import numpy as np
import matplotlib.pyplot as plt

L, Fs = 500, 1000                       # number of samples, sampling frequency [Hz]
t = np.arange(L) / Fs
omega = Fs * np.arange(L // 2 + 1) / L  # nonnegative frequencies

def power(z_hat):
    """One-sided magnitude |z_hat_k| for k >= 0 (uses |z_hat_k| = |z_hat_-k|)."""
    m = np.abs(z_hat[:L // 2 + 1]).copy()
    m[1:-1] *= 2
    return m

omega_1, omega_2 = 30, 60
f1, f2 = np.sin(2*np.pi*omega_1*t), np.sin(2*np.pi*omega_2*t)
f = f1 + f2
f_hat = np.fft.fft(f)

fig, axs = plt.subplots(4, 1, figsize=(8, 9))
for ax, s, lab in zip(axs[:3], (f, f1, f2), ('$f(t)$', '$f_1(t)$', '$f_2(t)$')):
    ax.plot(t, s, 'k'); ax.set_ylabel(lab); ax.set_xlabel('$t$')
axs[3].plot(omega, power(f_hat), 'k')
axs[3].set_xlim(0, 70)
axs[3].set_ylabel(r'$|\hat{f}(\omega)|$'); axs[3].set_xlabel(r'$\omega$')
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()
print("dominant frequencies:", omega[np.argsort(power(f_hat))[-2:]])

# %% [markdown]
# Convolution through the frequency domain: multiply $\hat{f}$ by the FT of a 30 Hz sine $g$
# and invert. Only the 30 Hz content survives.

# %%
g = np.sin(2*np.pi*30*t)
g_hat = np.fft.fft(g)
phi_hat = f_hat * g_hat
phi = np.fft.ifft(phi_hat).real

fig, axs = plt.subplots(3, 2, figsize=(11, 7))
for row, (s, s_hat, lab) in enumerate(((f, f_hat, 'f'), (g, g_hat, 'g'), (phi, phi_hat, r'\varphi'))):
    axs[row, 0].plot(t, s, 'k'); axs[row, 0].set_ylabel(f'${lab}(t)$')
    axs[row, 1].plot(omega, power(s_hat), 'k'); axs[row, 1].set_xlim(0, 70)
    axs[row, 1].set_ylabel(rf'$|\hat{{{lab}}}(\omega)|$')
for ax in axs.ravel():
    ax.grid(True)
plt.tight_layout()
plt.show()
