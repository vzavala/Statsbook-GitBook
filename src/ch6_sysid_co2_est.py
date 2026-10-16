# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# **Additional example: system identification with Fourier transforms** (Victor M Zavala, UW-Madison, 2024).
# A building has CO$_2$ input $u(t)$ (occupants) and indoor CO$_2$ concentration $x(t)$ above 400 ppm.
# We compute the magnitude of the Fourier transforms $|\hat{u}(\omega)|$ and $|\hat{x}(\omega)|$ by numerical
# integration, form the amplitude ratio $AR(\omega)=|\hat{x}(\omega)|/|\hat{u}(\omega)|$, and fit a first-order
# model $AR(\omega)=K/\sqrt{(\tau\omega)^2+1}$ by least squares.
#
# | Symbol | Python | Meaning |
# |---|---|---|
# | $t$, $u(t)$, $x(t)$ | `t`, `u`, `x` | time [min], input, output (ppm above 400) |
# | $\omega$ | `omega` | frequency [rad/min] |
# | $\lvert\hat{u}(\omega)\rvert$, $\lvert\hat{x}(\omega)\rvert$ | `mag_u`, `mag_x` | spectra |
# | $K$, $\tau$ | `K`, `tau` | gain and time constant |

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

data = np.loadtxt(path+'Data/data_sysid_co2.dat')
t, u, x = data[:, 0], data[:, 1], data[:, 2] - 400

omega = np.logspace(-3, 0, 100)

def spectrum(z):
    """|z_hat(omega)| = |integral z(t) e^{-i omega t} dt| by the trapezoidal rule."""
    return np.array([np.abs(np.trapz(z * np.exp(-1j * w * t), t)) for w in omega])

mag_u, mag_x = spectrum(u), spectrum(x)
AR = mag_x / mag_u

def residual(theta):
    K, tau = theta
    return AR - K / np.sqrt((tau * omega)**2 + 1)

K, tau = least_squares(residual, [1, 1], bounds=([0, 0], [100, 100]), max_nfev=5000).x
print(f"K = {K:.3f},  tau = {tau:.2f} min")

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
axs[0].semilogx(omega, mag_u, 'k'); axs[0].set_ylabel(r'$|\hat{u}(\omega)|$')
axs[1].semilogx(omega, mag_x, 'k'); axs[1].set_ylabel(r'$|\hat{x}(\omega)|$')
axs[2].semilogx(omega, AR, 'o', markerfacecolor='w', color='0.5', label='empirical')
axs[2].semilogx(omega, K / np.sqrt((tau * omega)**2 + 1), 'k-', label='first-order model')
axs[2].set_ylabel(r'$AR(\omega)$'); axs[2].legend()
for ax in axs:
    ax.set_xlabel(r'$\omega$ [rad/min]'); ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Physical parameters for a building of volume $V$: air flow $Q=V/\tau$ and generation $G=KQ$.

# %%
V = 10000                 # building volume [cubic feet]
Q = V / tau
G = K * Q
print(f"Q = {Q:.1f},  G = {G:.1f}")
