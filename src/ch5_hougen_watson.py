# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Nonlinear estimation for the Hougen-Watson model (Victor Z and Leo Gonzalez, UW-Madison, 2024).
#
# $$m_\omega(\theta)=\frac{\theta_0x_{2,\omega}-x_{3,\omega}/\theta_4}{1+\theta_1x_{1,\omega}+\theta_2x_{2,\omega}+\theta_3x_{3,\omega}}$$
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $x_{1,\omega},x_{2,\omega},x_{3,\omega}$ | `X[:, 0]`, `X[:, 1]`, `X[:, 2]` | partial pressures ($S=13$ experiments) |
# | $y_\omega$ | `y` | reaction rate |
# | $\theta=(\theta_0,\dots,\theta_4)$ | `theta[0]`, ..., `theta[4]` | parameters |
# | $m(\theta)$, $SSE(\theta)=\frac{1}{2}\lVert\mathbf{y}-\mathbf{m}(\theta)\rVert^2$ | `m(theta, X)`, `SSE(theta)` | model and objective |

# %% [part] The SSE function is highly nonlinear

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

data = np.loadtxt(path+'Data/hougenwatson.dat')
X, y = data[:, :3], data[:, -1]
S = len(y)

def m(theta, X):
    """Hougen-Watson model m(theta, x) for all rows of X."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (theta[0]*x2 - x3/theta[4]) / (1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3)

def SSE(theta):
    return 0.5 * np.sum((y - m(theta, X))**2)

# %% [markdown]
# $SSE(\theta)$ as we span $\theta_1,\theta_2\in[-2,2]$ (other parameters fixed at a reference value).
# Note the poles where the denominator of $m$ vanishes.

# %%
theta_ref = np.array([1.0, 0.05, 0.02, 0.1, 2.0])
grid = np.linspace(-2, 2, 201)
SSE_map = np.zeros((len(grid), len(grid)))
for i, t2 in enumerate(grid):
    for j, t1 in enumerate(grid):
        th = theta_ref.copy()
        th[1], th[2] = t1, t2
        SSE_map[i, j] = SSE(th)
plt.imshow(np.log10(SSE_map), extent=[-2, 2, -2, 2], origin='lower', cmap='gray_r', aspect='auto')
plt.colorbar(label=r'$\log_{10} SSE(\theta)$')
plt.xlabel(r'$\theta_1$', fontsize=14)
plt.ylabel(r'$\theta_2$', fontsize=14)
plt.show()

# %% [part] Good and bad initial guesses

# %% [markdown]
# Solve $\min_\theta \frac{1}{2}\lVert\mathbf{y}-\mathbf{m}(\theta)\rVert^2$ with a Levenberg-Marquardt
# (Newton-like) method, from $\theta=(1,\dots,1)$ and from $\theta=(-1,\dots,-1)$.
# (The slides use Matlab's `fitnlm`; `scipy.optimize.least_squares` plays the same role.)

# %%
fits = {}
for label, theta0 in (("theta0 = (1,...,1)", np.ones(5)), ("theta0 = (-1,...,-1)", -np.ones(5))):
    sol = least_squares(lambda th: y - m(th, X), theta0, method='lm', max_nfev=2000)
    fits[label] = sol.x
    print(f"{label}:  theta_hat = {np.round(sol.x, 2)},  SSE = {SSE(sol.x):.4f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, (label, th) in zip(axs, fits.items()):
    ax.plot([0, 15], [0, 15], 'k--', linewidth=1)
    ax.scatter(m(th, X), y, marker='o', color='w', edgecolor='k', s=50)
    ax.set_title(label)
    ax.set_xlabel(r'$\hat{y}$'); ax.set_ylabel('$y$')
    ax.axis([0, 15, 0, 15])
    ax.grid(True)
plt.tight_layout()
plt.show()
