# %% [markdown] [hide]
# **Setup.** Run this cell first: it locates the data files this notebook reads. On Google
# Colab it downloads them (the `Data/` and `Support/` folders) from the book's GitHub
# repository; on your computer, run the notebook from the `notebooks/` folder.

# %% [hide]
import os
import sys
path = ''
if 'google.colab' in sys.modules and not os.path.isdir('Data'):
    if not os.path.isdir('statsbook'):
        os.system('git clone -q --depth 1 --filter=blob:none --sparse '
                  'https://github.com/__GITHUB_REPO__.git statsbook')
        os.system('git -C statsbook sparse-checkout set notebooks/Data notebooks/Support')
    path = 'statsbook/notebooks/'

# %% [markdown]
# Kriging (Gaussian process) model for the Gibbs reactor (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $X=(P,T)$, $Y=C$ | `X`, `y` | inputs (pressure, temperature) and conversion |
# | $k(x,x',\gamma)$, $\hat{\gamma}$, $\sigma^2$ | `kernel`, `gp.kernel_` | Matern kernel plus noise term, and their MLE values |
# | $m(x)$, $\sigma(x)$ | `m_x`, `sigma_x` | predictive mean and SD |
# | $\epsilon_\omega$ | `eps_gp`, `eps_lin` | residuals of kriging and linear models |

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression

data = np.vstack((np.loadtxt(path+'Data/gibbs_hightemp_class.dat'),
                  np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')))
S = len(data)
np.random.seed(1)
X = np.column_stack((data[:, 1], data[:, 0])) + np.random.rand(S, 2) * 0.1   # (P, T)
y = data[:, 2] + np.random.rand(S) * 0.1                                     # C

# %% [markdown]
# Fit $\hat{\gamma}$ by maximizing $\log f(\mathbf{y}|\gamma)$, then predict with
# $m(x)=\mathbf{k}(x)^T\mathbf{C}^{-1}\mathbf{y}$ and $\sigma^2(x)=c(x)-\mathbf{k}(x)^T\mathbf{C}^{-1}\mathbf{k}(x)$.
# A linear model is fitted for comparison.

# %%
kernel = Matern(length_scale=1, nu=1.5) + WhiteKernel(noise_level=0.01)   # K(gamma) + sigma^2 I
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0).fit(X, y)
m_x, sigma_x = gp.predict(X, return_std=True)            # sigma_x includes the noise term
lin = LinearRegression().fit(X, y)
eps_gp, eps_lin = y - m_x, y - lin.predict(X)
print("fitted kernel:", gp.kernel_)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, title in ((axs[0], m_x, 'kriging'), (axs[1], lin.predict(X), 'linear')):
    ax.plot(y, pred, 'ko', markerfacecolor='w', markersize=3)
    ax.plot([0, 1.1], [0, 1.1], 'k-')
    ax.set_title(title); ax.set_xlabel('$y$'); ax.set_ylabel(r'$\hat{y}$')
    ax.axis([0, 1.1, 0, 1.1]); ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Compare the residuals through the empirical cdf of $|\epsilon|$.

# %%
for eps, style, label in ((eps_gp, 'k-', 'kriging'), (eps_lin, 'k--', 'linear')):
    e = np.sort(np.abs(eps))
    plt.step(e, np.arange(1, S + 1) / S, style, where='post', label=label)
plt.xlim(0, 0.1)
plt.xlabel('$t$'); plt.ylabel(r'$\hat{\mathbb{P}}(|\epsilon|\leq t)$')
plt.legend(loc='lower right'); plt.grid(True)
plt.show()

# %% [markdown]
# 95% confidence intervals $m(x_\omega)\pm 1.96\,\sigma(x_\omega)$ for the first 50 observations.

# %%
w = np.arange(1, 51)
plt.errorbar(w, m_x[:50], yerr=1.96 * sigma_x[:50], fmt='k.', markersize=6, label=r'$m(x_\omega)\pm1.96\sigma(x_\omega)$')
plt.scatter(w, y[:50], color='black', facecolors='none', label=r'$y_\omega$')
plt.xlabel(r'Observation $\omega$'); plt.ylabel('$y$')
plt.legend(); plt.grid(True)
plt.show()
