# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Gibbs reactor example (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $X$, $x_\omega$ | `x_omega` | reactor pressure (RV) and its realizations [bar] |
# | $Y(u)$, $Y(u')$, $y_\omega$ | `y_low`, `y_high` | conversion (RV) and its realizations at $u$, $u'$ [%] |
# | $\mathcal{S}$, $S$ | `S` | set and number of observations |
# | $\Delta x$ | `dx` | bin size of the empirical pdf |
# | $\hat{f}(x)$, $\hat{F}(x)$ | `f_hat`, `F_hat` | empirical pdf and cdf |
# | $u$, $u'$ | `u_low`, `u_high` | operating temperature (decision) [K] |

# %% [part] Realizations, frequency, and cumulative frequency

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# %% [markdown]
# Load the realizations at high temperature $u'$ (column 0: pressure $x_\omega$,
# column 1: conversion $y_\omega$) and at low temperature $u$.

# %%
data_high = pd.read_csv(path+'Data/gibbs_hightemp.dat', sep="\t", header=None)  # u' = 613 K
data_low = pd.read_csv(path+'Data/gibbs_lowtemp.dat', sep="\t", header=None)    # u  = 583 K
data_high.head(5)

# %%
x_omega = data_high.iloc[:, 0].to_numpy()   # realizations x_omega of pressure X
S = len(x_omega)                            # number of observations
omega = np.arange(S)                  # observation index

plt.scatter(omega, x_omega, marker='o', s=20, color='white', edgecolor='black')
plt.ylim([min(x_omega)-10, max(x_omega)+10])
plt.xlim([min(omega)-10, max(omega)+10])
plt.xlabel(r'Outcome $\omega$')
plt.ylabel(r'$x_\omega$ [bar]')
plt.grid(True)
plt.show()

# %% [markdown]
# The *frequency* counts the observations in each bin $[x, x+\Delta x]$:
#
# $$\sum_{\omega \in \mathcal{S}}\mathbf{1}[x\leq x_\omega\leq x+\Delta x]$$

# %%
dx = 10
x_grid = np.arange(60, 240+dx, dx)   # discretized empirical domain

freq = np.zeros(len(x_grid))
for i in range(len(x_grid)):
    for w in range(S):
        if x_grid[i] <= x_omega[w] < x_grid[i] + dx:   # indicator 1[x <= x_omega <= x + dx]
            freq[i] += 1

plt.step(x_grid, freq, color='black', where='mid')
plt.xlabel('$x$ [bar]')
plt.ylabel('Frequency')
plt.axis([60, 240, 0, 180])
plt.grid(True)
plt.show()

# %% [markdown]
# The same frequency plot can be produced directly with a histogram:

# %%
plt.hist(x_omega, bins=np.arange(60, 250, dx), edgecolor='black', facecolor='none', linewidth=1)
plt.xlabel('$x$ [bar]', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axis([60, 240, 0, 180])
plt.show()

# %% [markdown]
# The *cumulative frequency* counts the observations below $x$:
#
# $$\sum_{\omega \in \mathcal{S}}\mathbf{1}[x_\omega\leq x]$$

# %%
cum_freq = np.zeros(len(x_grid))
for i in range(len(x_grid)):
    for w in range(S):
        if x_omega[w] <= x_grid[i]:                    # indicator 1[x_omega <= x]
            cum_freq[i] += 1

plt.step(x_grid, cum_freq, color='black', where='mid')
plt.xlabel('$x$ [bar]')
plt.ylabel('Cumulative frequency')
plt.axis([60, 240, 0, 1000])
plt.grid(True)
plt.show()

# %% [part] Empirical pdf/cdf versus a Gaussian model

# %% [markdown]
# Normalizing by $S$ (and by $\Delta x$ for the pdf) gives the empirical cdf and pdf
#
# $$\hat{F}(x)=\frac{1}{S}\sum_{\omega \in \mathcal{S}}\mathbf{1}[x_\omega\leq x],\qquad
# \hat{f}(x)=\frac{1}{S\,\Delta x}\sum_{\omega \in \mathcal{S}}\mathbf{1}[x\leq x_\omega\leq x+\Delta x].$$
#
# We compare them with the pdf and cdf of a Gaussian model with parameters $\mu,\sigma$
# fitted to the data.

# %%
F_hat = cum_freq / S
f_hat = freq / (S * dx)

mu, sigma = norm.fit(x_omega)        # Gaussian parameters (mu, sigma)
x_plot = np.linspace(50, 250, 1000)

plt.step(x_grid, f_hat, color='black', where='mid')
plt.plot(x_plot, norm.pdf(x_plot, mu, sigma), 'black', linewidth=1.5)
plt.xlabel('$x$ [bar]', fontsize=14)
plt.ylabel(r'$\hat{f}(x)$, $f(x)$', fontsize=14)
plt.axis([50, 250, 0, 0.02])
plt.show()

# %%
plt.step(x_grid, F_hat, color='black', where='mid')
plt.plot(x_plot, norm.cdf(x_plot, mu, sigma), 'black', linewidth=1.5)
plt.xlabel('$x$ [bar]', fontsize=14)
plt.ylabel(r'$\hat{F}(x)$, $F(x)$', fontsize=14)
plt.grid(True)
plt.axis([50, 250, 0, 1.01])
plt.show()

# %% [part] Convergence of the empirical mean and SD

# %% [markdown]
# Empirical mean and standard deviation from $S$ observations:
#
# $$\hat{\mathbb{E}}_X=\frac{1}{S}\sum_{\omega \in \mathcal{S}}x_\omega,\qquad
# \hat{\mathbb{SD}}_X=\sqrt{\frac{1}{S}\sum_{\omega \in \mathcal{S}}(x_\omega-\hat{\mathbb{E}}_X)^2}$$
#
# We draw samples of increasing size $S$ and compare against the values of the
# fitted model, $\mathbb{E}_X=\mu$ and $\mathbb{SD}_X=\sigma$.

# %%
np.random.seed(1)  # reproducibility
S_values = np.arange(5, 1001, 5)
E_hat = np.zeros(len(S_values))
SD_hat = np.zeros(len(S_values))

for j, S_j in enumerate(S_values):
    x_sample = np.random.choice(x_omega, size=S_j, replace=True)
    E_hat[j] = np.mean(x_sample)
    SD_hat[j] = np.sqrt(np.var(x_sample))

fig, axs = plt.subplots(2, 1, figsize=(8, 10))
axs[0].plot(S_values, mu * np.ones(len(S_values)), 'k-', linewidth=1.5)
axs[0].plot(S_values, E_hat, 'ko', markerfacecolor='w', markersize=4)
axs[0].set_xlabel('Sample size $S$', fontsize=14)
axs[0].set_ylabel(r'$\hat{\mathbb{E}}_X$', fontsize=14)
axs[0].grid(True)
axs[1].plot(S_values, sigma * np.ones(len(S_values)), 'k-', linewidth=1.5)
axs[1].plot(S_values, SD_hat, 'ko', markerfacecolor='w', markersize=4)
axs[1].set_xlabel('Sample size $S$', fontsize=14)
axs[1].set_ylabel(r'$\hat{\mathbb{SD}}_X$', fontsize=14)
axs[1].grid(True)
plt.tight_layout()
plt.show()
print(f"E_X = mu = {mu:.1f} bar,  SD_X = sigma = {sigma:.1f} bar")

# %% [part] Quantiles and moments @ Summarizing Statistics (Mode and Moments)

# %% [markdown]
# The $\alpha$-quantile is the inverse of the cdf, $\mathbb{Q}_X(\alpha)=F_X^{-1}(\alpha)$.
# We estimate $\hat{\mathbb{Q}}_X(0.5)$ (the median $\mathbb{M}_X$) and
# $\hat{\mathbb{Q}}_X(0.9)$ from the data and show them on the empirical cdf $\hat{F}_X$.

# %%
x_grid = np.arange(60, 245, 5)
F_hat = np.array([np.sum(x_omega <= x_k) / S for x_k in x_grid])   # empirical cdf

Q_hat_50 = np.quantile(x_omega, 0.5)   # median M_X
Q_hat_90 = np.quantile(x_omega, 0.9)

plt.figure(figsize=(8, 5))
plt.step(x_grid, F_hat, where='post', linewidth=1, color='black')
plt.plot(x_plot, norm.cdf(x_plot, mu, sigma), 'black', linewidth=1.5)
for alpha, Q in ((0.5, Q_hat_50), (0.9, Q_hat_90)):
    plt.plot([60, 240], [alpha, alpha], 'k--')
    plt.plot([Q, Q], [0, 1], 'k--')
plt.axis([60, 240, 0, 1])
plt.grid(True)
plt.xlabel(r'$\mathbb{Q}_X(\alpha)$ [bar]', fontsize=14)
plt.ylabel(r'$\alpha$', fontsize=14)
plt.show()
print(f"Q_hat(0.5) = {Q_hat_50:.1f} bar,  Q_hat(0.9) = {Q_hat_90:.1f} bar")

# %% [markdown]
# Central moments $\mathbb{CMO}_k[X]=\mathbb{E}[(X-\mathbb{E}[X])^k]$ and standardized
# moments $\mathbb{SMO}_k[X]=\mathbb{CMO}_k[X]/\mathbb{SD}[X]^k$; $\mathbb{SMO}_3$ is the
# skewness and $\mathbb{SMO}_4$ the kurtosis.

# %%
from scipy.stats import moment

CMO = {k: moment(x_omega, moment=k) for k in (1, 2, 3, 4)}   # central moments CMO_k
SD_X = np.sqrt(CMO[2])
SMO_3 = CMO[3] / SD_X**3      # skewness
SMO_4 = CMO[4] / SD_X**4      # kurtosis (scipy.stats.kurtosis returns SMO_4 - 3)

for k in CMO:
    print(f"CMO_{k}[X] = {CMO[k]:.4g}")
print(f"Skewness SMO_3[X] = {SMO_3:.4f}")
print(f"Kurtosis SMO_4[X] = {SMO_4:.4f}")

# %% [part] Propagation of pressure to conversion

# %% [markdown]
# The conversion is the output $Y=\varphi(X,u)$ of the reactor; each realization
# $y_\omega=\varphi(x_\omega,u)$ comes with a pressure realization $x_\omega$.
# We compare the empirical pdfs of input $X$ and output $Y$.

# %%
y_high = 100 * data_high.iloc[:, 1].to_numpy()   # y_omega at u' (conversion, %)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].hist(x_omega, bins=np.arange(60, 250, 10), edgecolor='black', facecolor='none', linewidth=1)
axs[0].set_xlabel('$x$ [bar]', fontsize=14)
axs[0].set_ylabel('Frequency', fontsize=14)
axs[0].axis([60, 240, 0, 170])
axs[1].hist(y_high, bins=np.arange(10, 94, 4), edgecolor='black', facecolor='none', linewidth=1)
axs[1].set_xlabel('$y$ [%]', fontsize=14)
axs[1].set_ylabel('Frequency', fontsize=14)
axs[1].axis([10, 90, 0, 170])
plt.tight_layout()
plt.show()

# %% [part] Comparing decisions u and u'

# %% [markdown]
# Outputs $Y(u)$ at low temperature $u=583$ K and $Y(u')$ at high temperature
# $u'=613$ K. We compare their empirical pdfs and cdfs
# $\hat{F}_Y(y)=\frac{1}{S}\sum_{\omega}\mathbf{1}[y_\omega\leq y]$.

# %%
u_low, u_high = 583, 613
y_low = 100 * data_low.iloc[:, 1].to_numpy()     # y_omega at u

def F_hat_of(y_data, y_grid):
    """Empirical cdf  F_hat(y) = (1/S) sum_omega 1[y_omega <= y]."""
    return np.array([np.mean(y_data <= y_k) for y_k in y_grid])

y_grid = np.arange(10, 105, 5)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].hist(y_high, bins=np.arange(10, 94, 4), edgecolor='black', facecolor='white', label=f"u' = {u_high} K")
ax[0].hist(y_low, bins=np.arange(10, 94, 4), edgecolor='black', facecolor='black', alpha=0.7, label=f"u = {u_low} K")
ax[0].set_xlabel('$y$ [%]', fontsize=14)
ax[0].set_ylabel('Frequency', fontsize=14)
ax[0].legend(loc='upper left')
ax[1].step(y_grid, F_hat_of(y_high, y_grid), where='post', color='black', linestyle='--', label="$Y(u')$")
ax[1].step(y_grid, F_hat_of(y_low, y_grid), where='post', color='black', linestyle='-', label='$Y(u)$')
ax[1].set_xlabel('$y$ [%]', fontsize=14)
ax[1].set_ylabel(r'$\hat{F}_Y(y)=\mathbb{P}(Y\leq y)$', fontsize=14)
ax[1].grid(True)
ax[1].axis([10, 100, 0, 1.01])
ax[1].legend(loc='upper left')
plt.tight_layout()
plt.show()

# %% [part] Best decision by coefficient of variation @ Decision-Making under Uncertainty

# %% [markdown]
# Use the risk measure $\rho(Y(u))=\mathbb{SD}[Y(u)]/\mathbb{E}[Y(u)]$ (coefficient of
# variation) and pick $u^*\in\mathcal{U}=\{583, 613\}$ that minimizes it.

# %%
def rho(y_data):
    """Coefficient of variation CV = SD[Y] / E[Y]."""
    return np.std(y_data) / np.mean(y_data)

U = {u_low: y_low, u_high: y_high}
for u, y_u in U.items():
    print(f"u = {u} K:  E_hat[Y] = {np.mean(y_u):.1f} %,  SD_hat[Y] = {np.std(y_u):.1f} %,  CV = {rho(y_u):.3f}")
u_star = min(U, key=lambda u: rho(U[u]))
print(f"Optimal decision u* = {u_star} K")
