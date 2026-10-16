# %% [markdown]
# 10-min wind speed as a Weibull RV (Victor Z and David Cole, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\beta$, $\xi$ | `beta`, `xi` | scale [m/s] and shape [-] |
# | $\mathbb{M}_X$ | `M_X` | median wind speed |
# | $\mathbb{E}_X$, $\mathbb{MO}_X$ | `E_X`, `MO_X` | mean and mode |
# | $x_\omega$ | `x_omega` | simulated 10-min wind speeds |

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.special import gamma as Gamma

# %% [markdown]
# From the empirical cdf: $F(x)=0.63$ at $x=6$ m/s gives $\beta=6$; the median $\mathbb{M}_X=5$
# and $\log 5=\log\beta+(1/\xi)\log(\log 2)$ give $\xi$.

# %%
beta = 6.0
M_X = 5.0
xi = np.log(np.log(2)) / (np.log(M_X) - np.log(beta))
print(f"xi from the median: {xi:.2f}  ->  use xi = 2")
xi = 2.0
X = weibull_min(xi, scale=beta)    # scipy: shape xi first, scale beta

# %%
x_bins = np.arange(0.5, 20, 1)     # measured frequencies (% of time) per 1 m/s bin
f_data = np.array([2.75, 7.8, 11.6, 13.79, 14.20, 13.15, 11.14, 8.72, 6.34, 4.30, 2.73, 1.62,
                   0.91, 0.48, 0.24, 0.11, 0.05, 0.02, 0.01, 0.001]) / 100
x = np.linspace(0, 19, 1000)

np.random.seed(0)
S = 24 * 7 * 6
x_omega = X.rvs(size=S)

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(np.arange(S), x_omega, s=8, edgecolor='black', facecolor='white')
ax1.set_xlim([0, S])
ax1.set_ylim([0, 20])
ax1.set_xlabel(r'Observation $\omega$', fontsize=14)
ax1.set_ylabel(r'$x_\omega$ [m/s]', fontsize=14)
ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(x_bins, f_data, edgecolor='black', facecolor='white', linewidth=0.5, label='Empirical')
ax2.plot(x, X.pdf(x), 'black', linewidth=1.5, label='Weibull model')
ax2.set_xlabel('$x$ [m/s]', fontsize=14)
ax2.set_ylabel('$f(x)$', fontsize=14)
ax2.legend(fontsize=11)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(x, X.cdf(x), 'black', linewidth=1.5)
ax3.set_xlabel('$x$ [m/s]', fontsize=14)
ax3.set_ylabel('$F(x)$', fontsize=14)
for ax in (ax1, ax2, ax3):
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# $\mathbb{E}_X=\beta\,\Gamma(1+1/\xi)$, $\mathbb{MO}_X=\beta\left(\frac{\xi-1}{\xi}\right)^{1/\xi}$, and
# $\mathbb{P}(10\leq X\leq 20)=F(20)-F(10)$.

# %%
E_X = beta * Gamma(1 + 1/xi)
MO_X = beta * ((xi - 1) / xi) ** (1/xi)
P_high = X.cdf(20) - X.cdf(10)
print(f"E_X  = {E_X:.1f} m/s  (scipy mean: {X.mean():.1f})")
print(f"MO_X = {MO_X:.1f} m/s")
print(f"P(10 <= X <= 20) = {P_high:.2f}")
