# %% [markdown]
# Comparing investment options through their net present value (Victor Z and Leo Gonzalez, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $c_I$, $c_{II}$ | `c_I`, `c_II` | capital [thousand USD] (25 and 50) |
# | $R_I\sim\mathcal{N}(0.5,0.1)$, $R_{II}\sim\mathcal{N}(0.4,0.2)$ | `R_I`, `R_II` | annual rate of return (mean, SD) |
# | $i$ | `i` | interest rate (5%, 10 years) |
# | $NPV$, i.e. $Y(u)$ and $Y(u')$ | `NPV_I`, `NPV_II` | net present value (negative = revenue) |

# %% [part] Distributions of the NPV

# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
S = 1000
i, years = 0.05, 10
annuity = (1 - (1 + i)**(-years)) / i
c_I, c_II = 25, 50
R_I = np.random.normal(0.5, 0.1, S)
R_II = np.random.normal(0.4, 0.2, S)
NPV_I = c_I - R_I * c_I * annuity
NPV_II = c_II - R_II * c_II * annuity

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].hist(NPV_I, 20, density=True, color='w', edgecolor='k', label='Option I')
axs[0].hist(NPV_II, 20, density=True, color='k', alpha=0.5, edgecolor='k', label='Option II')
axs[0].set_xlabel('$NPV$ [kUSD]'); axs[0].set_ylabel('$f(NPV)$'); axs[0].legend()
for NPV, style, label in ((NPV_I, 'k--', 'Option I'), (NPV_II, 'k-', 'Option II')):
    axs[1].step(np.sort(NPV), np.arange(1, S + 1) / S, style, where='post', label=label)
axs[1].set_xlabel('$NPV$ [kUSD]'); axs[1].set_ylabel('$F(NPV)$'); axs[1].legend()
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [part] Risk measures for the two options @ Example: Measuring Risk for Investment Optimization

# %% [markdown]
# $\mathbb{E}$, $\mathbb{SD}$, probability of loss $\mathbb{P}(Y>0)$, probability of gain $\mathbb{P}(Y\leq\bar{y})$
# with $\bar{y}=-100$ kUSD, $\mathbb{Q}_Y(0.9)$, $\mathrm{CVaR}_{0.9}$, worst case $\max Y$ and best case $\min Y$.

# %%
def risk_measures(Y, alpha=0.9, y_bar=-100):
    Q = np.quantile(Y, alpha)
    return {'E': Y.mean(), 'SD': Y.std(ddof=1), 'P(loss)': np.mean(Y > 0), 'P(gain)': np.mean(Y <= y_bar),
            'VaR': Q, 'CVaR': Y[Y >= Q].mean(), 'max': Y.max(), 'min': Y.min()}

rm_I, rm_II = risk_measures(NPV_I), risk_measures(NPV_II)
print(f"{'measure':10s}{'Option I':>12s}{'Option II':>12s}")
for k in rm_I:
    fmt = '{:12.1%}' if k.startswith('P') else '{:12.1f}'
    print(f"{k:10s}" + fmt.format(rm_I[k]) + fmt.format(rm_II[k]))
