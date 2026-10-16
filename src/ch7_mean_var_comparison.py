# %% [markdown]
# Comparing two random costs $Y(u)\sim\mathcal{N}(3,1.5^2)$ (option I) and $Y(u')\sim\mathcal{N}(4,0.5^2)$
# (option II), in MUSD (Victor Z and Leo Gonzalez, UW-Madison, 2023).
# Python names: `Y_u`, `Y_up` (frozen distributions), `y_bar` (threshold), `alpha`, `VaR`, `CVaR`.

# %% [part] Pdfs and cdfs of the two options

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

Y_u, Y_up = norm(3, 1.5), norm(4, 0.5)
y = np.linspace(0, 10, 1000)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for Y, style, label in ((Y_u, 'k-', "Option I ($u$)"), (Y_up, 'k--', "Option II ($u'$)")):
    axs[0].plot(y, Y.pdf(y), style, label=label)
    axs[1].plot(y, Y.cdf(y), style, label=label)
axs[0].set_ylabel('$f(y)$'); axs[1].set_ylabel('$F(y)$')
for ax in axs:
    ax.set_xlabel('$y$ [MUSD]'); ax.grid(True); ax.legend()
plt.tight_layout()
plt.show()

# %% [part] Risk attitudes @ Example: Attitudes Towards Risk

# %% [markdown]
# Risk-neutral: $\mathbb{E}[Y]$; risk-conscious: $\mathbb{P}(Y\leq\bar{y})$ for $\bar{y}=5$;
# value-at-risk $\mathbb{Q}_Y(\alpha)$ and CVaR $\mathbb{E}[Y\mid Y\geq\mathbb{Q}_Y(\alpha)]$ for $\alpha=0.9$
# (estimated from $S$ samples, two ways).

# %%
y_bar, alpha = 5, 0.9
np.random.seed(0)
S = 100_000
for name, Y in (("Option I ", Y_u), ("Option II", Y_up)):
    y_omega = Y.rvs(size=S)
    VaR = Y.ppf(alpha)
    CVaR = y_omega[y_omega >= VaR].mean()
    CVaR_alt = VaR + np.mean(np.maximum(y_omega - VaR, 0)) / (1 - alpha)   # Rockafellar-Uryasev form
    print(f"{name}: E = {Y.mean():.2f},  P(Y <= {y_bar}) = {Y.cdf(y_bar):.3f},  "
          f"VaR = {VaR:.2f},  CVaR = {CVaR:.2f} ({CVaR_alt:.2f})")
