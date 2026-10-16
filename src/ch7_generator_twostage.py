# %% [markdown]
# Two-stage formulation for generator sizing (Victor Z and David Cole, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $X=(X_1,X_2)$ | `X1`, `X2` | two random loads ($S=200$ samples) |
# | $u\in\mathcal{U}=\{1,10\}$ | `u`, `U` | capacity (first stage, preemptive) |
# | $W(X)\in\mathcal{W}(u)$ | `W` | generator output (second stage, recourse) |
# | $R=X_1+X_2-W(X)$ | `R` | residual load |
# | $Y(u,W(X))=\alpha_R R^2+\alpha_u u^2$ | `cost` | cost |
#
# For a given $u$ the best recourse in every scenario is $W(X)=\min(u,\,X_1+X_2)$: it satisfies
# $0\leq W\leq X_1+X_2$ and $W\leq u$, and it makes $R$ (hence each scenario's cost) as small as possible,
# so it also minimizes $\mathrm{CVaR}_\alpha$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

alpha_R, alpha_u, alpha = 5, 15, 0.9
np.random.seed(0)
S = 200
X1 = weibull_min.rvs(2, scale=10, size=S)
X2 = weibull_min.rvs(1, scale=2, size=S)
U = [1, 10]

def recourse(u, X1, X2):
    return np.minimum(u, X1 + X2)

def cost(u, X1, X2):
    R = X1 + X2 - recourse(u, X1, X2)
    return alpha_R * R**2 + alpha_u * u**2

def CVaR(y):
    return y[y >= np.quantile(y, alpha)].mean()

rho_stoch = {u: CVaR(cost(u, X1, X2)) for u in U}
u_star = min(rho_stoch, key=rho_stoch.get)
rho_det = {u: cost(u, X1.max(), X2.max()) for u in U}      # worst-case loads
u_D = min(rho_det, key=rho_det.get)
print("CVaR of cost:", {u: round(v, 1) for u, v in rho_stoch.items()}, "-> u* =", u_star)
print("deterministic (worst-case) cost:", {u: round(float(v), 1) for u, v in rho_det.items()}, "-> u_D* =", u_D)

# %%
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
axs[0].hist(X1, bins=np.arange(0, 31, 1), edgecolor='black', facecolor='white', label='$x_1$')
axs[0].hist(X2, bins=np.arange(0, 31, 1), edgecolor='black', facecolor='black', alpha=0.5, label='$x_2$')
axs[0].set_xlabel('load'); axs[0].set_ylabel('Frequency'); axs[0].legend()
for u, style in ((u_star, 'k-'), (u_D, 'k--')):
    y_u = cost(u, X1, X2)
    axs[1].step(np.sort(y_u), np.arange(1, S + 1) / S, style, where='post', label=f'$u={u}$')
axs[1].set_xlabel('$y$'); axs[1].set_ylabel(r'$\hat{F}(y)$'); axs[1].legend(loc='lower right')
axs[2].scatter(X1 + X2, recourse(u_star, X1, X2), s=10, c='k')
axs[2].set_xlabel('$x_1+x_2$'); axs[2].set_ylabel(f'$W(X)$ for $u={u_star}$')
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()
