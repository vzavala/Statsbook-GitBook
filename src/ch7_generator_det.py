# %% [markdown]
# Generator sizing: the deterministic decision (Victor Z and David Cole, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $L$, realizations | `L_omega` | random power load (Weibull, scale 10, shape 2) |
# | $u$, $\mathcal{U}=\{0,1,\dots,10\}$ | `u`, `U` | generator capacity and options |
# | $R=\max(L-u,0)$ | `R` | residual demand bought from the grid |
# | $Y(u)=\alpha_R R^2+\alpha_u u^2$ | `cost(u, L)` | total cost ($\alpha_R=5$, $\alpha_u=15$) |
# | $r=\max_\omega r_\omega$ | `L_peak` | representative (peak) load |
# | $u_D^*$ | `u_D` | deterministic decision |

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

alpha_R, alpha_u = 5, 15

def cost(u, L):
    R = np.maximum(L - u, 0)
    return alpha_R * R**2 + alpha_u * u**2

np.random.seed(0)
S = 1000
L_omega = weibull_min.rvs(2, scale=10, size=S)
U = np.arange(0, 11)

# %% [markdown]
# Deterministic problem with the peak load: $\min_{u\in\mathcal{U}}\ \alpha_R r^2+\alpha_u u^2$ with $r=\max(L_{peak}-u,0)$.

# %%
L_peak = np.max(L_omega)
y_det = np.array([cost(u, L_peak) for u in U])
u_D = U[np.argmin(y_det)]
print(f"peak load = {L_peak:.1f};  deterministic decision u_D* = {u_D}")

# %% [markdown]
# Monte Carlo evaluation of $u_D^*$ and of the do-nothing option $u=0$ under all load outcomes
# $y_\omega(u)=\alpha_R R_\omega^2+\alpha_u u^2$.

# %%
y_D, y_0 = cost(u_D, L_omega), cost(0, L_omega)
print(f"average cost: u_D* -> {y_D.mean():.1f},  do nothing -> {y_0.mean():.1f}")
print(f"maximum cost: u_D* -> {y_D.max():.1f},  do nothing -> {y_0.max():.1f}")

fig, axs = plt.subplots(2, 2, figsize=(11, 9))
axs[0, 0].hist(L_omega, bins=np.arange(0, 31, 1), edgecolor='black', facecolor='white')
axs[0, 0].set_xlabel(r'load $\ell$'); axs[0, 0].set_ylabel('Frequency')
axs[0, 1].plot(U, y_det, 'k--o', markerfacecolor='w')
axs[0, 1].set_xlabel('$u$'); axs[0, 1].set_ylabel('$y(u)$ at peak load')
for y_u, style, label in ((y_D, '--', f'deterministic ($u={u_D}$)'), (y_0, '-', 'do nothing ($u=0$)')):
    axs[1, 0].step(np.sort(y_u), np.arange(1, S + 1) / S, 'k' + style, where='post', label=label)
axs[1, 0].set_xlabel('$y$'); axs[1, 0].set_ylabel(r'$\hat{F}(y)$'); axs[1, 0].legend(loc='lower right')
axs[1, 1].hist(y_D, bins=np.arange(0, 3100, 100), edgecolor='black', facecolor='white', label='deterministic')
axs[1, 1].hist(y_0, bins=np.arange(0, 3100, 100), edgecolor='black', facecolor='black', alpha=0.5, label='do nothing')
axs[1, 1].set_xlabel('$y$'); axs[1, 1].set_ylabel('Frequency'); axs[1, 1].legend()
for ax in axs.ravel():
    ax.grid(True)
plt.tight_layout()
plt.show()
