# %% [markdown]
# Joint, marginal, and conditional pdfs of $X\sim\mathcal{N}(\mu,\Sigma)$ with $\mu=(0,1)$ and
# $\Sigma=\begin{bmatrix}1&-1/2\\-1/2&1\end{bmatrix}$ (Victor Z, UW-Madison, 2024).
# Python names: `mu`, `Sigma`, joint pdf/cdf `f`, `F`, marginals `f_1`, `f_2`,
# conditional moments `mu_1g2`, `Sigma_1g2` ($\mu_{1|2}$, $\Sigma_{1|2}$) and `mu_2g1`, `Sigma_2g1`.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

mu = np.array([0, 1])
Sigma = np.array([[1, -0.5], [-0.5, 1]])
X = multivariate_normal(mean=mu, cov=Sigma)

x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-2, 4, 100)
X1, X2 = np.meshgrid(x1, x2)
grid = np.dstack([X1, X2])
f = X.pdf(grid)
F = X.cdf(grid)

fig = plt.figure(figsize=(10, 8))
for k, (Z, label) in enumerate(((f, '$f(x_1,x_2)$'), (F, '$F(x_1,x_2)$'))):
    ax = fig.add_subplot(2, 2, k + 1, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='gray')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(label)
    ax2 = fig.add_subplot(2, 2, k + 3)
    ax2.imshow(Z, extent=[-3, 3, -2, 4], cmap='gray', origin='lower')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
plt.tight_layout()
plt.show()

# %% [markdown]
# Marginals $X_1\sim\mathcal{N}(\mu_1,\Sigma_{11})$, $X_2\sim\mathcal{N}(\mu_2,\Sigma_{22})$ and conditionals with
# $\mu_{1|2}=\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)$, $\Sigma_{1|2}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$
# (and symmetrically for $2|1$), evaluated at $x_2=1$ and $x_1=1$.

# %%
f_1 = norm.pdf(x1, mu[0], np.sqrt(Sigma[0, 0]))
f_2 = norm.pdf(x2, mu[1], np.sqrt(Sigma[1, 1]))

x2_given, x1_given = 1.0, 1.0
mu_1g2 = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (x2_given - mu[1])
Sigma_1g2 = Sigma[0, 0] - Sigma[0, 1] / Sigma[1, 1] * Sigma[1, 0]
mu_2g1 = mu[1] + Sigma[1, 0] / Sigma[0, 0] * (x1_given - mu[0])
Sigma_2g1 = Sigma[1, 1] - Sigma[1, 0] / Sigma[0, 0] * Sigma[0, 1]

fig, axs = plt.subplots(2, 2, figsize=(10, 7))
axs[0, 0].plot(x1, f_1, 'k'); axs[0, 0].set_ylabel('$f_1(x_1)$'); axs[0, 0].set_xlabel('$x_1$')
axs[0, 1].plot(x2, f_2, 'k'); axs[0, 1].set_ylabel('$f_2(x_2)$'); axs[0, 1].set_xlabel('$x_2$')
axs[1, 0].plot(x1, norm.pdf(x1, mu_1g2, np.sqrt(Sigma_1g2)), 'k')
axs[1, 0].set_ylabel('$f(x_1|x_2=1)$'); axs[1, 0].set_xlabel('$x_1$')
axs[1, 1].plot(x2, norm.pdf(x2, mu_2g1, np.sqrt(Sigma_2g1)), 'k')
axs[1, 1].set_ylabel('$f(x_2|x_1=1)$'); axs[1, 1].set_xlabel('$x_2$')
for ax in axs.ravel():
    ax.grid(True)
    ax.set_ylim(0, 0.5)
plt.tight_layout()
plt.show()
print(f"X1 | x2=1 ~ N({mu_1g2:.2f}, {Sigma_1g2:.2f}),   X2 | x1=1 ~ N({mu_2g1:.2f}, {Sigma_2g1:.2f})")
