# %% [markdown]
# Bayesian optimization of pectin extraction (Victor Z and Leo Gonzalez, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $u=(u_1,u_2,u_3)\in[-1,1]^3$ | `u` | scaled temperature, time, acid concentration |
# | $\varphi(u)$ | `phi(u)` | yield (a quadratic model plays the "black box") |
# | $\mathcal{D}_\ell$ | `data_u`, `data_y` | experiments so far |
# | $m_\ell(u)$, $\sigma_\ell(u)$ | `gp.predict(u, return_std=True)` | kriging mean and SD |
# | $AF_\ell(u)=m_\ell(u)+\kappa\sigma_\ell(u)$ | `AF(u)`, `kappa` | acquisition function ($\kappa=0.2$) |

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize, Bounds

b = np.array([16.5780, 3.8563, 0.9250, 0.4212, -1.2840, -0.8515, -1.0940, -0.1425, 0.1900, -0.1675])

def phi(u):
    u = np.atleast_2d(u)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    return (b[0] + b[1]*u1 + b[2]*u2 + b[3]*u3 + b[4]*u1**2 + b[5]*u2**2 + b[6]*u3**2
            + b[7]*u1*u2 + b[8]*u1*u3 + b[9]*u2*u3)

# %% [markdown]
# Start from 3 experiments and iterate: fit the kriging model, maximize $AF_\ell(u)$, run the new
# experiment, and add it to the data set.

# %%
kappa, n_iter = 0.2, 20
bounds = Bounds(-np.ones(3), np.ones(3))
data_u = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]], dtype=float)
data_y = phi(data_u)
gp = GaussianProcessRegressor(kernel=RBF(length_scale=np.ones(3)), normalize_y=True)

for ell in range(n_iter):
    gp.fit(data_u, data_y)
    def neg_AF(u):
        m, s = gp.predict(np.atleast_2d(u), return_std=True)
        return -(m[0] + kappa * s[0])
    u_new = minimize(neg_AF, data_u[np.argmax(data_y)], method='SLSQP', bounds=bounds).x
    data_u = np.vstack([data_u, u_new])
    data_y = np.append(data_y, phi(u_new))

def descale(u, lb=np.array([70, 1, 0.1]), ub=np.array([100, 2, 0.3])):
    """Map [-1, 1] to temperature [C], time [h], concentration [M]."""
    return lb + (u + 1) / 2 * (ub - lb)

best = np.argmax(data_y)
print("best scaled conditions u =", np.round(data_u[best], 3), "  yield =", round(float(data_y[best]), 3))
print("in physical units:", np.round(descale(data_u[best]), 3))

ell = np.arange(len(data_y))
plt.step(ell, np.maximum.accumulate(data_y), 'k-', where='post', label='best so far')
plt.scatter(ell, data_y, color='w', edgecolor='k', label=r'$y_\ell$')
plt.xlabel(r'Experiment $\ell$'); plt.ylabel(r'Observed yield $y_\ell$')
plt.legend(); plt.grid(True)
plt.show()
