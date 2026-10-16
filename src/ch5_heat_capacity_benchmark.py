# %% [markdown]
# **Additional example: benchmarking models and cross-validation** (Victor Z and Leo Gonzalez,
# UW-Madison, 2020). Model I is $Y=\theta_0+\theta_1X+\theta_2X^2$, model II is $Y=\theta_0+\theta_1X$.
# We compare their squared errors $\hat{\epsilon}_\omega^2$ (mean and empirical cdf), and then their
# 3-fold cross-validation errors on data not used for estimation.
# Python names: `X_I`, `X_II` (data matrices), `eps2_I`, `eps2_II`, `MSE_cv`.

# %%
import numpy as np
import matplotlib.pyplot as plt

theta = np.array([6.190, 2.923e-3, -7.052e-7])
x = np.repeat([300.0, 600.0, 900.0], 10)
S = len(x)
np.random.seed(0)
y = theta[0] + theta[1]*x + theta[2]*x**2 + np.random.normal(0, 0.1, S)

X_I = np.column_stack([np.ones(S), x, x**2])
X_II = np.column_stack([np.ones(S), x])

def fit(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

eps2_I = (y - X_I @ fit(X_I, y))**2
eps2_II = (y - X_II @ fit(X_II, y))**2
print(f"MSE I = {eps2_I.mean():.4f},  MSE II = {eps2_II.mean():.4f}")

for e2, lw, label in ((eps2_I, 1.5, 'Model I'), (eps2_II, 0.5, 'Model II')):
    plt.step(np.sort(e2), np.arange(1, S + 1) / S, 'k-', where='post', linewidth=lw, label=label)
plt.xlim(0, 0.1)
plt.xlabel(r'$\epsilon^2$', fontsize=14)
plt.ylabel(r'$\hat{F}(\epsilon^2)$', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# %% [markdown]
# 3-fold cross-validation: estimate on two folds, test on the third.

# %%
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(S), 3)
for name, X in (("I", X_I), ("II", X_II)):
    MSE_cv = []
    for k in range(3):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(3) if j != k])
        theta_k = fit(X[train], y[train])
        MSE_cv.append(np.mean((y[test] - X[test] @ theta_k)**2))
    print(f"Model {name}: sum of test MSE over folds = {np.sum(MSE_cv):.4f}")
