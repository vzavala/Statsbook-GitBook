# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Pump failures as a Poisson RV (Victor Z and David Cole, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $z$ | `z` | interval length (days in a month) |
# | $x_{Jan},x_{Feb},x_{Mar}$ | `x_omega` | failures recorded per month |
# | $\hat{\mathbb{E}}_X$, $\hat{\lambda}$ | `E_hat`, `lam_hat` | empirical mean and estimate of $\lambda$ |
# | $f(x)=\lambda^x e^{-\lambda}/x!$ | `f(x)` | Poisson pdf |

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# %% [markdown]
# Daily records for three months with $x_{Jan}=8$, $x_{Feb}=12$, $x_{Mar}=14$ failures.

# %%
np.random.seed(0)  # reproducibility
z = 30
days = np.arange(1, z + 1)
x_omega = np.array([8, 12, 14])            # failures in Jan, Feb, Mar

plt.figure(figsize=(10, 3))
for k, (month, x_k) in enumerate(zip(['Jan', 'Feb', 'Mar'], x_omega)):
    failed = np.zeros(z, dtype=bool)
    failed[np.random.choice(z, x_k, replace=False)] = True
    plt.scatter(days, np.full(z, 3 - k), c=np.where(failed, 'black', 'white'), edgecolors='black')
plt.yticks([1, 2, 3], ['Mar', 'Feb', 'Jan'])
plt.xlabel('Time [days]', fontsize=14)
plt.grid(True)
plt.axis([0, z + 1, 0, 4])
plt.show()

# %% [markdown]
# Since $\mathbb{E}_X=\lambda$, estimate $\hat{\lambda}=\hat{\mathbb{E}}_X=\frac{1}{3}(8+12+14)$ and
# evaluate $\mathbb{P}(X=0)$, $\mathbb{P}(X=30)$, and $\mathbb{P}(X=11)$.

# %%
E_hat = np.mean(x_omega)
lam_hat = E_hat

def f(x):
    """Poisson pdf f(x) = lam^x e^(-lam) / x!  with lam = lam_hat."""
    return poisson.pmf(x, lam_hat)

print(f"lam_hat = E_hat = {lam_hat:.1f}")
print(f"P(X=0)  = {f(0):.3g}")
print(f"P(X=30) = {f(30):.3g}")
print(f"P(X=11) = {f(11):.3f}")
