# %% [markdown] [hide]
# **Setup.** Run this cell first: it locates the data files this notebook reads. On Google
# Colab it downloads them (the `Data/` and `Support/` folders) from the book's GitHub
# repository; on your computer, run the notebook from the `notebooks/` folder.

# %% [hide]
import os
import sys
path = ''
if 'google.colab' in sys.modules and not os.path.isdir('Data'):
    if not os.path.isdir('statsbook'):
        os.system('git clone -q --depth 1 --filter=blob:none --sparse '
                  'https://github.com/__GITHUB_REPO__.git statsbook')
        os.system('git -C statsbook sparse-checkout set notebooks/Data notebooks/Support')
    path = 'statsbook/notebooks/'

# %% [markdown]
# Realizations of a 2D Gaussian and the 95% ellipsoidal region (Victor Z, UW-Madison, 2019).
# A realization $x_\omega$ (`x_omega[w]`) is inside $\mathcal{E}$ if
# $(x_\omega-\mu)^T\Sigma^{-1}(x_\omega-\mu)\leq\mathbb{Q}(1-\alpha)$ (`Q`, the $\chi^2(2)$ quantile).

# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
sys.path.append(path + 'Support')
from ellipse import ellipse

mu = np.array([0, 0])
Sigma = np.array([[1, -0.7], [-0.7, 1]])
alpha = 0.05
Q = chi2.ppf(1 - alpha, 2)
x_e, y_e, _, _, _ = ellipse(np.linalg.inv(Sigma), Q, 200, mu)

np.random.seed(0)
S = 1000
x_omega = np.random.multivariate_normal(mu, Sigma, S)
Sigma_inv = np.linalg.inv(Sigma)
inside = np.array([(x_w - mu) @ Sigma_inv @ (x_w - mu) <= Q for x_w in x_omega])
print(f"{inside.sum()} of S = {S} realizations ({inside.mean():.1%}) lie inside the ellipse")

plt.figure(figsize=(6, 6))
plt.plot(x_omega[:, 0], x_omega[:, 1], 'ko', markerfacecolor='w', markersize=3, label='realizations')
plt.plot(x_e, y_e, 'k', linewidth=1.5, label='95% ellipse')
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.legend(loc='lower left')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()
