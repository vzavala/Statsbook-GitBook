# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Empirical event probabilities for the Gibbs reactor (Victor Z, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $x_{\omega,1}$, $x_{\omega,2}$ | `x1`, `x2` | pressure [bar], conversion [-] |
# | $\mathcal{A}_1$, $\mathcal{A}_2$ | `in_A1`, `in_A2` | indicators $\mathbf{1}[x_\omega\in\mathcal{A}_i]$ |
# | $\hat{\mathbb{P}}(X\in\mathcal{A}_1\cap\mathcal{A}_2)$ | `P_A1_and_A2` | empirical joint probability |
# | $\hat{\mathbb{P}}(X\in\mathcal{A}_1\mid X\in\mathcal{A}_2)$ | `P_A1_given_A2` | empirical conditional probability |
#
# Events: $\mathcal{A}_1=\{x_1\leq 170\ \&\ x_2\geq 0.5\}$ and $\mathcal{A}_2=\{x_1\geq 120\ \&\ x_2\geq 0.5\}$.

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(path+'Data/gibbs_hightemp_class.dat')
S = len(data)
np.random.seed(1)
x1 = data[:, 1] + np.random.randn(S)          # pressure
x2 = data[:, 2] + np.random.randn(S) * 0.05   # conversion

in_A1 = (x1 <= 170) & (x2 >= 0.5)
in_A2 = (x1 >= 120) & (x2 >= 0.5)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, mask, title in ((axs[0], in_A1, r'$\mathcal{A}_1$'), (axs[1], in_A2, r'$\mathcal{A}_2$'),
                        (axs[2], in_A1 & in_A2, r'$\mathcal{A}_1\cap\mathcal{A}_2$')):
    ax.scatter(x1[~mask], x2[~mask], s=8, c='white', edgecolors='black')
    ax.scatter(x1[mask], x2[mask], s=8, c='black')
    ax.set_title(title)
    ax.set_xlabel('$x_1$ [bar]')
    ax.set_ylabel('$x_2$ [-]')
    ax.axis([50, 250, 0, 1])
    ax.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# $\hat{\mathbb{P}}(X\in\mathcal{A})=\frac{1}{S}\sum_\omega\mathbf{1}[x_\omega\in\mathcal{A}]$ and
# $\hat{\mathbb{P}}(X\in\mathcal{A}_1|X\in\mathcal{A}_2)=\hat{\mathbb{P}}(X\in\mathcal{A}_1\cap\mathcal{A}_2)/\hat{\mathbb{P}}(X\in\mathcal{A}_2)$.

# %%
P_A1 = np.mean(in_A1)
P_A2 = np.mean(in_A2)
P_A1_and_A2 = np.mean(in_A1 & in_A2)
P_A1_or_A2 = np.mean(in_A1 | in_A2)
P_A1_given_A2 = P_A1_and_A2 / P_A2
P_A2_given_A1 = P_A1_and_A2 / P_A1
print(f"P_hat(A1) = {P_A1:.3f},  P_hat(A2) = {P_A2:.3f}")
print(f"P_hat(A1 & A2) = {P_A1_and_A2:.3f},  P_hat(A1 or A2) = {P_A1_or_A2:.3f}")
print(f"P_hat(A1 | A2) = {P_A1_given_A2:.3f},  P_hat(A2 | A1) = {P_A2_given_A1:.3f}")
print(f"Bayes' rule check: P(A2|A1) P(A1) / P(A2) = {P_A2_given_A1 * P_A1 / P_A2:.3f}")
