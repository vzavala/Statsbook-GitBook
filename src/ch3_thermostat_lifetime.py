# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Lifetime of a thermostat (written for the book to accompany the slide examples).
# Sensor and processor lifetimes $(X_1,X_2)$ have joint pdf
# $f(x_1,x_2)=\frac{1}{\mu_1\mu_2}e^{-(x_1/\mu_1+x_2/\mu_2)}$ on $[0,\infty)^2$ with
# $\mu_1=1$ and $\mu_2=10$ years.
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mu_1$, $\mu_2$ | `mu_1`, `mu_2` | mean lifetimes [yr] |
# | $f(x_1,x_2)$, $F(t_1,t_2)$ | `f`, `F` | joint pdf and cdf |
# | $f_1$, $f_2$, $F_1$, $F_2$ | `f_1`, `f_2`, `F_1`, `F_2` | marginal pdfs and cdfs |
# | $f(x_1\mid x_2)$ | `f_1g2` | conditional pdf ("1 given 2") |
# | $\mathbb{P}(X\in\mathcal{A})$ | `P_A` | probability of event $\mathcal{A}$ |

# %% [part] Probabilities of box events

# %%
import numpy as np
from scipy.integrate import dblquad, quad

mu_1, mu_2 = 1.0, 10.0

def f(x1, x2):
    """Joint pdf f(x1, x2)."""
    return np.exp(-(x1/mu_1 + x2/mu_2)) / (mu_1 * mu_2)

def P_box(a1, b1, a2, b2):
    """P(a1 <= X1 <= b1 & a2 <= X2 <= b2) by numerical integration of f."""
    val, _ = dblquad(lambda x2, x1: f(x1, x2), a1, b1, a2, b2)
    return val

print(f"P(X1 <= 1 & X2 <= 1) = {P_box(0, 1, 0, 1):.3f}")
print(f"P(X1 >= 1 & X2 >= 2) = {P_box(1, np.inf, 2, np.inf):.2f}")

# %% [part] Joint cdf @ Example: Lifetime of Thermostat #2

# %% [markdown]
# $F(t_1,t_2)=\left(1-e^{-t_1/\mu_1}\right)\left(1-e^{-t_2/\mu_2}\right)$; we check it against
# the integral of the joint pdf.

# %%
def F(t1, t2):
    return (1 - np.exp(-t1/mu_1)) * (1 - np.exp(-t2/mu_2))

print(f"F(1, 1) = {F(1, 1):.3f}   (integral: {P_box(0, 1, 0, 1):.3f})")

# %% [part] Marginal pdfs and survival @ Example: Lifetime of Thermostat #3

# %% [markdown]
# Integrating out the other variable, $f_1(x_1)=\int_0^\infty f(x_1,x_2)dx_2=\frac{1}{\mu_1}e^{-x_1/\mu_1}$
# (and similarly $f_2$), so $\mathbb{P}(X_i\geq 1)=1-F_i(1)=e^{-1/\mu_i}$.

# %%
def f_1(x1):
    return quad(lambda x2: f(x1, x2), 0, np.inf)[0]

def f_2(x2):
    return quad(lambda x1: f(x1, x2), 0, np.inf)[0]

print(f"f_1(0.5) = {f_1(0.5):.4f}   vs (1/mu_1) e^(-0.5/mu_1) = {np.exp(-0.5/mu_1)/mu_1:.4f}")
F_1 = lambda x1: 1 - np.exp(-x1/mu_1)
F_2 = lambda x2: 1 - np.exp(-x2/mu_2)
print(f"P(X1 >= 1) = 1 - F_1(1) = {1 - F_1(1):.2f}")
print(f"P(X2 >= 1) = 1 - F_2(1) = {1 - F_2(1):.2f}")

# %% [part] Conditional pdfs @ Example: Lifetime of Thermostat #4

# %% [markdown]
# $f(x_1|x_2)=f(x_1,x_2)/f_2(x_2)$ does not depend on $x_2$: it equals $f_1(x_1)$.

# %%
def f_1g2(x1, x2):
    return f(x1, x2) / f_2(x2)

for x2 in (0.5, 1.0, 5.0):
    print(f"f(x1=0.5 | x2={x2}) = {f_1g2(0.5, x2):.4f}")
P_1g2 = quad(lambda x1: f_1g2(x1, 1.0), 1, np.inf)[0]
print(f"P(X1 >= 1 | X2 = 1) = {P_1g2:.2f}  (= P(X1 >= 1))")

# %% [part] Independence @ Example: Lifetime of Thermostat #5

# %% [markdown]
# $X_1,X_2$ are independent if $f(x_1,x_2)=f_1(x_1)f_2(x_2)$; then $F(x_1,x_2)=F_1(x_1)F_2(x_2)$.

# %%
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 5, 6), np.linspace(0, 30, 7))
gap_pdf = np.max(np.abs(f(x1_grid, x2_grid) - np.exp(-x1_grid/mu_1)/mu_1 * np.exp(-x2_grid/mu_2)/mu_2))
gap_cdf = np.max(np.abs(F(x1_grid, x2_grid) - F_1(x1_grid) * F_2(x2_grid)))
print(f"max |f - f_1 f_2| = {gap_pdf:.1e},   max |F - F_1 F_2| = {gap_cdf:.1e}")

# %% [part] Union of events @ Example: Lifetime of Thermostat #6

# %% [markdown]
# $\mathbb{P}(X\in\mathcal{A}_1\cup\mathcal{A}_2)=\mathbb{P}(X\in\mathcal{A}_1)+\mathbb{P}(X\in\mathcal{A}_2)-\mathbb{P}(X\in\mathcal{A}_1\cap\mathcal{A}_2)$
# with $\mathcal{A}_1=\{x_1\geq 1\}$ and $\mathcal{A}_2=\{x_2\geq 1\}$.

# %%
P_A1 = P_box(1, np.inf, 0, np.inf)
P_A2 = P_box(0, np.inf, 1, np.inf)
P_A1_and_A2 = P_box(1, np.inf, 1, np.inf)
P_A1_or_A2 = P_A1 + P_A2 - P_A1_and_A2
print(f"P(A1) = {P_A1:.2f}, P(A2) = {P_A2:.2f}, P(A1 & A2) = {P_A1_and_A2:.2f}")
print(f"P(A1 or A2) = {P_A1_or_A2:.2f}")

# Monte Carlo check with S samples from the joint pdf (independent exponentials)
np.random.seed(0)
S = 100_000
x_omega = np.column_stack([np.random.exponential(mu_1, S), np.random.exponential(mu_2, S)])
P_hat = np.mean((x_omega[:, 0] >= 1) | (x_omega[:, 1] >= 1))
print(f"Empirical estimate P_hat(A1 or A2) = {P_hat:.2f}")
