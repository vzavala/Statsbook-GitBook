# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Data reconciliation with Bayesian estimation (written for the book to accompany the slide
# example; the measurement values are illustrative). Conservation requires $Y=X$; we observe
# $(x_\omega,y_\omega)$ with error variances $\sigma_x^2$, $\sigma_y^2$ and solve
#
# $$\min_x\ \frac{1}{2\sigma_y^2}(y_\omega-x)^2+\frac{1}{2\sigma_x^2}(x-x_\omega)^2
# \quad\Rightarrow\quad \hat{x}=\frac{y_\omega+\frac{\sigma_y^2}{\sigma_x^2}x_\omega}{1+\frac{\sigma_y^2}{\sigma_x^2}}.$$
#
# Python names: `x_omega`, `y_omega`, `sigma2_x`, `sigma2_y`, `x_hat`.

# %%
import numpy as np
from scipy.optimize import minimize_scalar

x_omega, y_omega = 100.0, 104.0      # measured input and output flows (illustrative)

def x_hat_formula(sigma2_x, sigma2_y):
    r = sigma2_y / sigma2_x
    return (y_omega + r * x_omega) / (1 + r)

for sigma2_x, sigma2_y in ((1.0, 1.0), (1.0, 0.01), (0.01, 1.0)):
    obj = lambda x: (y_omega - x)**2 / (2*sigma2_y) + (x - x_omega)**2 / (2*sigma2_x)
    x_num = minimize_scalar(obj).x
    print(f"sigma2_x = {sigma2_x:5.2f}, sigma2_y = {sigma2_y:5.2f}:  "
          f"x_hat = {x_hat_formula(sigma2_x, sigma2_y):7.3f}  (numerical: {x_num:7.3f})")
