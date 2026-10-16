# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Neural net versus kriging for the Gibbs reactor (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024).
# Inputs $x_\omega=(P,T)$ (`X`, scaled), output conversion $y_\omega$ (`y`). The NN has one hidden layer
# with two perceptrons (logistic activations) and is trained by minimizing the squared loss $L(\theta)$.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

data = np.vstack((np.loadtxt(path+'Data/gibbs_hightemp_class.dat'),
                  np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')))
S = len(data)
np.random.seed(0)
X = np.column_stack((data[:, 1], data[:, 0])) + np.random.rand(S, 2) * 0.1   # (P, T)
y = data[:, 2] + np.random.rand(S) * 0.1
X = StandardScaler().fit_transform(X)

nn = MLPRegressor(hidden_layer_sizes=(2,), activation='logistic', solver='lbfgs',
                  max_iter=20000, random_state=2024).fit(X, y)
y_nn = nn.predict(X)
gp = GaussianProcessRegressor(kernel=Matern(length_scale=1, nu=1.5) + WhiteKernel(0.01), normalize_y=True,
                              n_restarts_optimizer=5, random_state=0).fit(X, y)
y_gp = gp.predict(X)
n_params = sum(w.size for w in nn.coefs_) + sum(b.size for b in nn.intercepts_)
print(f"NN parameters: {n_params};  MSE NN = {np.mean((y - y_nn)**2):.2e},  MSE kriging = {np.mean((y - y_gp)**2):.2e}")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, pred, title in ((axs[0], y_gp, 'kriging'), (axs[1], y_nn, 'neural net (2 hidden perceptrons)')):
    ax.plot(y, pred, 'ko', markerfacecolor='w', markersize=3)
    ax.plot([0, 1.1], [0, 1.1], 'k-')
    ax.set_title(title); ax.set_xlabel(r'$y_\omega$'); ax.set_ylabel(r'$\hat{y}_\omega$')
    ax.axis([0, 1.1, 0, 1.1]); ax.grid(True)
plt.tight_layout()
plt.show()
