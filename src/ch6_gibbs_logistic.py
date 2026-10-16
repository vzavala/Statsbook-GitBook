# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# Logistic classification for fault detection in the Gibbs reactor
# (Victor Z and Ugochukwu Ikegwu, UW-Madison, 2024).
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $x_\omega=(P_\omega,C_\omega)$ | `X` (columns `P`, `C`) | features: pressure and conversion |
# | $y_\omega\in\{0,1\}$ | `y` | label: 1 = normal (low $T$), 0 = failure (high $T$) |
# | $\theta$, $\theta^Tx$ | `theta`, `evidence` | parameters and evidence |
# | $m(x,\theta)=\mathbb{P}(Y=1\mid x,\theta)$ | `m_hat` | classification probability |
# | $\hat{y}_\omega$ | `y_hat` | predicted label |

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = np.vstack((np.loadtxt(path+'Data/gibbs_hightemp_class.dat'),
                  np.loadtxt(path+'Data/gibbs_lowtemp_class.dat')))
S = len(data)
y = (data[:, 0] == 583.15).astype(int)          # 1 = normal (low T), 0 = failure
np.random.seed(1)
P = data[:, 1] + np.random.randn(S) * 0.1
C = data[:, 2] + np.random.randn(S) * 0.1
X = np.column_stack((P, C))

plt.scatter(P[y == 0], C[y == 0], c='k', s=20, label='$y=0$ (failure)')
plt.scatter(P[y == 1], C[y == 1], c='w', edgecolors='0.5', s=20, label='$y=1$ (normal)')
plt.xlabel('$P$'); plt.ylabel('$C$'); plt.legend(loc='lower right'); plt.grid(True)
plt.show()

# %% [markdown]
# MLE: $\max_\theta \sum_\omega y_\omega\log m(x_\omega,\theta)+(1-y_\omega)\log(1-m(x_\omega,\theta))$.
# (`penalty=None` gives plain MLE; scikit-learn regularizes by default.)

# %%
model = LogisticRegression(penalty=None, max_iter=1000).fit(X, y)
theta = np.concatenate([model.intercept_, model.coef_.ravel()])   # (theta_0, theta_P, theta_C)
evidence = theta[0] + X @ theta[1:]
m_hat = 1 / (1 + np.exp(-evidence))                               # = model.predict_proba(X)[:, 1]
y_hat = (m_hat >= 0.5).astype(int)
print("theta_hat =", np.round(theta, 3))
print(f"accuracy = {np.mean(y_hat == y):.2%}  ({np.sum(y_hat == y)} of {S})")

def plot_confusion(ax, cm, title):
    ax.imshow(cm, cmap='Greys', alpha=0.3)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha='center', va='center', fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
plot_confusion(axs[0], confusion_matrix(y, y_hat), 'confusion matrix')
axs[1].plot(m_hat, 'ko', markerfacecolor='w', markersize=3)
axs[1].set_xlabel(r'$\omega$'); axs[1].set_ylabel(r'$\mathbb{P}(y_\omega=1\mid x_\omega,\hat{\theta})$')
axs[1].grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Train/test split (60/40) to check that the classifier generalizes.

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=42)
model_tr = LogisticRegression(penalty=None, max_iter=1000).fit(X_tr, y_tr)
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
plot_confusion(axs[0], confusion_matrix(y_tr, model_tr.predict(X_tr)), 'training')
plot_confusion(axs[1], confusion_matrix(y_te, model_tr.predict(X_te)), 'test')
plt.tight_layout()
plt.show()
