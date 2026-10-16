# %% [markdown] [hide]
# **Setup.** Run this notebook from the `notebooks/` folder of the book
# repository (it reads files from `Data/` and `Support/`). On Google Colab,
# upload the folder to your Drive and set `path` to its location, e.g.
# `path = '/content/drive/MyDrive/StatsBook/notebooks/'`.

# %% [hide]
path = ''

# %% [markdown]
# CNN classification of flow-cytometry scatter fields: which bacterial species produced the
# endotoxin? (Python version of `ch6_cnn_endotoxins_caseII.m`, Victor Z, UW-Madison, 2020; uses PyTorch.)
#
# | Slides | Python | Meaning |
# |---|---|---|
# | $\mathbf{X}_\omega$ | `X[w, 0]` | $50\times 50$ grayscale scatter field |
# | $y_\omega$ | `y` | class: EC, PA, or SM (*E. coli*, *P. aeruginosa*, *S. minnesota*) |
# | $\Theta^I_j$, $\theta^I_j$ | `cnn[0].weight`, `cnn[0].bias` | two $3\times 3$ filters and biases |
# | $\mathbf{A}=\mathbf{X}*\Theta^I+\theta^I$, $\mathbf{Z}=h(\mathbf{A})$ | `cnn[0]`, `cnn[1]` | evidence and ReLU activation |
# | $NN(\mathbf{w},\gamma)$ | `cnn[3]` | fully connected layer (3 classes, softmax) |
# | $L(\beta)$ | `loss_fn` | cross-entropy (log) loss |
#
# As in the Matlab script: two $3\times 3$ filters with "same" padding, ReLU, a fully connected
# layer, 15 training images per class, the rest for validation, SGD with momentum (100 epochs here; the three classes are harder to separate).

# %%
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage import io
from sklearn.metrics import confusion_matrix

folder = path + 'Data/Endotoxins_CaseII/Gray'
classes = sorted(os.listdir(folder))
X, y = [], []
for c, name in enumerate(classes):
    for f in sorted(glob.glob(f'{folder}/{name}/*.png')):
        X.append(io.imread(f).astype(np.float32) / 255)
        y.append(c)
X, y = np.array(X)[:, None], np.array(y)
print("classes:", classes, " images per class:", np.bincount(y), " image size:", X.shape[2:])

rng = np.random.default_rng(0)
train = np.concatenate([rng.choice(np.where(y == c)[0], 15, replace=False) for c in range(len(classes))])
valid = np.setdiff1d(np.arange(len(y)), train)
X_t = torch.tensor(X - X[train].mean(axis=0, keepdims=True))      # zero-center (as Matlab's input layer)
y_t = torch.tensor(y)

# %%
torch.manual_seed(0)
cnn = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, padding=1),   # A = X * Theta + theta
                    nn.ReLU(),                                     # Z = h(A)
                    nn.Flatten(),                                  # feature vector w
                    nn.Linear(2 * 50 * 50, len(classes)))              # fully connected layer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)

history = []
for epoch in range(100):
    perm = torch.tensor(train)[torch.randperm(len(train))]
    for b in range(0, len(perm), 10):
        idx = perm[b:b + 10]
        optimizer.zero_grad()
        loss = loss_fn(cnn(X_t[idx]), y_t[idx])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = [(cnn(X_t[ids]).argmax(1) == y_t[ids]).float().mean().item() for ids in (train, valid)]
    history.append(acc)

history = np.array(history)
plt.plot(history[:, 0], 'k-', label='training'); plt.plot(history[:, 1], 'k--', label='validation')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.grid(True)
plt.show()
print(f"training accuracy = {history[-1, 0]:.0%},  validation accuracy = {history[-1, 1]:.0%}")

# %%
with torch.no_grad():
    y_hat = cnn(X_t).argmax(1).numpy()
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, ids, title in ((axs[0], train, 'training'), (axs[1], valid, 'validation')):
    cm = confusion_matrix(y[ids], y_hat[ids], labels=range(len(classes)))
    ax.imshow(cm, cmap='Greys', alpha=0.3)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha='center', va='center', fontsize=14)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title)
plt.tight_layout()
plt.show()

# %% [markdown]
# The learned filters $\Theta_j$ and the activated images (first filter) for nine validation images.

# %%
for j in range(2):
    print(f"Theta_{j+1} =\n", np.round(cnn[0].weight.detach().numpy()[j, 0], 3))
with torch.no_grad():
    Z = cnn[1](cnn[0](X_t[valid[:9]])).numpy()
fig, axs = plt.subplots(2, 9, figsize=(16, 4))
for k in range(9):
    axs[0, k].imshow(X[valid[k], 0], cmap='gray'); axs[0, k].set_title(classes[y[valid[k]]], fontsize=9)
    axs[1, k].imshow(Z[k, 0], cmap='gray')
for ax in axs.ravel():
    ax.axis('off')
plt.tight_layout()
plt.show()
