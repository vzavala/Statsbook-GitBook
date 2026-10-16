# %% [markdown]
# Images as matrices (Shiyi (Amy) Qin and Victor Z, UW-Madison, 2020). We use the $10\times 10$
# matrix $\mathbf{X}$ shown on the slide, a noisy version $\mathbf{X}_\epsilon$ (`X_eps`), and the rank-1
# approximation $s_1\mathbf{u}_1\mathbf{v}_1^T$ from the expansion $\mathbf{X}=\sum_i s_i\mathbf{u}_i\mathbf{v}_i^T$.

# %%
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
              [0, 0, 1, 2, 4, 4, 2, 1, 0, 0],
              [0, 1, 2, 6, 9, 9, 6, 2, 1, 0],
              [0, 1, 4, 9, 14, 14, 9, 4, 1, 0],
              [0, 1, 4, 9, 14, 14, 9, 4, 1, 0],
              [0, 1, 2, 6, 9, 9, 6, 2, 1, 0],
              [0, 0, 1, 2, 4, 4, 2, 1, 0, 0],
              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
np.random.seed(2024)
X_eps = X + np.random.normal(0, 1, X.shape)

def rank1(A):
    U, s, Vt = np.linalg.svd(A)
    return s, s[0] * np.outer(U[:, 0], Vt[0])

s, X1 = rank1(X)
s_eps, X1_eps = rank1(X_eps)
print("nonzero singular values of X:", np.round(s[s > 1e-10], 4))
print("rank(X) =", np.linalg.matrix_rank(X), "  rank(X_eps) =", np.linalg.matrix_rank(X_eps))

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
for ax, A, title in ((axes[0, 0], X, r'$\mathbf{X}$'), (axes[0, 1], X_eps, r'$\mathbf{X}_\epsilon$'),
                     (axes[1, 0], X1, r'$s_1\mathbf{u}_1\mathbf{v}_1^T$ of $\mathbf{X}$'),
                     (axes[1, 1], X1_eps, r'$s_1\mathbf{u}_1\mathbf{v}_1^T$ of $\mathbf{X}_\epsilon$')):
    ax.imshow(A, cmap='gray_r')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()
