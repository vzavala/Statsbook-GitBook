# %% [markdown]
# Waiting time at a charging station as a Gamma RV (written for the book to accompany the
# slide example). With $\alpha$ trucks in the queue and average charge time $\beta$,
#
# $$\mathbb{P}(X>x)=1-F(x)=\sum_{k=0}^{\alpha-1}\frac{1}{k!}\left(\frac{x}{\beta}\right)^k e^{-x/\beta}.$$
#
# Python names: `alpha`, `beta`, `x`; `P_exceed(x, alpha, beta)` implements the sum above.

# %%
from math import exp, factorial
from scipy.stats import gamma

def P_exceed(x, alpha, beta):
    """P(X > x) for X ~ Gamma(alpha, beta) with integer alpha."""
    return sum((x/beta)**k * exp(-x/beta) / factorial(k) for k in range(alpha))

cases = [  # (description, x [hr], alpha [trucks], beta [hr])
    ("2 trucks in queue, wait > 3 hr", 3, 2, 4),
    ("3 trucks in queue, wait > 3 hr", 3, 3, 4),
    ("1 truck in queue,  wait > 4 hr", 4, 1, 4),
    ("1 truck, new batteries (beta = 2 hr), wait > 3 hr", 3, 1, 2),
]
for text, x, alpha, beta in cases:
    print(f"{text:52s} P = {P_exceed(x, alpha, beta):.3f}   "
          f"(scipy: {gamma.sf(x, alpha, scale=beta):.3f})")
