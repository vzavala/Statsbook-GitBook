# %% [markdown]
# Setting up a pressure alarm with a uniform model $X\sim\mathcal{U}_c(a,b)$, $a=63$, $b=234$ bar
# (written for the book to accompany the slide example).
# Python names: `a`, `b`, cdf `F(x)`, quantile `Q(alpha)`.

# %%
from scipy.stats import uniform

a, b = 63, 234   # bar
X = uniform(loc=a, scale=b - a)

def F(x):
    """Cdf F(x) = (x - a)/(b - a)."""
    return X.cdf(x)

def Q(alpha):
    """Quantile Q_X(alpha) = a + alpha (b - a)."""
    return X.ppf(alpha)

print(f"P(X <= 100) = F(100) = {F(100):.2f}")
print(f"Alarm threshold Q_X(0.99) = {Q(0.99):.1f} bar  (check: a + 0.99(b-a) = {a + 0.99*(b-a):.1f})")
