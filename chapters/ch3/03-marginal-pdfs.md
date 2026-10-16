# Marginal PDFs

* Want probs for $$X_1$$ *regardless* of what value $$X_2$$ takes (and other way around).
* Probs are obtained from *marginal pdfs*. For continuous RV:

  $$
  \begin{aligned}
  f_1(x_1)&=\int_{x_2\in \mathcal{D}_2}f(x_1,x_2)dx_2,\; x_1\in \mathcal{D}_1\\
  f_2(x_2)&=\int_{x_1\in \mathcal{D}_1}f(x_1,x_2)dx_1,\; x_2\in \mathcal{D}_2
  \end{aligned}
  $$

* Marginal pdfs are obtained by “integrating out” effect of the RV that we ignore.
* Consider we want $$\mathbb{P}(X_1\in \mathcal{A}_1)$$ regardless of what value $$X_2$$ takes. This is:

  $$
  \begin{aligned}
  \mathbb{P}(X_1\in \mathcal{A}_1)&=\int_{x_1\in \mathcal{A}_1}\overbrace{\int_{x_2\in \mathcal{D}_2} f(x_1,x_2)dx_2}^{f_1(x_1)}dx_1\\
  &=\int_{x_1\in \mathcal{A}_1}f_1(x_1)dx_1
  \end{aligned}
  $$

* Marginal pdfs $$f_1(x_1)$$ and $$f_2(x_2)$$ have associated marginal cdfs $$F_1(x_1),F_2(x_2)$$.

## Example: Lifetime of Thermostat

{% hint style="info" %}
* Marginal pdf and cdf of component lifetimes $$X_1,X_2$$.
* Compute prob that  $$X_1$$ survives after 1 year.
* Compute prob that $$X_2$$ survives after 1 year.
{% endhint %}

* Marginal pdf for $$X_1$$ is obtained by integrating $$X_2$$ out over $$\mathcal{D}_2=[0,\infty)$$:

  $$
  \begin{aligned}
  f_1(x_1)&=\int_{0}^{\infty}f(x_1,x_2)dx_2=\int_{0}^{\infty}\left(\frac{1}{\mu_1\cdot \mu_2}e^{-(x_1/\mu_1+x_2/\mu_2)}\right)dx_2\\
  &=\frac{1}{\mu_1\cdot \mu_2}e^{-x_1/\mu_1}\left(\int_{0}^{\infty}e^{-x_2/\mu_2}dx_2\right)=\frac{1}{\mu_1}e^{-x_1/\mu_1}.
  \end{aligned}
  $$

* Applying a similar procedure we obtain:

  $$
  \begin{aligned}
  f_2(x_2)&=\frac{1}{\mu_2}e^{-x_2/\mu_2}.
  \end{aligned}
  $$

* Marginals are pdfs of exponential RVs with $$\mu_1=1$$ and $$\mu_2=10$$ years and thus:

  $$
  F_1(x_1)=1-e^{-x_1/\mu_1}\; \qquad
  F_2(x_2)=1-e^{-x_2/\mu_2}.
  $$

* Probabilities that components survive after one year are:

  $$
  \begin{aligned}
  \mathbb{P}(X_1\geq 1)&=1-F_1(1)=e^{-1/1}=0.37\\
  \mathbb{P}(X_2\geq 1)&=1-F_2(1)=e^{-1/10}=0.90.
  \end{aligned}
  $$

#### Python implementation: Marginal pdfs and survival

Integrating out the other variable, $$f_1(x_1)=\int_0^\infty f(x_1,x_2)dx_2=\frac{1}{\mu_1}e^{-x_1/\mu_1}$$
(and similarly $$f_2$$), so $$\mathbb{P}(X_i\geq 1)=1-F_i(1)=e^{-1/\mu_i}$$.

```python
def f_1(x1):
    return quad(lambda x2: f(x1, x2), 0, np.inf)[0]

def f_2(x2):
    return quad(lambda x1: f(x1, x2), 0, np.inf)[0]

print(f"f_1(0.5) = {f_1(0.5):.4f}   vs (1/mu_1) e^(-0.5/mu_1) = {np.exp(-0.5/mu_1)/mu_1:.4f}")
F_1 = lambda x1: 1 - np.exp(-x1/mu_1)
F_2 = lambda x2: 1 - np.exp(-x2/mu_2)
print(f"P(X1 >= 1) = 1 - F_1(1) = {1 - F_1(1):.2f}")
print(f"P(X2 >= 1) = 1 - F_2(1) = {1 - F_2(1):.2f}")
```

```
f_1(0.5) = 0.6065   vs (1/mu_1) e^(-0.5/mu_1) = 0.6065
P(X1 >= 1) = 1 - F_1(1) = 0.37
P(X2 >= 1) = 1 - F_2(1) = 0.90
```
