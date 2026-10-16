# Independence

Conditional pdfs tell us how knowledge in one RV resolves uncertainty in another (i.e., how much knowledge of one is embedded in the other).

{% hint style="info" %}
So what if knowledge of one does not resolve uncertainty of the other?

This gives rise to concept of *independence*.
{% endhint %}

* RVs $$X_1$$ and $$X_2$$ are said to be independent if:

  $$
  \begin{aligned}
  f(x_1|x_2)&=f_1(x_1),\; x_1\in \mathcal{D}_1\\
  f(x_2|x_1)&=f_2(x_2),\; x_2\in \mathcal{D}_2
  \end{aligned}
  $$

* This implies that:

  $$
  {f(x_1,x_2)}={f_2(x_2)}f_1(x_1),\quad (x_1,x_2)\in \mathcal{D}
  $$

* For discrete $$X$$ this is equivalent to:

  $$
  \begin{aligned}
  \mathbb{P}(X_1=x_1\,\&\,X_2=x_2)=\mathbb{P}(X_1=x_1)\mathbb{P}(X_2=x_2)
  \end{aligned}
  $$

* For continuous $$X$$ this is interpreted in infinitesimal sense.

## Example: Lifetime of Thermostat

Consider the thermostat example and address the following:

{% hint style="info" %}
* Prove that lifetimes $$X_1$$ and $$X_2$$ are independent and provide a physical explanation.
* Prove that joint cdf is the product of marginal cdfs.
{% endhint %}

* Condition for independence is $$f(x_1,x_2)=f_1(x_1)f_2(x_2)$$.
* From previous examples we have that:

  $$
  \begin{aligned}
  f(x_1,x_2)&=\frac{1}{\mu_1\cdot \mu_2}e^{-(x_1/\mu_1+x_2/\mu_2)}=\left(\frac{1}{\mu_1}e^{-x_1/\mu_1}\right)\left(\frac{1}{\mu_2}e^{-x_2/\mu_2}\right),
  \end{aligned}
  $$

   and,

  $$
  \begin{aligned}
  f_1(x_1)f_2(x_2)&=\left(\frac{1}{\mu_1}e^{-x_1/\mu_1}\right)\left(\frac{1}{\mu_2}e^{-x_2/\mu_2}\right).
  \end{aligned}
  $$

* Failure of components might be independent if they do not interact physically.
* Joint cdf is $$F(x_1,x_2)=(1-e^{-x_1/\mu_1})(1-e^{-x_2/\mu_2})$$ and marginals are $$F_1(x_1)=1-e^{-x_1/\mu_1}$$ and $$F_2(x_2)=1-e^{-x_2/\mu_2}$$.
* We thus have $$F(x_1,x_2)=F_1(x_1)\cdot F_2(x_2)$$ and this implies $$\mathbb{P}(X_1\leq x_1\; \& \;X_2\leq x_2)=\mathbb{P}(X_1\leq x_1)\cdot \mathbb{P}(X_2\leq x_2)$$.

#### Python implementation: Independence

$$X_1,X_2$$ are independent if $$f(x_1,x_2)=f_1(x_1)f_2(x_2)$$; then $$F(x_1,x_2)=F_1(x_1)F_2(x_2)$$.

```python
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 5, 6), np.linspace(0, 30, 7))
gap_pdf = np.max(np.abs(f(x1_grid, x2_grid) - np.exp(-x1_grid/mu_1)/mu_1 * np.exp(-x2_grid/mu_2)/mu_2))
gap_cdf = np.max(np.abs(F(x1_grid, x2_grid) - F_1(x1_grid) * F_2(x2_grid)))
print(f"max |f - f_1 f_2| = {gap_pdf:.1e},   max |F - F_1 F_2| = {gap_cdf:.1e}")
```

```
max |f - f_1 f_2| = 3.5e-18,   max |F - F_1 F_2| = 0.0e+00
```
