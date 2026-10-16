# Conditional PDFs and CDFs

* Want probs for $$X_1$$ *given knowledge* that $$X_2$$ takes value $$x_2$$ (or other way around).
* These probabilities are obtained from *conditional pdfs*:

  $$
  \begin{aligned}
  f(x_1|x_2)&=\frac{f(x_1,x_2)}{f_2(x_2)},\; x_1\in \mathcal{D}_1\\
  f(x_2|x_1)&=\frac{f(x_1,x_2)}{f_1(x_1)},\; x_2\in \mathcal{D}_2
  \end{aligned}
  $$

   i.e., $$f(x_1|x_2)=\mathbb{P}(X_1=x_1|X_2=x_2)$$ and $$f(x_2|x_1)=\mathbb{P}(X_2=x_2|X_1=x_1)$$.

* Note resemblance of conditional pdfs with conditional probabilities seen previously.

* These expressions can also be written as:

  $$
  \begin{aligned}
  {f_2(x_2)}f(x_1|x_2)&={f(x_1,x_2)},\; x_1\in \mathcal{D}_1\\
  {f_1(x_1)}f(x_2|x_1)&={f(x_1,x_2)},\; x_2\in \mathcal{D}_2
  \end{aligned}
  $$

* Consider we want $$\mathbb{P}(a_1\leq X_1\leq b_1|X_2=x_2)$$ this is given by:

  $$
  \begin{aligned}
  \mathbb{P}(a_1\leq X_1\leq b_1|X_2=x_2)&=\int_{a_1}^{b_1}f(x_1|x_2)dx_1
  \end{aligned}
  $$

* Joint pdfs have associated marginal cdfs $$F(x_1|x_2)$$ and $$F(x_2|x_1)$$.

## Example: Lifetime of Thermostat

{% hint style="info" %}
* Obtain the conditional pdf and cdf of component lifetimes $$X_1,X_2$$.
* Compute prob that $$X_1$$ survives after 1 year given that $$X_2$$ survives  1 year.
* Compute prob that $$X_2$$ survives after 1 year given that $$X_1$$ survives  1 year.
* Compare probs against the marginal probabilities of survival.
{% endhint %}

* Conditional pdfs are obtained from joint and marginal pdfs as:

  $$
  \begin{aligned}
  f(x_1|x_2)&=\frac{f(x_1,x_2)}{f_2(x_2)}=\frac{\frac{1}{\mu_1\cdot \mu_2}e^{-(x_1/\mu_1+x_2/\mu_2)}}{\frac{1}{\mu_2}e^{-x_2/\mu_2}}\\
  &=\frac{1}{\mu_1}e^{-x_1/\mu_1}.
  \end{aligned}
  $$

* We apply a similar procedure to obtain:

  $$
  \begin{aligned}
  f(x_2|x_1)&=\frac{1}{\mu_2}e^{-x_2/\mu_2}.
  \end{aligned}
  $$

* Note  $$f(x_1|x_2)$$ does not depend on $$x_2$$ and $$f(x_2|x_1)$$ does not depend on $$x_1$$.
* Knowing $$X_2$$ does not influence probability of failure of $$X_1$$ and viceversa.
* Note conditional pdfs are marginal pdfs of $$X_1$$ and $$X_2$$.
* Thus have $$\mathbb{P}(X_1\geq 1|X_2 = 1)=\mathbb{P}(X_1 \geq 1)=0.37$$ and $$\mathbb{P}(X_2 \geq 1|X_1 = 1)=\mathbb{P}(X_2 \geq 1)=0.90$$.

#### Python implementation: Conditional pdfs

$$f(x_1|x_2)=f(x_1,x_2)/f_2(x_2)$$ does not depend on $$x_2$$: it equals $$f_1(x_1)$$.

```python
def f_1g2(x1, x2):
    return f(x1, x2) / f_2(x2)

for x2 in (0.5, 1.0, 5.0):
    print(f"f(x1=0.5 | x2={x2}) = {f_1g2(0.5, x2):.4f}")
P_1g2 = quad(lambda x1: f_1g2(x1, 1.0), 1, np.inf)[0]
print(f"P(X1 >= 1 | X2 = 1) = {P_1g2:.2f}  (= P(X1 >= 1))")
```

```
f(x1=0.5 | x2=0.5) = 0.6065
f(x1=0.5 | x2=1.0) = 0.6065
f(x1=0.5 | x2=5.0) = 0.6065
P(X1 >= 1 | X2 = 1) = 0.37  (= P(X1 >= 1))
```
