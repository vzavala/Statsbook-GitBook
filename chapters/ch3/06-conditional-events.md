# Conditional Events

* Focused on probs that are conditional to simple events (e.g., $$X_1=x_1$$,  $$X_2\leq x_2$$).
* Probs that are conditional of general events can be derived from domain properties.
* Return to definition of conditional probability:

  $$
  \begin{aligned}
  \mathbb{P}(X\in \mathcal{A}_1|X\in \mathcal{A}_2)=\frac{\mathbb{P}(X\in\mathcal{A}_1\;\&\; X\in \mathcal{A}_2)}{\mathbb{P}(X\in \mathcal{A}_2)}.
  \end{aligned}
  $$

   where $$\mathcal{A}_1$$ and $$\mathcal{A}_2$$ are subdomains of $$\mathcal{D}$$ that capture specific events.

* $$\mathbb{P}(X\in \mathcal{A}_1|X\in \mathcal{A}_2)$$ is prob that event $$\mathcal{A}_1$$ occurs given that $$\mathcal{A}_2$$ has occurred.
* $$\mathbb{P}(X\in\mathcal{A}_1\;\&\; X\in \mathcal{A}_2)$$ is joint prob of $$\mathcal{A}_1$$ and $$\mathcal{A}_2$$, and $$\mathbb{P}(X\in \mathcal{A}_2)$$ is marginal prob.

* For continuous $$X$$, prob $$\mathbb{P}(X\in \mathcal{A}_1|X\in \mathcal{A}_2)$$ is:

  $$
  \mathbb{P}(X\in \mathcal{A}_1|X\in \mathcal{A}_2)=\frac{\displaystyle\int_{x\in \mathcal{A}_1\cap\mathcal{A}_2}f(x)dx}{\displaystyle\int_{x\in \mathcal{A}_2}f(x)dx}.
  $$

* Recall that joint event $$\{X\in\mathcal{A}_1\;\&\; X\in \mathcal{A}_2\}$$ is equivalent to $$X\in \mathcal{A}_1\cap \mathcal{A}_2$$.
* Conditional prob $$\mathbb{P}(X_1=x_1|X_2=x_2)$$ is a special case with events of the form $$\mathcal{A}_1=\{X_1=x_1\;\&\; X_2\in \mathcal{D}_2\}$$ and $$\mathcal{A}_2=\{X_1\in \mathcal{D}_1\; \& \; X_2=x_2\}$$.

* Recall that, if $$\mathcal{A}_1$$ and $$\mathcal{A}_2$$ are *independent* events,  then:

  $$
  \begin{aligned}
  \mathbb{P}(X\in\mathcal{A}_1\;\&\; X\in \mathcal{A}_2)=\mathbb{P}(X\in \mathcal{A}_1)\cdot \mathbb{P}(X\in \mathcal{A}_2).
  \end{aligned}
  $$

* This implies that:

  $$
  \int_{x\in \mathcal{A}_1\cap\mathcal{A}_2}f(x)dx=\left(\int_{x\in \mathcal{A}_1}f(x)dx\right)\cdot \left(\int_{x\in \mathcal{A}_2}f(x)dx\right)
  $$

* This explains why joint cdf is product of marginal cdfs for $$X_1,X_2$$ independent.

* Prob that event $$\mathcal{A}_1$$ *or* $$\mathcal{A}_2$$ occur is given by:

  $$
  \mathbb{P}(X\in \mathcal{A}_1\cup \mathcal{A}_2)=\mathbb{P}(X\in \mathcal{A}_1)+\mathbb{P}(X\in \mathcal{A}_2)- \mathbb{P}(X\in \mathcal{A}_1\cap \mathcal{A}_2).
  $$

* This is computed by using joint and marginal pdfs of $$X$$ as:

  $$
  \mathbb{P}(X\in \mathcal{A}_1\cup \mathcal{A}_2)=\int_{x\in \mathcal{A}_1}f(x)dx+\int_{x\in \mathcal{A}_2}f(x)dx- \int_{x\in \mathcal{A}_1\cap\mathcal{A}_2}f(x)dx.
  $$

* Pdfs and event properties can be used to compute probs of complex events.

## Example: Lifetime of Thermostat

{% hint style="info" %}
Compute probability that sensor or processor last more than a year.
{% endhint %}

* We build events $$\mathcal{A}_1=\{x_1\geq 1\;\&\; x_2\in \mathcal{D}_2\}$$ and $$\mathcal{A}_2=\{x_1\in \mathcal{D}_1\;\& \;x_2\geq 1\}$$, which give $$\mathcal{A}_1\cap\mathcal{A}_2=\{x_1\geq 1\; \& \; x_2\geq 1\}$$, and $$\mathcal{A}_1\cup\mathcal{A}_2=\{x_1\geq 1\; \textrm{or} \; x_2\geq 1\}$$.
* Computing probabilities we have:

  $$
  \begin{aligned}
  \int_{x\in \mathcal{A}_1}f(x)dx&=\int_{1}^{\infty}\int_{0}^{\infty}f(x_1,x_2)dx_1dx_2=0.37\\
  \int_{x\in \mathcal{A}_2}f(x)dx&=\int_{0}^{\infty}\int_{1}^{\infty}f(x_1,x_2)dx_1dx_2=0.90\\
  \int_{x\in \mathcal{A}_1\cap\mathcal{A}_2}f(x)dx&=\int_{1}^{\infty}\int_{1}^{\infty}f(x_1,x_2)dx_1dx_2=0.33.
  \end{aligned}
  $$

   and thus $$\mathbb{P}(X\in \mathcal{A}_1\cup \mathcal{A}_2)=0.37+0.90-0.33=0.94$$.

* There is a high probability that either one of them  lasts more than a year.

#### Python implementation: Union of events

$$\mathbb{P}(X\in\mathcal{A}_1\cup\mathcal{A}_2)=\mathbb{P}(X\in\mathcal{A}_1)+\mathbb{P}(X\in\mathcal{A}_2)-\mathbb{P}(X\in\mathcal{A}_1\cap\mathcal{A}_2)$$
with $$\mathcal{A}_1=\{x_1\geq 1\}$$ and $$\mathcal{A}_2=\{x_2\geq 1\}$$.

```python
P_A1 = P_box(1, np.inf, 0, np.inf)
P_A2 = P_box(0, np.inf, 1, np.inf)
P_A1_and_A2 = P_box(1, np.inf, 1, np.inf)
P_A1_or_A2 = P_A1 + P_A2 - P_A1_and_A2
print(f"P(A1) = {P_A1:.2f}, P(A2) = {P_A2:.2f}, P(A1 & A2) = {P_A1_and_A2:.2f}")
print(f"P(A1 or A2) = {P_A1_or_A2:.2f}")

# Monte Carlo check with S samples from the joint pdf (independent exponentials)
np.random.seed(0)
S = 100_000
x_omega = np.column_stack([np.random.exponential(mu_1, S), np.random.exponential(mu_2, S)])
P_hat = np.mean((x_omega[:, 0] >= 1) | (x_omega[:, 1] >= 1))
print(f"Empirical estimate P_hat(A1 or A2) = {P_hat:.2f}")
```

```
P(A1) = 0.37, P(A2) = 0.90, P(A1 & A2) = 0.33
P(A1 or A2) = 0.94
Empirical estimate P_hat(A1 or A2) = 0.94
```
