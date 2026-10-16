# Coherent Risk Measures

* There are a number of properties that a proper (a.k.a. coherent) risk measure should satisfy.
* These properties have been proposed based on usefulness in *practical applications* and based on mathematical consistency.
* Interestingly and, as we we will see shortly, these properties are analogous to properties of vector norms.
* One can think of the following properties as a set of rules (a.k.a. mathematical axioms) that give us guidance into how to select a suitable risk measure.
* Similar types of axioms exist for defining proper measures of fairness.

We say that the risk measure $$\rho$$ is *proper* if it satisfies the properties:

* Monotonicity:

   If $$Y\preceq Y'$$ then $$\rho(Y)\leq \rho(Y')$$.

   If $$Y$$ stochastically dominates $$Y'$$, then its risk should also be lower.

* Translation Invariance:

   $$\rho(Y+c)=\rho(Y)+c$$ for $$c\in \mathbb{R}$$.

   Adding a constant should not affect risk.

* Positive Homogeneity:

   $$\rho(c\cdot Y)=c\cdot \rho(Y)$$ for $$c\in \mathbb{R}_+$$.

   Scaling an RV by a constant should not affect risk (e.g., risk of a cost expressed in USD or Euros should be the same up to a factor).

* Subadditivity:

   $$\rho(Y+Y')\leq \rho(Y)+\rho(Y')$$.

   The risk of a combined pair of RVs should not exceed sum of their individual risks.

## Properties of Risk Measures: Expected Value

We can now formally analyze if risk measures are proper or not.

* $$\mathbb{E}[Y]$$ captures the magnitude of $$Y$$ and has an intuitive interpretation.
* $$\mathbb{E}[Y]$$ is a proper risk measure; as such, it is consistent with SD.
* $$\mathbb{E}[Y]$$ ignores outcomes with extreme values.

## Properties of Risk Measures: Variance

* $$\mathbb{V}[Y]$$ is a measure of variability/uncertainty and is thus intuitive.
* Variance is often used in conjunction with expected value: $$\mathbb{E}[Y]+\kappa\mathbb{V}[Y]$$.
* Mean-variance was proposed by Markowitz (1950s) and became standard in finance.
* Reducing variance reduces small/large outcomes (reduces left/right tails).
* Symmetric penalization is often undesirable (only interested in  right tail).
* Variance is *not proper*; it does not satisfy monotonicity (not consistent with SD).

## Properties of Risk Measures: Expected Loss

* Expected loss (CVaR) overcomes deficiencies of variance and has become widely used in finance.
* Expected loss of $$Y$$ at a given probability $$\alpha$$ is:

  $$
  \mathbb{E}[Y|Y\geq \mathbb{Q}_Y(\alpha)]
  $$

* Recall that $$\alpha$$-quantile of $$Y$$ is threshold value $$\bar{y}$$ satisfying

  $$
  \mathbb{P}(Y\leq \bar{y})=\alpha
  $$

* Expected loss is expected value of costs above $$\bar{y}=\mathbb{Q}_Y(\alpha)$$.
* Expected loss can also be written as:

  $$
  \mathbb{E}[(Y-\mathbb{Q}_Y(\alpha))_+]
  $$

* CVaR captures magnitude of outcomes of high value, while ignoring those of small magnitude (asymmetric measure).
* This provides important benefit over the variance, as it only focuses on right tail.
* If $$\alpha=0$$, quantile $$\mathbb{Q}_Y(\alpha)$$ is min value of $$Y$$ $$\implies$$ CVaR is $$\mathbb{E}[Y]$$.
* If $$\alpha=1$$, quantile $$\mathbb{Q}_Y(\alpha)$$ is max value of $$Y$$ $$\implies$$ CVaR is $$\max Y$$.
* CVaR is a proper risk measure.
* CVaR captures size of losses above quantile.
* A caveat of CVaR is that it does not offer direct control on probability of loss.

## Properties of Risk Measures: Probability of Loss

* The probability of loss is one of most widely used measures of risk.
* The meaning of probability of loss is intuitive in many applications and is a widely used of notion of risk.
* This measure does not take into consideration the actual magnitude of the losses.
* Probability of loss is *not coherent*;  it does not satisfy translation invariance.
* However, probability of loss is directly compatible with SD (compares probabilities).

## Risk Measures as Norms

* Coherency of risk measures resemble coherency of *vector norms* (e.g., triangle inequality is subadditivity).
* Norms allow us to measure and compare vectors.

   How can we tell if vector $$\mathbf{y}=(y_1,y_2,...,y_S)$$ is better than $$\mathbf{y}'=(y_1',y_2',...,y_S')$$?

* We can say that $$\mathbf{y}$$ is better than $$\mathbf{y}'$$ if total magnitude of its entries is lower:

  $$
  \sum_{i=1}^Sy_i\leq \sum_{i=1}^Sy_i'.
  $$

   Can be written as $$\displaystyle \frac{1}{S}\sum_{i=1}^Sy_i\leq \frac{1}{S}\sum_{i=1}^Sy_i'$$  (analogous to mean and $$\ell_1$$ norm $$\|\mathbf{y}\|_1$$).

* We can say that $$\mathbf{y}$$ is better than $$\mathbf{y}'$$ if:

  $$
  \max_i\; y_i\leq \max_i\; y_i'.
  $$

   Analogous to worst-case measure and to an $$\ell_\infty$$ norm $$\|\mathbf{y}\|_\infty$$.

* We can say that $$\mathbf{y}$$ is better than $$\mathbf{y}'$$ if:

  $$
  y_i\leq y_i'\; \textrm{for all} \; i=1,...,S.
  $$

   Analogous to say that RV $$Y$$ is *always better* than the RV $$Y'$$.

* This comparison is *not consistent* because te vector entries of $$\mathbf{y}$$ and $$\mathbf{y}'$$ are not arranged in order.
* More consistent approach would first  arrange entries in decreasing order and then compare entries.
* This is *precisely analogous* to SD; we establish a threshold on entries and count how many entries are below that threshold value.
* Another possibility is to arrange entries in increasing order and sum the largest $$k$$ entries. This is *directly analogous to CVaR*.

## Example: Risk Measures as Norms

Consider the vector $$y=(100,70,50,20,10)$$

* Mean is:

   $$(1/5)\cdot (100+70+50+20+10)=50$$

* Worst-Case is: $$100$$

The CVaR values for vector is:

* If $$\alpha=5/5$$ then CVaR is:

   $$(1/5)\cdot (100+70+50+20+10)=50$$

* If $$\alpha=4/5$$ then CVaR is:

   $$(1/4)\cdot (100+70+50+20)=60$$

* If $$\alpha=3/5$$ then CVaR is:

   $$(1/3)\cdot (100+70+50)=73$$

* If $$\alpha=1/5$$ then CVaR is:

   $$(1/1)\cdot (100)=100$$

Note how the CVaR has mean and worst-case values as extremes.

#### Python implementation

{% hint style="success" %}
Notebook [`ch7_risk_measures_norms.ipynb`](https://colab.research.google.com/github/vzavala/StatsBook-GitBook/blob/main/notebooks/ch7_risk_measures_norms.ipynb) — open it in Colab and run the cells alongside this section.
{% endhint %}

Risk measures as norms (written for the book to accompany the slide example). For
$$\mathbf{y}=(100,70,50,20,10)$$ the slide's $$\mathrm{CVaR}$$ at $$\alpha=k/S$$ is the average of the $$k$$ largest
entries: $$k=S$$ gives the mean ($$\ell_1$$-like) and $$k=1$$ the worst case ($$\ell_\infty$$-like).
Python names: `y`, `S`, `CVaR(y, k)`.

```python
import numpy as np

y = np.array([100, 70, 50, 20, 10])
S = len(y)

def CVaR(y, k):
    """Average of the k largest entries of y."""
    return np.sort(y)[::-1][:k].mean()

print(f"mean = {y.mean():.0f},  worst case = {y.max():.0f}")
for k in range(S, 0, -1):
    print(f"alpha = {k}/{S}:  CVaR = {CVaR(y, k):.1f}")
```

```
mean = 50,  worst case = 100
alpha = 5/5:  CVaR = 50.0
alpha = 4/5:  CVaR = 60.0
alpha = 3/5:  CVaR = 73.3
alpha = 2/5:  CVaR = 85.0
alpha = 1/5:  CVaR = 100.0
```
