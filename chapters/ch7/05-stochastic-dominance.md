# Stochastic Dominance

* Given that there is a wide range of possible risk measures, we can ask ourselves:

{% hint style="info" %}
What are desirable properties of a risk measure?

What is a proper risk measure?
{% endhint %}

* An important concept that arises here is that of *stochastic dominance* (SD).
* SD provides a formal framework to compare RVs in a consistent manner.
* We say that $$Y$$ stochastically dominates $$Y'$$ (written as $$Y\preceq Y'$$) if:

  $$
  \mathbb{P}(Y\leq \bar{y})\geq \mathbb{P}(Y'\leq \bar{y})\;\; \textrm{for all}\;\;  \bar{y}\in \mathcal{D}_Y
  $$

* Notation $$Y\preceq Y'$$ is generalization of $$y\leq y'$$ (used for comparing deterministic vars).
* Sense of better defined by SD is *probabilistic* in nature.
* SD says that $$Y$$ is better $$Y'$$ if prob of $$Y$$ taking a value less than $$\bar{y}$$ is higher than prob of $$Y'$$ being below same threshold (for all the possible threshold values $$\bar{y}$$).
* We can also express SD using the cdf as

  $$
  F_{Y}(\bar{y})\geq F_{Y'}(\bar{y}) \; \textrm{for all}\;  \bar{y}\in\mathcal{D}_Y.
  $$

* In simple terms, $$Y\preceq Y'$$ indicates that cdf of $$Y$$ is above that of $$Y'$$ for all $$\bar{y}$$.

* It is important to note how SD differs from other ways to compare RVs.
* We often think that a decision dominates another if this *always* better; this implies:

  $$
  y_{\omega}\leq y'_\omega,\quad \omega \in \Omega
  $$

* This requirement is quite strict and uncommon in practice.
* SD tells us is that outcomes of $$Y$$ should be lower than threshold $$\bar{y}$$ more often than outcomes of $$Y$$. This is more common.
* A risk-conscious DM would require that $$\mathbb{P}(Y\leq \bar{y})\geq \mathbb{P}(Y'\leq \bar{y})$$ *for a single* threshold value $$\bar{y}$$. SD requires that this holds for all thresholds.
* There are *relaxed* versions of SD that are also often used; e.g., might require that  $$\mathbb{P}(Y\leq \bar{y})\geq \mathbb{P}(Y'\leq \bar{y})$$ holds *for some* $$\bar{y}$$.
* SD makes it clear that, to *properly* compare RVs, it is necessary to compare their *density functions*; moreover, comparison is based on cdfs  (not pdfs).
* Comparing cdfs is more intuitive.
