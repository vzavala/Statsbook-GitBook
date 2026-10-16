# Motivation

* We have used *statistical techniques* to model known/unknown aspects of a system.
* We now want to use these capabilities to *make decisions*.
* Accounting for uncertainty is *essential* when making decisions (ensure robustness).
* Making decisions under uncertainty boils down to *comparing* RVs and to *shaping* the probability distribution of RVs.

* Discussion will address following specific questions:
   * What can happen if we ignore uncertainty?
   * How do we measure risk/uncertainty?
   * How can we obtain a decision that anticipates and responds to uncertainty?
   * How can we compare random variables?
   * How can we shape probability distribution of random variables?
* These questions are studied in area of *stochastic optimization*.
* This is a broad area that has applications in engineering, science, and finance.

* Consider system:

  $$
  Y=\varphi(X,u)
  $$

* $$X=(X_1,...,X_n)$$ is multivariate input RV and $$Y$$ is univariate output  RV.
* System model is $$\varphi$$ and we interpret this as performance function (e.g., cost).
* Realizations of $$X$$  propagate through system to generate realizations of $$Y$$:

  $$
  y_\omega=\varphi(x_\omega,u),\; \omega \in \mathcal{S}
  $$

* Uncertainty in $$X$$ propagates through system to generate variability in $$Y$$.
* e.g., input can be prices of energy that affect profit of a process.
* Can use decision $$u$$ to protect system against variability of $$X$$.
