# Shaping Formulations and Flexibility

## Stochastic Optimization: Shaping Formulations

* Goal of stochastic optimization is to *shape* distribution of $$Y(u)$$.
* Instead of using risk measures, consider a *target* $$Y^*$$ that captures *ideal behavior*.
* We aim to minimize distance between $$Y(u)$$ and target $$Y^*$$.
* Distance between RVs can be measured as:

  $$
  \mathbb{D}(Y(u),{Y}^*)=\int_{y\in \mathcal{D}_Y}|F_{Y(u)}(y)-F_{Y^*}(y)|dy.
  $$

* This is known as *Wasserstein distance*.
* Formulate a stochastic optimization problem that minimizes the distance:

  $$
  \begin{aligned}
  \mathop{\textrm{min}}_{u\in \mathcal{U}}\;&\;  \mathbb{D}(Y(u),{Y}^*).
  \end{aligned}
  $$

* Formulation is analogous to how optimal trajectories are obtained (e.g., for rockets).
* e.g., compute a trajectory that is as close as possible to target trajectory.

## Quantifying Flexibility and Robustness

* We are often interested in exploring how *flexible/robust* a system is.
* Consider performance $$\varphi(X,u)$$; operation is *feasible* if event $$\varphi(X,u)\leq \bar{\varphi}$$ holds.
* Conversely, operation is deemed *infeasible* if event $$\varphi(X,u)> \bar{\varphi}$$ holds.
* Want to determine prob that system has feasible operation under decision $$u$$:

  $$
  \delta(u)=\mathbb{P}(\varphi(X,u)\leq \bar{\varphi}).
  $$

* $$\delta(u)\in [0,1]$$ is a *measure of flexibility/robustness*.
* We can find decision $$u$$ that maximizes flexibility $$\delta(u)$$ by solving:

  $$
  u^*=\mathop{\textrm{argmax}}_{u\in \mathcal{U}}\; \delta(u).
  $$

* Consider now that system is described by *performance functions* $$\varphi_k(X,u),\, k\in \mathcal{K}$$.
* We can measure the flexibility of this system by using probability:

  $$
  \delta(u)=\mathbb{P}(\varphi_k(X,u)\leq \bar{\varphi}_k,\; k\in \mathcal{K}).
  $$
