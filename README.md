# Statistics for Chemical Engineers: From Data to Models to Decisions

**Victor M. Zavala** · Department of Chemical and Biological Engineering, University of Wisconsin-Madison

This online companion follows the lecture slides of the course and places the Python implementation
of every example right next to the concepts and equations it implements. Read a concept, see its
equations, then see the code that computes them, with its output, in the same notation.

{% hint style="info" %}
This material supports the textbook [*Statistics for Chemical Engineers*](https://www.cambridge.org/highereducation/product/A8DB49C10B02BADA637D13FC02CD62A7)
by Victor M. Zavala (Cambridge University Press, © 2025).
{% endhint %}

## How the pages are organized

Each chapter follows the slides. Wherever a slide works through an example, the page continues with a
**Python implementation** section:

* a short table that maps the symbols on the slide to the Python variable names,
* the equations of the example, each followed by the code that computes it,
* the printed results and figures produced by that code.

Notebooks that reproduce a slide figure are collapsed under **"Python: reproduce this figure"**.
Notebooks that go beyond the slides are marked **Additional example**.

Every implementation is also a Jupyter notebook (folder `notebooks/` of the repository) that you can open in
Google Colab or run locally (see [How to use this book](how-to-use.md)).

## Chapters

1. **Introduction**: random variables, data and models, decisions under uncertainty
2. **Univariate Random Variables**: discrete and continuous models
3. **Multivariate Random Variables**: joint, marginal, and conditional densities; covariance; the multivariate Gaussian
4. **Estimation for Random Variables**: moments, least squares, maximum likelihood, Monte Carlo, CLT, extreme values
5. **Estimation for Structural Models**: linear and nonlinear regression, uncertainty, prior knowledge, Bayesian estimation
6. **Statistical Data Analysis and Learning**: PCA and SVD; convolutions and Fourier transforms; classification, kernels, kriging, and neural nets
7. **Decision-Making Under Uncertainty**: risk measures, stochastic dominance, stochastic and Bayesian optimization

The [Nomenclature](nomenclature.md) page summarizes how the mathematical notation maps to Python names.

## Credits

Python notebooks by Victor M. Zavala with Ugochukwu Ikegwu, David Cole, J. E. Umaña, Jaron T.,
Leo Gonzalez, and Shiyi (Amy) Qin.
