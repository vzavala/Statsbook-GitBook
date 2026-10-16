# Nomenclature: math and Python

The Python code uses the same symbols as the slides wherever Python allows it. This page lists the
general rules, how `scipy.stats` parameters map to the slides' notation, and (below) the symbol table
of every notebook in the book.

## General conventions

| Slides | Python | Notes |
|---|---|---|
| random variable $$X$$, realizations $$x_\omega$$ | `x_omega` | array of observations; `x` is a grid where a pdf/cdf is evaluated |
| observation index $$\omega$$, sample set $$\mathcal{S}$$, size $$S$$ | `omega` (or loop index `w`), `S` | |
| pdf $$f(x)$$, cdf $$F(x)$$, threshold $$t$$ | `f`, `F`, `t` | |
| empirical estimates $$\hat{f}$$, $$\hat{F}$$, $$\hat{\mathbb{E}}_X$$, $$\hat{\theta}$$ | `f_hat`, `F_hat`, `E_hat`, `theta_hat` | a hat becomes the suffix `_hat` |
| $$\mathbb{E}_X$$, $$\mathbb{V}_X$$, $$\mathbb{SD}_X$$ | `E_X`, `V_X`, `SD_X` | |
| quantile $$\mathbb{Q}_X(\alpha)$$ | `Q`, or `X.ppf(alpha)` | |
| Greek letters $$\mu,\sigma,\alpha,\beta,\xi,\theta,\kappa,\gamma$$ | `mu`, `sigma`, `alpha`, `beta`, `xi`, `theta`, `kappa`, `gamma` | |
| $$\lambda$$ | `lam` | `lambda` is a reserved word in Python |
| subscripts $$\theta_0$$, $$X_1$$, $$\Sigma_{12}$$ | `theta[0]`, `X1` / `x1`, `Sigma[0, 1]` | Python indexes from 0 |
| conditional quantities $$\mu_{1\mid 2}$$, $$\Sigma_{2\mid 1}$$, $$f(x_1\mid x_2)$$ | `mu_1g2`, `Sigma_2g1`, `f_1g2` | `g` reads "given" |
| vectors and matrices $$\mathbf{y}$$, $$\mathbf{X}$$, $$\mathbf{W}$$ | `y`, `X`, `W` | |
| $$\mathrm{Cov}[X]$$, $$\mathrm{PCorr}[X]$$ | `Cov`, `PCorr` (`Cov_hat`, ...) | sample covariance uses $$1/S$$ as on the slides |
| decision $$u$$, decision set $$\mathcal{U}$$, optimum $$u^*$$ | `u`, `U`, `u_star` | |
| risk measure $$\rho(Y(u))$$ | `rho` | |

## `scipy.stats` parameters in the slides' notation

| Slides | `scipy.stats` |
|---|---|
| $$\mathcal{N}(\mu,\sigma^2)$$ | `norm(loc=mu, scale=sigma)` (standard deviation, not variance) |
| $$\mathcal{U}_c(a,b)$$ | `uniform(loc=a, scale=b - a)` |
| $$\mathcal{B}(p)$$, $$\mathrm{Bi}(n,p)$$ | `bernoulli(p)`, `binom(n, p)` |
| $$\mathcal{H}(N_y,N_t,n,p)$$ | `hypergeom(M=N_t, n=N_y, N=n)` |
| $$\mathcal{P}(\lambda)$$ | `poisson(mu=lam)` |
| $$\mathrm{Exp}(\beta)$$ (intensity $$\eta=1/\beta$$) | `expon(scale=beta)` |
| $$\mathrm{Gamma}(\alpha,\beta)$$ | `gamma(a=alpha, scale=beta)` |
| $$\chi^2(r)$$ | `chi2(df=r)` |
| $$\mathrm{Weibull}(\xi,\beta)$$ (shape $$\xi$$, scale $$\beta$$) | `weibull_min(c=xi, scale=beta)` |
| log-normal with $$\mu,\sigma$$ | `lognorm(s=sigma, scale=np.exp(mu))` |
| $$\mathrm{GEV}(a,b,c)$$ | `genextreme(c=-c, loc=a, scale=b)` (opposite sign of the shape) |
| $$\mathcal{N}(\mu,\Sigma)$$ (multivariate) | `multivariate_normal(mean=mu, cov=Sigma)` |

## Symbols used in each notebook
