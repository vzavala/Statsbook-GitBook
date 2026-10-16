# Nomenclature: math and Python

The Python code uses the same symbols as the slides wherever Python allows it. This page lists the
general rules, how `scipy.stats` parameters map to the slides' notation, and (below) the symbol table
of every notebook in the book.

## General conventions

| Slides | Python | Notes |
|---|---|---|
| random variable $$X$$, realizations $$x_\omega$$ | `x_omega` | array of observations; `x` is a grid where a pdf/cdf is evaluated |
| observation index $$\omega$$, sample set $$\mathcal{S}$$, size $$S$$ | `omega` (or loop index `w`), `S` | — |
| pdf $$f(x)$$, cdf $$F(x)$$, threshold $$t$$ | `f`, `F`, `t` | — |
| empirical estimates $$\hat{f}$$, $$\hat{F}$$, $$\hat{\mathbb{E}}_X$$, $$\hat{\theta}$$ | `f_hat`, `F_hat`, `E_hat`, `theta_hat` | a hat becomes the suffix `_hat` |
| $$\mathbb{E}_X$$, $$\mathbb{V}_X$$, $$\mathbb{SD}_X$$ | `E_X`, `V_X`, `SD_X` | — |
| quantile $$\mathbb{Q}_X(\alpha)$$ | `Q`, or `X.ppf(alpha)` | — |
| Greek letters $$\mu,\sigma,\alpha,\beta,\xi,\theta,\kappa,\gamma$$ | `mu`, `sigma`, `alpha`, `beta`, `xi`, `theta`, `kappa`, `gamma` | — |
| $$\lambda$$ | `lam` | `lambda` is a reserved word in Python |
| subscripts $$\theta_0$$, $$X_1$$, $$\Sigma_{12}$$ | `theta[0]`, `X1` / `x1`, `Sigma[0, 1]` | Python indexes from 0 |
| conditional quantities $$\mu_{1\mid 2}$$, $$\Sigma_{2\mid 1}$$, $$f(x_1\mid x_2)$$ | `mu_1g2`, `Sigma_2g1`, `f_1g2` | `g` reads "given" |
| vectors and matrices $$\mathbf{y}$$, $$\mathbf{X}$$, $$\mathbf{W}$$ | `y`, `X`, `W` | — |
| $$\mathrm{Cov}[X]$$, $$\mathrm{PCorr}[X]$$ | `Cov`, `PCorr` (`Cov_hat`, ...) | sample covariance uses $$1/S$$ as on the slides |
| decision $$u$$, decision set $$\mathcal{U}$$, optimum $$u^*$$ | `u`, `U`, `u_star` | — |
| risk measure $$\rho(Y(u))$$ | `rho` | — |

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


### Chapter 1: Introduction

#### [`ch1_mixer_example`](chapters/ch1/01-motivation.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\omega$$ | `omega` | observation (time) index |
| $$S$$ | `S` | number of observations |
| $$f_1(\omega)$$, $$f_2(\omega)$$, $$f_3(\omega)$$ | `f1`, `f2`, `f3` | measured flows [gpm] |
| $$\epsilon(\omega)$$ | `epsilon` | conservation mismatch $$f_3-f_1-f_2$$ |

#### [`ch1_gibbs_data`](chapters/ch1/01-motivation.md)

| Slides | Python | Meaning |
|---|---|---|
| $$P$$ | `P` | pressure [bar] |
| $$T$$ | `T` | temperature [K] |
| $$\xi$$ | `xi` | conversion [-] |

#### [`ch1_gibbs_example`](chapters/ch1/02-random-variables.md)

| Slides | Python | Meaning |
|---|---|---|
| $$X$$, $$x_\omega$$ | `x_omega` | reactor pressure (RV) and its realizations [bar] |
| $$Y(u)$$, $$Y(u')$$, $$y_\omega$$ | `y_low`, `y_high` | conversion (RV) and its realizations at $$u$$, $$u'$$ [%] |
| $$\mathcal{S}$$, $$S$$ | `S` | set and number of observations |
| $$\Delta x$$ | `dx` | bin size of the empirical pdf |
| $$\hat{f}(x)$$, $$\hat{F}(x)$$ | `f_hat`, `F_hat` | empirical pdf and cdf |
| $$u$$, $$u'$$ | `u_low`, `u_high` | operating temperature (decision) [K] |

#### [`ch1_pdf_cdf_discrete`](chapters/ch1/03-types-of-random-variables.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathcal{D}$$ | `D` | domain $$\{-1,0,1,2\}$$ |
| $$f(x)$$ | `f` | pdf on the domain |
| $$x_\omega,\ \omega\in\Omega$$ | `x_omega` | the 10 realizations |
| $$t$$, $$F(t)$$ | `t`, `F` | cdf evaluated on a grid of $$t$$ |

#### [`ch1_pdf_cdf_continuous`](chapters/ch1/03-types-of-random-variables.md)

| Slides | Python | Meaning |
|---|---|---|
| $$f(x)$$, $$F(t)$$ | `f`, `F` | pdf and cdf |
| $$x_\omega$$, $$S$$ | `x_omega`, `S` | realizations and their number |
| $$\Delta x$$ | `dx` | discretization (bin) size |
| $$\hat{f}$$, $$\hat{F}$$ | `f_hat`, `F_hat` | discrete (empirical) approximations |

#### [`ch1_mean_var_descriptors`](chapters/ch1/04-from-data-to-models.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_\omega$$, $$S$$ | `x_omega`, `S` | realizations and their number |
| $$\hat{\mathbb{E}}_X$$, $$\hat{\mathbb{V}}_X$$, $$\hat{\mathbb{SD}}_X$$ | `E_hat`, `V_hat`, `SD_hat` | empirical mean, variance, SD |
| $$\hat{f}(x)$$, $$\hat{F}(x)$$ | `f_hat` (histogram), `F_hat` | empirical pdf and cdf |


### Chapter 2: Univariate Random Variables

#### [`ch2_example_uniform`](chapters/ch2/02-discrete-uniform.md)

| Slides | Python | Meaning |
|---|---|---|
| $$X$$, $$x_\omega$$ | `x_omega` | weekly status: $$0$$ = fails, $$1$$ = works |
| $$S$$ | `S` | number of weeks observed (156) |
| $$a$$, $$b$$, $$n$$ | `a`, `b`, `n` | uniform parameters, $$n=b-a+1$$ |
| $$\hat{\mathbb{P}}(X=x)$$ | `P_hat[x]` | empirical pdf |

#### [`ch2_compare_hypergeometric`](chapters/ch2/04-hypergeometric.md)

| Slides | Python | Meaning |
|---|---|---|
| $$N_t$$, $$N_y$$ | `N_t`, `N_y` | population size and number of type-$$Y$$ items |
| $$n$$ | `n` | sample size |
| $$p=N_y/N_t$$ | `p` | fraction of type-$$Y$$ items |
| $$\mathcal{D}$$, $$f(x)$$, $$F(x)$$ | `D`, `f`, `F` | domain, pdf, cdf |

#### [`ch2_example_hypergeometric`](chapters/ch2/04-hypergeometric.md)

| Slides | Python | Meaning |
|---|---|---|
| $$N_t$$ | `N_t` | weeks in the record (population) |
| $$N_y$$ | `N_y` | weeks in failure mode |
| $$p=N_y/N_t$$ | `p` | fraction of failure weeks |
| $$n$$ | `n` | number of inspections |
| $$f(x)=\mathbb{P}(X=x)$$ | `f(x, n)` | hypergeometric pdf |

#### [`ch2_example_poisson`](chapters/ch2/06-poisson.md)

| Slides | Python | Meaning |
|---|---|---|
| $$z$$ | `z` | interval length (days in a month) |
| $$x_{Jan},x_{Feb},x_{Mar}$$ | `x_omega` | failures recorded per month |
| $$\hat{\mathbb{E}}_X$$, $$\hat{\lambda}$$ | `E_hat`, `lam_hat` | empirical mean and estimate of $$\lambda$$ |
| $$f(x)=\lambda^x e^{-\lambda}/x!$$ | `f(x)` | Poisson pdf |

#### [`ch2_mixing_gaussians`](chapters/ch2/08-gaussian.md)

| Slides | Python | Meaning |
|---|---|---|
| $$f_1$$ | `f1` | deterministic input flow (10 gpm) |
| $$F_2\sim\mathcal{N}(20,1)$$ | `mu_2`, `sigma_2` | uncertain input flow |
| $$\kappa$$ | `kappa` | valve coefficient |
| $$F_3=f_1+\kappa F_2$$ | `mu_3`, `sigma_3` | output flow, $$\mathcal{N}(10+20\kappa,\kappa^2)$$ |
| $$z_\alpha=\mathbb{Q}_Z(\alpha)$$ | `z_alpha` | standard-normal quantile |

#### [`ch2_example_lognormal`](chapters/ch2/09-log-normal.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathbb{M}_X$$ | `M_X` | median particle size (200 $$\mu m$$) |
| $$\mathbb{Q}_X(\alpha)$$, $$\alpha$$ | `Q_alpha`, `alpha` | 90% quantile (400 $$\mu m$$) |
| $$\mu$$, $$\sigma$$ | `mu`, `sigma` | log-normal parameters |
| $$F(x)$$, $$f(x)$$ | `F`, `f` | cdf and pdf |

#### [`ch2_example_chisquare`](chapters/ch2/12-chi-squared.md)

| Slides | Python | Meaning |
|---|---|---|
| $$F_3\sim\mathcal{N}(\mu,\sigma^2)$$ | `mu`, `sigma` | output flow of the mixer |
| $$\alpha$$ | `alpha` | probability outside the interval |
| $$Y\sim\chi^2(1)$$, $$\mathbb{Q}_Y(1-\alpha)$$ | `Q_Y` | chi-squared quantile |
| $$z_{\alpha/2}=\sqrt{\mathbb{Q}_Y(1-\alpha)}$$ | `z_alpha_2` | critical value |

#### [`ch2_example_weibull`](chapters/ch2/13-weibull.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\beta$$, $$\xi$$ | `beta`, `xi` | scale [m/s] and shape [-] |
| $$\mathbb{M}_X$$ | `M_X` | median wind speed |
| $$\mathbb{E}_X$$, $$\mathbb{MO}_X$$ | `E_X`, `MO_X` | mean and mode |
| $$x_\omega$$ | `x_omega` | simulated 10-min wind speeds |


### Chapter 3: Multivariate Random Variables

#### [`ch3_gibbs_conditional`](chapters/ch3/02-joint-pdfs-and-cdfs.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_{\omega,1}$$, $$x_{\omega,2}$$ | `x1`, `x2` | pressure [bar], conversion [-] |
| $$\mathcal{A}_1$$, $$\mathcal{A}_2$$ | `in_A1`, `in_A2` | indicators $$\mathbf{1}[x_\omega\in\mathcal{A}_i]$$ |
| $$\hat{\mathbb{P}}(X\in\mathcal{A}_1\cap\mathcal{A}_2)$$ | `P_A1_and_A2` | empirical joint probability |
| $$\hat{\mathbb{P}}(X\in\mathcal{A}_1\mid X\in\mathcal{A}_2)$$ | `P_A1_given_A2` | empirical conditional probability |

#### [`ch3_thermostat_lifetime`](chapters/ch3/02-joint-pdfs-and-cdfs.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mu_1$$, $$\mu_2$$ | `mu_1`, `mu_2` | mean lifetimes [yr] |
| $$f(x_1,x_2)$$, $$F(t_1,t_2)$$ | `f`, `F` | joint pdf and cdf |
| $$f_1$$, $$f_2$$, $$F_1$$, $$F_2$$ | `f_1`, `f_2`, `F_1`, `F_2` | marginal pdfs and cdfs |
| $$f(x_1\mid x_2)$$ | `f_1g2` | conditional pdf ("1 given 2") |
| $$\mathbb{P}(X\in\mathcal{A})$$ | `P_A` | probability of event $$\mathcal{A}$$ |

#### [`ch3_flow_cytometer_discrete`](chapters/ch3/02-joint-pdfs-and-cdfs.md)

| Slides | Python | Meaning |
|---|---|---|
| $$X_1$$ (FSC), $$X_2$$ (SSC) | `x1`, `x2` | log front and side scatter of each droplet |
| $$S$$ | `S` | number of droplets |
| $$\mathcal{D}_1=\mathcal{D}_2=\{L,M,H\}$$ | `edges_1`, `edges_2` | bin edges of the three categories |
| $$\hat{f}(x_1,x_2)$$ | `f_hat` | empirical joint pdf (rows: SSC, columns: FSC) |
| $$\hat{f}_1$$, $$\hat{f}_2$$ | `f1_hat`, `f2_hat` | empirical marginal pdfs |
| $$\hat{f}(x_1\mid x_2)$$, $$\hat{f}(x_2\mid x_1)$$ | `f_hat_1g2`, `f_hat_2g1` | empirical conditional pdfs |

#### [`ch3_flow_cytometer_empirical`](chapters/ch3/02-joint-pdfs-and-cdfs.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_{\omega,1}$$ (FSC), $$x_{\omega,2}$$ (SSC) | `x1`, `x2` | realizations |
| $$\hat{\mathcal{D}}$$ | `x1.min()`, ... | empirical domain (bounding box of the data) |
| $$\Delta x_1=\Delta x_2$$ | `dx` | bin size (0.05) |
| $$\hat{f}(x_1,x_2)$$ | `f_hat` | empirical joint pdf |

#### [`ch3_gibbs_covariance`](chapters/ch3/07-summarizing-statistics.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_\omega\in\mathbb{R}^5$$, $$S$$ | `x_omega` (S x 5), `S` | pressure, conversion, outflows of CO, H$$_2$$, CH$$_3$$OH |
| $$\hat{m}_i=\hat{\mathbb{E}}[X_i]$$ | `m_hat` | empirical means |
| $$\hat{\mathrm{Cov}}[X]$$ | `Cov_hat` | sample covariance (with $$1/S$$) |
| $$\hat{\mathrm{PCorr}}[X]=D^{-1}\hat{\mathrm{Cov}}[X]D^{-1}$$ | `PCorr_hat`, `D` | Pearson correlation |

#### [`ch3_gaussian_geometry`](chapters/ch3/08-multivariate-gaussian.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mu$$, $$\Sigma$$ | `mu`, `Sigma` | mean and covariance |
| $$1-\alpha$$ | `1 - alpha` | probability of the region (95%) |
| $$\mathbb{Q}(1-\alpha)$$ of $$\chi^2(n)$$ | `Q_n` | quantile defining the ellipse $$\mathcal{E}$$ and its bounding box $$\mathcal{B}$$ |
| $$\mathbb{Q}(1-\alpha)$$ of $$\chi^2(1)$$ | `Q_1` | quantile defining the marginal box |


### Chapter 4: Estimation for Random Variables

#### [`ch4_gaussian_moments_example`](chapters/ch4/02-method-of-moments.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_\omega$$, $$S$$ | `x_omega`, `S` | data and sample size |
| $$\hat{\mathbb{E}}_X$$, $$\hat{\mathbb{V}}_X$$ | `E_hat`, `V_hat` | sample mean and variance |
| $$\hat{\mu}=\hat{\mathbb{E}}_X$$, $$\hat{\sigma}^2=\hat{\mathbb{V}}_X$$ | `mu_hat`, `sigma_hat` | moment-matching estimates |

#### [`ch4_leastsquares_example`](chapters/ch4/03-least-squares-method.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_\omega$$, $$S$$ | `x_omega`, `S` | data and sample size |
| $$t_k$$, $$N$$ | `t`, `N` | cdf thresholds |
| $$\hat{F}_k$$ | `F_hat` | empirical cdf at $$t_k$$ |
| $$\theta=(\xi,\beta)$$ | `theta = (xi, beta)` | Weibull shape and scale |
| $$F_X(t\mid\theta)$$ | `F_model(t, theta)` | model cdf $$1-e^{-(t/\beta)^\xi}$$ |

#### [`ch4_example_mle_exp`](chapters/ch4/04-maximum-likelihood-estimation.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_\omega$$, $$S$$ | `x_omega`, `S` | data and sample size |
| $$\beta$$ | `beta` | grid of candidate scale values |
| $$\log L(\beta)=-S\log\beta-\frac{1}{\beta}\sum_\omega x_\omega$$ | `logL` | log likelihood |
| $$\hat{\beta}=\hat{\mathbb{E}}_X$$ | `beta_hat` | MLE |

#### [`ch4_montecarlo_approx`](chapters/ch4/06-monte-carlo-approximations.md)

| Slides | Python | Meaning |
|---|---|---|
| $$S$$ | `S_values` | sample sizes |
| $$\hat{\mathbb{E}}^S_{\phi(X)}=\frac{1}{S}\sum_\omega\phi(x_\omega)$$ | `E_hat`, `E_log_hat` | MC estimates of $$\mathbb{E}[X]$$, $$\mathbb{E}[\log X]$$ |
| $$\hat{\mathbb{V}}^S$$ | `V_hat` | MC estimate of $$\mathbb{V}[e^{-X}+X^2]$$ |
| $$\hat{\mathbb{P}}(X\in\mathcal{A})$$ | `P_hat` | MC estimate of $$\mathbb{P}(X\leq 1)$$ |

#### [`ch4_clt_example`](chapters/ch4/07-central-limit-theorem.md)

| Slides | Python | Meaning |
|---|---|---|
| $$S$$ | `S` | sample size of each average |
| $$\bar{X}=\hat{\mathbb{E}}^S_X$$ | `x_bar` | sample averages (one per repetition) |
| $$e=\mathbb{E}[X]$$, $$v^2=\mathbb{V}[X]$$ | `e`, `v2` | true mean and variance |
| $$\mathcal{N}(e, v^2/S)$$ | `norm(e, sqrt(v2/S))` | CLT limit |

#### [`ch4_evt_example`](chapters/ch4/08-extreme-value-theorem.md)

| Slides | Python | Meaning |
|---|---|---|
| $$X\sim\mathcal{N}(2,1)$$ | `mu`, `sigma` | underlying RV |
| $$X^S_{max}=\max\{X_1,\dots,X_S\}$$ | `x_max` | sample maxima (one per repetition) |
| $$\mathrm{GEV}(a,b,c)$$ | `a`, `b`, `c` | location, scale, shape |


### Chapter 5: Estimation for Structural Models

#### [`ch5_linear_est_example`](chapters/ch5/02-scalar-linear-models.md)

| Slides | Python | Meaning |
|---|---|---|
| $$(x_\omega,y_\omega)$$, $$S$$ | `x_omega`, `y_omega`, `S` | observation pairs |
| $$\theta$$, $$\sigma$$ | `theta`, `sigma` | true slope and noise SD ($$\sigma^2=0.0625$$) |
| $$\hat{\theta}=\sum_\omega x_\omega y_\omega/\sum_\omega x_\omega^2$$ | `theta_hat` | MLE (least squares) |
| $$\hat{y}_\omega=\hat{\theta}x_\omega$$, $$\hat{\epsilon}_\omega$$ | `y_hat`, `eps_hat` | predictions and residuals |
| $$SSE(\theta)$$ | `SSE` | sum of squared errors |

#### [`ch5_heat_capacity_lin_est`](chapters/ch5/03-multidimensional-linear-models.md)

| Slides | Python | Meaning |
|---|---|---|
| $$X$$ (temperature), $$Y$$ (heat capacity) | `x`, `y` | inputs [K] and outputs [cal/mol-K] |
| $$\mathbf{X}=[1,\ x,\ x^2]$$, $$\mathbf{y}$$ | `X`, `y` | data matrix and output vector |
| $$\theta=(\theta_0,\theta_1,\theta_2)$$ | `theta` (true), `theta_hat` | parameters |
| $$\hat{\mathbf{y}}$$, $$\hat{\epsilon}$$ | `y_hat`, `eps_hat` | predictions and residuals |
| $$\hat{\sigma}$$ | `sigma_hat` | estimated noise SD |
| $$\mathrm{Cov}[\hat{\theta}]$$, $$\mathcal{I}(\theta)$$ | `Cov_theta`, `FI` | parameter covariance and Fisher information |

#### [`ch5_heat_capacity_uncertainty`](chapters/ch5/05-uncertainty-quantification.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\hat{\theta}$$, $$\mathrm{Cov}[\hat{\theta}]$$ | `theta_hat`, `Cov_theta` | estimate and its covariance |
| $$q_{n+1}$$ | `q_np1` | 0.95 quantile of $$\chi^2(n+1)$$ |
| $$m_i=\sqrt{q_{n+1}\mathrm{Cov}[\hat{\theta}]_{ii}}$$ | `m_theta` | parameter CI half-widths |
| $$\mathrm{Cov}[\hat{\mathbf{y}}]=\mathbf{X}\mathrm{Cov}[\hat{\theta}]\mathbf{X}^T$$ | `Cov_y` | prediction covariance |
| $$q_S$$, $$m_\omega$$ | `q_S`, `m_y` | 0.95 quantile of $$\chi^2(S)$$ and prediction CI half-widths |

#### [`ch5_hougen_watson`](chapters/ch5/07-estimation-for-general-nonlinear-models.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_{1,\omega},x_{2,\omega},x_{3,\omega}$$ | `X[:, 0]`, `X[:, 1]`, `X[:, 2]` | partial pressures ($$S=13$$ experiments) |
| $$y_\omega$$ | `y` | reaction rate |
| $$\theta=(\theta_0,\dots,\theta_4)$$ | `theta[0]`, ..., `theta[4]` | parameters |
| $$m(\theta)$$, $$SSE(\theta)=\frac{1}{2}\lVert\mathbf{y}-\mathbf{m}(\theta)\rVert^2$$ | `m(theta, X)`, `SSE(theta)` | model and objective |

#### [`ch5_hougen_watson_constraints`](chapters/ch5/08-prior-knowledge.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\underline{\theta}$$, $$\overline{\theta}$$ | `theta_lb`, `theta_ub` | bounds $$(0,\dots,0)$$ and $$(2,\dots,2)$$ |
| $$\nabla_\theta\mathbf{m}(\hat{\theta})$$ | `J` | Jacobian returned by the solver |
| $$\mathcal{H}(\hat{\theta})\approx\nabla_\theta\mathbf{m}^T\nabla_\theta\mathbf{m}$$ | `H` | Gauss-Newton Hessian |
| $$\lambda$$ | `lam` | eigenvalues of $$\mathcal{H}$$ |
| $$\mathrm{Cov}[\hat{\theta}]$$, $$\mathbb{V}[\hat{\theta}]$$ | `Cov_theta`, `V_theta` | covariance and variances |

#### [`ch5_bayes_estimation`](chapters/ch5/09-bayesian-estimation.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\theta\sim\mathcal{N}(\bar{\theta},\Sigma_\theta)$$ | `theta_bar`, `Sigma_theta` | prior (0, 1) |
| $$y\sim\mathcal{N}(\theta x,\sigma^2)$$ | `sigma2` | likelihood noise variance (1) |
| $$\log f(\theta)$$, $$\log f(y\mid\theta)$$, $$\log f(\theta\mid y)$$ | `log_f_theta`, `log_f_y_theta`, `log_f_theta_y` | log prior, log likelihood, log posterior (up to a constant) |


### Chapter 6: Statistical Data Analysis and Learning — Part I: Dimensionality Reduction

#### [`ch6_gibbs_pca`](chapters/ch6a/02-principal-component-analysis.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathbf{X}$$ ($$S\times n$$) | `X` | data matrix with columns $$(T, P, C)$$, centered and scaled |
| $$\boldsymbol{\Sigma}=\mathbf{X}^T\mathbf{X}$$ | `Sigma` | kernel matrix |
| $$\mathbf{W}$$, $$\lambda_k$$ | `W`, `lam` | eigenvectors and eigenvalues (descending) |
| $$\alpha_k$$ | `alpha` | spectrum (% of information) |
| $$\mathbf{T}=\mathbf{X}\mathbf{W}$$ | `T_pc` | principal components (scores) |

#### [`ch6_gibbs_pca_estimation`](chapters/ch6a/03-principal-component-estimation.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathbf{X}=[1, X_1,\dots,X_5]$$ | `X` | intercept, $$P$$, $$CO$$, $$H_2$$, $$CH_3OH$$ flows, $$T$$ ($$S=250$$) |
| $$\mathbf{y}$$ | `y` | conversion |
| $$\hat{\theta}$$, $$\mathrm{Cov}(\hat{\theta})$$ | `theta_hat`, `Cov_theta` | standard estimate |
| $$\mathbf{W}_1$$, $$\boldsymbol{\Lambda}_1$$, $$n_1$$ | `W1`, `Lam1`, `n1` | retained eigenvectors/eigenvalues |
| $$\mathbf{T}_1=\mathbf{X}\mathbf{W}_1$$ | `T1` | reduced inputs |
| $$\hat{\gamma}$$, $$\hat{\theta}=\mathbf{W}_1\hat{\gamma}$$ | `gamma_hat`, `theta_hat_pca` | PCA estimate |

#### [`ch6_gibbs_pca_estimation_sparse`](chapters/ch6a/04-sparse-pca.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathbf{W}$$ | `W` | eigenvectors of $$\mathbf{X}^T\mathbf{X}$$ (rows: 1, $$X_1,\dots,X_5$$) |
| $$\tilde{\mathbf{W}}$$, $$\kappa$$ | `W_sparse`, `kappa` | sparsified eigenvectors and sparsity weight |

#### [`ch6_gibbs_pca_estimation_svd`](chapters/ch6a/05-singular-value-decomposition.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathbf{X}=\mathbf{U}\mathbf{S}\mathbf{V}^T$$ | `U`, `s`, `Vt` | SVD (`s` holds the singular values $$s_j$$; `S` is the sample size) |
| $$\lambda_i=s_i^2$$ | `lam` | eigenvalues of $$\mathbf{X}^T\mathbf{X}$$ |
| $$\mathrm{rank}(\mathbf{X}[:,1..k])$$ | `np.linalg.matrix_rank` | rank as columns are added |


### Chapter 6: Statistical Data Analysis and Learning — Part II: Signal Processing

#### [`ch6_example_convolutions`](chapters/ch6b/02-1d-convolution-operations.md)

| Slides | Python | Meaning |
|---|---|---|
| $$f_j$$ | `f` | signal |
| $$g_j$$ | `g` | filter (weights over a window of $$N$$ entries) |
| $$\varphi=f*g$$ | `phi` | convolution $$\varphi_k=\sum_j g_j f_{k+j}$$ |
| $$N$$, $$\alpha$$, $$\sigma=(N-1)/(2\alpha)$$ | `N`, `alpha`, `sigma` | window length, width factor, Gaussian SD |

#### [`ch6_fourier_transform`](chapters/ch6b/03-1d-fourier-transforms.md)

| Slides | Python | Meaning |
|---|---|---|
| $$f(t)=\sin(2\pi\omega_1t)+\sin(2\pi\omega_2t)$$ | `f`, `omega_1`, `omega_2` | signal (30 and 60 Hz) |
| $$\hat{f}_k=\sum_j f_je^{-i2\pi kj/N}$$ | `f_hat = np.fft.fft(f)` | discrete FT |
| $$\omega_k$$, $$\lvert\hat{f}_k\rvert$$ | `omega`, `power(f_hat)` | frequencies and one-sided magnitude |
| $$\mathcal{F}\{f*g\}=\mathcal{F}\{f\}\cdot\mathcal{F}\{g\}$$ | `phi_hat = f_hat * g_hat` | convolution theorem |

#### [`ch6_image_filters_2D`](chapters/ch6b/06-2d-fourier-transforms.md)

| Slides | Python | Meaning |
|---|---|---|
| $$f$$, $$\hat{f}=\mathcal{F}\{f\}$$ | `f`, `f_hat` | image and its 2D FT |
| $$g_1$$, $$g_2$$ ($$\sigma=1$$, $$\sigma=10$$) | `g1`, `g2` | Gaussian filters (same size as the image) |
| $$\hat{f}_i=\hat{f}\cdot\hat{g}_i$$ | `f1_hat`, `f2_hat` | filtered images in the frequency domain |
| $$f_i=\mathcal{F}^{-1}\{\hat{f}_i\}$$ | `f1`, `f2` | filtered images |


### Chapter 6: Statistical Data Analysis and Learning — Part III: Machine Learning Models

#### [`ch6_gibbs_logistic`](chapters/ch6c/01-classification-models.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_\omega=(P_\omega,C_\omega)$$ | `X` (columns `P`, `C`) | features: pressure and conversion |
| $$y_\omega\in\{0,1\}$$ | `y` | label: 1 = normal (low $$T$$), 0 = failure (high $$T$$) |
| $$\theta$$, $$\theta^Tx$$ | `theta`, `evidence` | parameters and evidence |
| $$m(x,\theta)=\mathbb{P}(Y=1\mid x,\theta)$$ | `m_hat` | classification probability |
| $$\hat{y}_\omega$$ | `y_hat` | predicted label |

#### [`ch6_hougen_watson_kernel`](chapters/ch6c/02-kernel-models.md)

| Slides | Python | Meaning |
|---|---|---|
| $$x_\omega$$, $$y_\omega$$ | `X`, `y` | partial pressures and reaction rates |
| $$k(x_\omega,x_{\omega'})=\exp(-\gamma\lVert x_\omega-x_{\omega'}\rVert^2)$$ | `rbf(xi, xj, gamma)` | RBF kernel |
| $$\mathbf{K}$$, $$\lambda$$ | `K`, `lam` | kernel matrix and regularization |
| $$\hat{\mathbf{y}}=\mathbf{K}(\mathbf{K}+\lambda I)^{-1}\mathbf{y}$$ | `y_hat` | kernel prediction |

#### [`ch6_gibbs_kriging`](chapters/ch6c/03-kriging.md)

| Slides | Python | Meaning |
|---|---|---|
| $$X=(P,T)$$, $$Y=C$$ | `X`, `y` | inputs (pressure, temperature) and conversion |
| $$k(x,x',\gamma)$$, $$\hat{\gamma}$$, $$\sigma^2$$ | `kernel`, `gp.kernel_` | Matern kernel plus noise term, and their MLE values |
| $$m(x)$$, $$\sigma(x)$$ | `m_x`, `sigma_x` | predictive mean and SD |
| $$\epsilon_\omega$$ | `eps_gp`, `eps_lin` | residuals of kriging and linear models |

#### [`ch6_neuralnet_basis`](chapters/ch6c/04-neural-nets.md)

| Slides | Python | Meaning |
|---|---|---|
| $$a_j=\theta^I_{j,1}x+\theta^I_{j,0}$$ | `theta_I1[j] * x + theta_I0[j]` | evidence of perceptron $$j$$ |
| $$h_j(a)=1/(1+e^{-a_j})$$ | `h(a)` | sigmoidal activation (basis function) |
| $$\hat{y}=\sum_j\theta^O_jh(a_j)$$ | `m(theta, x)` | NN output |
| $$\theta=(\theta^I_{\cdot,1},\theta^O,\theta^I_{\cdot,0})$$ | `theta` | the 18 parameters ($$N=6$$ perceptrons) |
| $$\mathcal{H}\approx\mathcal{J}^T\mathcal{J}$$ | `H` | Gauss-Newton Hessian |

#### [`ch6_cnn_endotoxins_caseI`](chapters/ch6c/05-convolutional-neural-nets.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathbf{X}_\omega$$ | `X[w, 0]` | $$50\times 50$$ grayscale scatter field |
| $$y_\omega$$ | `y` | class: `bipolar` (no endotoxin) or `radial` (endotoxin) |
| $$\Theta^I$$, $$\theta^I$$ | `cnn[0].weight`, `cnn[0].bias` | $$3\times 3$$ filter and bias of the convolutional layer |
| $$\mathbf{A}=\mathbf{X}*\Theta^I+\theta^I$$, $$\mathbf{Z}=h(\mathbf{A})$$ | `cnn[0]`, `cnn[1]` | evidence and ReLU activation |
| $$NN(\mathbf{w},\gamma)$$ | `cnn[3]` | fully connected layer (2 classes, softmax) |
| $$L(\beta)$$ | `loss_fn` | cross-entropy (log) loss |

#### [`ch6_cnn_endotoxins_caseII`](chapters/ch6c/05-convolutional-neural-nets.md)

| Slides | Python | Meaning |
|---|---|---|
| $$\mathbf{X}_\omega$$ | `X[w, 0]` | $$50\times 50$$ grayscale scatter field |
| $$y_\omega$$ | `y` | class: EC, PA, or SM (*E. coli*, *P. aeruginosa*, *S. minnesota*) |
| $$\Theta^I_j$$, $$\theta^I_j$$ | `cnn[0].weight`, `cnn[0].bias` | two $$3\times 3$$ filters and biases |
| $$\mathbf{A}=\mathbf{X}*\Theta^I+\theta^I$$, $$\mathbf{Z}=h(\mathbf{A})$$ | `cnn[0]`, `cnn[1]` | evidence and ReLU activation |
| $$NN(\mathbf{w},\gamma)$$ | `cnn[3]` | fully connected layer (3 classes, softmax) |
| $$L(\beta)$$ | `loss_fn` | cross-entropy (log) loss |


### Chapter 7: Decision-Making Under Uncertainty

#### [`ch7_generator_det`](chapters/ch7/02-deterministic-decision-making.md)

| Slides | Python | Meaning |
|---|---|---|
| $$L$$, realizations | `L_omega` | random power load (Weibull, scale 10, shape 2) |
| $$u$$, $$\mathcal{U}=\{0,1,\dots,10\}$$ | `u`, `U` | generator capacity and options |
| $$R=\max(L-u,0)$$ | `R` | residual demand bought from the grid |
| $$Y(u)=\alpha_R R^2+\alpha_u u^2$$ | `cost(u, L)` | total cost ($$\alpha_R=5$$, $$\alpha_u=15$$) |
| $$r=\max_\omega r_\omega$$ | `L_peak` | representative (peak) load |
| $$u_D^*$$ | `u_D` | deterministic decision |

#### [`ch7_npv_comparison`](chapters/ch7/03-attitudes-towards-risk.md)

| Slides | Python | Meaning |
|---|---|---|
| $$c_I$$, $$c_{II}$$ | `c_I`, `c_II` | capital [thousand USD] (25 and 50) |
| $$R_I\sim\mathcal{N}(0.5,0.1)$$, $$R_{II}\sim\mathcal{N}(0.4,0.2)$$ | `R_I`, `R_II` | annual rate of return (mean, SD) |
| $$i$$ | `i` | interest rate (5%, 10 years) |
| $$NPV$$, i.e. $$Y(u)$$ and $$Y(u')$$ | `NPV_I`, `NPV_II` | net present value (negative = revenue) |

#### [`ch7_generator_twostage`](chapters/ch7/09-formulations-with-recourse.md)

| Slides | Python | Meaning |
|---|---|---|
| $$X=(X_1,X_2)$$ | `X1`, `X2` | two random loads ($$S=200$$ samples) |
| $$u\in\mathcal{U}=\{1,10\}$$ | `u`, `U` | capacity (first stage, preemptive) |
| $$W(X)\in\mathcal{W}(u)$$ | `W` | generator output (second stage, recourse) |
| $$R=X_1+X_2-W(X)$$ | `R` | residual load |
| $$Y(u,W(X))=\alpha_R R^2+\alpha_u u^2$$ | `cost` | cost |

#### [`ch7_bayesopt_yield`](chapters/ch7/11-bayesian-optimization.md)

| Slides | Python | Meaning |
|---|---|---|
| $$u=(u_1,u_2,u_3)\in[-1,1]^3$$ | `u` | scaled temperature, time, acid concentration |
| $$\varphi(u)$$ | `phi(u)` | yield (a quadratic model plays the "black box") |
| $$\mathcal{D}_\ell$$ | `data_u`, `data_y` | experiments so far |
| $$m_\ell(u)$$, $$\sigma_\ell(u)$$ | `gp.predict(u, return_std=True)` | kriging mean and SD |
| $$AF_\ell(u)=m_\ell(u)+\kappa\sigma_\ell(u)$$ | `AF(u)`, `kappa` | acquisition function ($$\kappa=0.2$$) |
