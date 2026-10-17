# Notes for the author

Things found while aligning the Python notebooks with the slides. Nothing here
was changed in the slides; numbers in the notebooks were left as computed.

## Chapter 1
- `ch1_gibbs_example`: with the data in `notebooks/Data`, the coefficients of
  variation are CV = 0.108 (u = 583 K) and 0.262 (u' = 613 K); the slide quotes
  0.07 and 0.10. The conclusion (u* = 583 K) is unchanged.
- `ch1_mean_var_descriptors` (original): the empirical-cdf loop overwrote the
  sample size (`N = len(t)`), so only the first 151 samples were used. Fixed.

## Chapter 2
- Slides, Discrete Uniform: `\matcal{D}` typo (rendered as `\mathcal{D}` in the book).
- Slides, Example Hypergeometric: text says N_t = 176, but the record is 156 weeks;
  with N_t = 156 the notebook reproduces P(X=10) = 8e-4, but gives P(X=5) = 0.031
  (slide: 0.1312).
- `ch2_compare_weibull` (original): shape and scale were swapped in
  `weibull_min.pdf(x, beta, scale=xi)`, so the curves did not match their labels. Fixed.
- `ch2_compare_hypergeometric` (original): `n = len(x)` overwrote the sample size
  (n became 51). Fixed.
- `ch2_example_poisson`, `ch2_example_uniform` (original): estimates came from new
  random draws; they now use the slide data (8, 12, 14 failures; 79 of 156 weeks).
- New notebooks for slide examples that had no code: `ch2_example_alarm`,
  `ch2_example_service_time`; Bernoulli example added as part 2 of `ch2_example_uniform`.
- `ch2_example_exponential` (adhesive failure) is not in the slides; shown as an
  additional example after the residence-time example. It called the scale `eta`;
  renamed to `beta` (slides: eta = 1/beta is the intensity).

## Chapter 3
- Slides, Empirical Event Probabilities: `\begin{align*}` is closed with
  `\end{align}` (the converter tolerates it).
- Slides, Example: Uncertainty Reduction in Gibbs Reactor: Sigma_2|1 is computed
  with the correlation 0.53 in place of the covariance 1.83. For a bivariate
  Gaussian the relative variance reduction is rho^2 ~ 0.28 in *both* directions,
  so the quoted 5% vs 21% (and "conversion has a larger effect") does not hold.
  `ch3_gibbs_covariance` part 2 computes both with the slide's numbers.
- `ch3_gaussian_geometry` (original): the "joint pdf" image was the product of the
  marginals (ignores the correlation the slide illustrates). Now uses the true
  joint pdf.
- `ch3_gibbs_covariance`: the slides define the sample covariance with 1/S;
  `np.cov` uses 1/(S-1). Now uses 1/S (`bias=True`).
- New notebook `ch3_thermostat_lifetime` for the six thermostat examples
  (numerical integration plus a Monte Carlo check).
- The three copies of `ellipse()` pasted into ch3 notebooks were replaced by
  `Support/ellipse.py`.

## Chapter 4
- Slides (Ch. 2 Weibull and Ch. 4 Method of Moments for Weibull): the variance is
  written beta^2 (Gamma(1+2/xi) **+** Gamma(1+1/xi)^2); it should be a minus.
  The notebook uses the minus sign.
- Slides, MC / CLT examples: the text says "Weibull(2,1)", but the figures (and the
  code) use scale beta = 2, shape xi = 1, i.e. Weibull(1, 2) in the slides' own
  Weibull(xi, beta) notation (Matlab's wblrnd takes (scale, shape)). The notebooks
  follow the figures and state the parameters explicitly.
- Slides, Example: Monte Carlo Approximations: the text lists V[exp(X)+X^2] and
  P(X >= 1) (typo "P(X>>=1)"); the figure shows V[exp(-X)+X^2] and P(X <= 1). With
  this RV V[exp(X)] is infinite, so the notebook keeps exp(-X) and P(X <= 1).
  In the slide figure, the V[...] panel is a copy of the E[log X] panel (it has
  negative values).
- Slides, Method Moments for Gaussian frame: `\itseptbig` typo (ignored).
- `ch4_gaussian_moments_example`, `ch4_leastsquares_example` (original): plotted the
  *true* model while labelling it as the fitted one; now plot the estimated model.
- `ch4_leastsquares_example`: parameters reordered as theta = (xi, beta) to match
  the slides.

## Chapter 5
- Slides, Multidimensional Linear Models: theta_hat = theta + (X^T X)^{-1} **X** e
  should read (X^T X)^{-1} **X^T** e.
- Slides, Bayesian Estimation: log f(theta) and log f(y|theta) are written with "+"
  in front of the quadratic terms; both should be negative (the min problem that
  follows is correct).
- Slides, Example: Hougen-Watson: SSE(theta_hat) = 0.2989 equals sum of squared
  errors *without* the 1/2 used in the SSE definition (with 1/2: 0.1495).
- Slides, Example: Hougen-Watson: leftover "\tt Matlab function fitnlm" text.
- `ch5_hougen_watson_regularized` (original): the ridge penalty was added once per
  data point (13 times) and centred at the initial guess; now added once with
  kappa = 1 and theta_bar = 0, as on the slide.
- `ch5_hougen_watson` (original): the 1D sweeps were labelled theta_2 / theta_1
  but varied theta_1 / theta_0. Replaced by the slide's 2D sweep over (theta_1, theta_2).
- `ch5_heat_capacity_bayes`, `ch5_heat_capacity_mc` (original): generated data with
  theta_1 = 0.02923, ten times the value used elsewhere (2.923e-3). Fixed.
- `ch5_heat_capacity_mc`: the slides use 1/(K-1) in the MC covariance; the code used 1/K.
- `ch5_linear_est_example` (original) wrote `ch5_lin_est.eps` to disk; removed.
- New notebook `ch5_data_reconciliation` (illustrative numbers; the slide has none).
- Notebooks not cited by the slides are placed as "additional examples":
  wrong-model residuals (Residual Analysis), SSE surfaces and cross-validation
  (Data Sufficiency), MC covariance (Nonlinear Models), lasso trade-off for the
  Gibbs reactor (Prior Knowledge), recursive Bayes (Bayesian Estimation).

## Chapter 6, Part I
- Slides, Principal Component Estimation: the partition labels W_1 as "low
  information" and W_2 as "high information"; W_1 holds the largest eigenvalues
  (high information) and is what is kept.
- Slides, Sparse PCA: for eigenvectors with zero eigenvalue, X w_j = 0, so the
  stated problem min ||X w_j - X w||^2 + kappa ||w||_1 returns w = 0. The notebook
  sparsifies the eigenvectors directly (soft thresholding); it reaches the slide's
  conclusion (drop CO, CH3OH, and T; T is constant, so it duplicates the intercept).
- `ch6_gibbs_pca_estimation_sparse` (original) kept CO, CH3OH and the intercept,
  the opposite of the slide's conclusion. Now keeps intercept, P, H2.
- `ch6_svd_image_field`: now uses the 10x10 matrix printed on the slide (the
  original generated a different matrix); singular values match the slide exactly.
- `ch6_svd_reconstruction` (original) overwrote `Data/droplet1g.jpg`; removed.
  Uses n = 1, 3, 10, 100 as on the slide. The image in `Data/` is 199x199, not 605x605.

## Chapter 6, Part II
- Slides, 1D convolution: the discrete Gaussian filter is written g_j = e^{-j^2/sigma^2};
  the quoted values (0.13, 0.6, 1, 0.6, 0.13) correspond to e^{-j^2/(2 sigma^2)}.
  The second noisy-signal filter is written (0.0003, 0.41, 0.41, 0.003); last entry
  should be 0.0003.
- `ch6_image_filters_2D` (original): in the convolution-theorem part the Gaussian
  kernel was built by blurring a zero array and *then* setting the centre to 1, so
  it was a delta and the "filtered" images equalled the original. Fixed and checked
  against direct convolution.
- `ch6_fourier_transform_2D` (original): sampled sin(2 pi 60 x) on a 0.1 grid, so the
  high-frequency wave aliased to zero; also used omega_1 = 10 and "+" instead of
  the slide's omega_1 = 20 and "-". Fixed.
- `ch6_derivative_filter`: now divides by 2 dt so f*g matches the pdf (as the slide says).
- New notebook `ch6_matrix_convolution` for the 5x5 example.

## Chapter 6, Part III
- Slides, Kriging: sigma^2(x) = c(x) - k(x)^T C(gamma) k(x) should use C(gamma)^{-1}.
- Slides, Fully Connected Neural Nets: `\kapp` typo in the loss (rendered as \kappa).
- Slides, CNN example: caption "Figure \ref{fig:caseIdata}" (rendered as "above").
- Slides cite `ch6_cnn_endotoxins_caseI/II.m`, but the Python folder only had
  `ch6_cnn_sds_caseI/II` (a different data set: LC droplets with SDS). New PyTorch
  notebooks `ch6_cnn_endotoxins_caseI/II` follow the Matlab scripts on
  `Data/Endotoxins_Case*/Gray` (case I: 100% train/validation accuracy, as on the
  slide; case II: ~93% train / ~74% validation). The SDS notebooks are kept as
  additional examples.
- `ch6_hougen_watson_kernel` (original): used exp(-gamma ||x - x'||) (not squared, as
  in the slide's RBF kernel), and the "gamma = 0.0001" section reran gamma = 0.001.
  Fixed.
- `ch6_gibbs_logistic` (original): plotted P(y = 0 | x) labelled as P(y = 1 | x); used
  scikit-learn's default L2 penalty (not plain MLE). Fixed. Accuracy with the data in
  `Data/` is 87.8% (439 of 500); the slide quotes 85% (214 of 250).
- `ch6_gibbs_kriging`: added the noise term sigma^2 I of the slide (WhiteKernel); without
  it the GP interpolated the noisy data exactly (MSE ~ 1e-22).
- `ch6_gibbs_neuralnet`: now one hidden layer with two perceptrons, as on the slide
  (the original used four).
- New notebook `ch6_logistic_function` (the slides cite `ch6_logistic_function.m`).

## Chapter 7
- `ch7_npv_comparison` (original): Option II's NPV was computed from Option I's rates
  of return (`enumerate(ROI1)`). Fixed; the risk measures now match the slide
  (e.g. P(loss) 0% vs 8.7%, P(gain) 6.7% vs 53.8%; slide: 0% vs 9%, 6.4% vs 55.9%).
- Slides, Measuring Risk for Investment Optimization: "V[Y(u')] = $79,288" is the
  standard deviation (SD), not the variance.
- Generator sizing: the slide states U = {0, 1, ..., 10}; the original code used
  {0, 2, ..., 10}. With the Python samples the decisions are u_D* = 7 and u* = 5
  (slide: 6 and 4); the peak load, and hence u_D*, depends on the random sample.
  The slide's average costs (67.4 and 50.1) are on a different scale than the code's
  cost function (844.6 and 501.8 with alpha_R = 5, alpha_u = 15).
- `ch7_generator_twostage`: the recourse is computed in closed form,
  W(X) = min(u, X1 + X2), instead of a nonsmooth optimization over 200 variables;
  results match the slide (u* = 1, u_D* = 10).
- New notebook `ch7_risk_measures_norms` for the CVaR-as-norm example.

## Python folder
- On 2026-10-17 the notebooks in `Content/Python` were replaced by the book's
  versions (same file names, plus 9 new notebooks). The previous versions are in
  `ARCHIVE/Python_10_17_2026`. `tools/build_notebooks.py` now copies every rebuilt
  notebook to `Content/Python`, so the two stay identical.
