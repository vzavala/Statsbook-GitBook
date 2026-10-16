"""Book layout: chapters, how slide frames are split into pages, and where
notebooks that the slides do not cite by name are placed."""

# Where students open notebooks. Set this to the GitHub repository that GitBook
# syncs with (owner/name and branch); used for "Open in Colab" links.
GITHUB_REPO = "vzavala/StatsBook-GitBook"
GITHUB_BRANCH = "main"

SLIDES_DIR = "../Slides/Slides-New"

# Each page starts at the first frame (at or after the previous page start)
# whose title begins with the given string.
CHAPTERS = [
    dict(key="ch1", tex="stats_ch1_intro.tex", title="Chapter 1: Introduction", pages=[
        ("Motivation", "Motivation"),
        ("Random Variables", "Random Variables"),
        ("Types of Random Variables", "Types of Random Variables"),
        ("From Data to Models", "From Data to Models"),
        ("From Knowledge to Decisions", "From Knowledge to Decisions"),
    ]),
    dict(key="ch2", tex="stats_ch2_univRV.tex", title="Chapter 2: Univariate Random Variables", pages=[
        ("Models of Univariate RVs", "Models of Univariate RVs"),
        ("Discrete Uniform", "Discrete Uniform"),
        ("Bernoulli", "Bernoulli"),
        ("Hypergeometric", "Hypergeometric"),
        ("Binomial", "Binomial"),
        ("Poisson", "Poisson"),
        ("Uniform (Continuous)", "Uniform (Continuous)"),
        ("Gaussian", "Gaussian"),
        ("Log-Normal", "Log-Normal"),
        ("Exponential", "Exponential"),
        ("Gamma", "Gamma"),
        ("Chi-Squared", "Chi-Squared"),
        ("Weibull", "Weibull"),
        ("Families of Random Variables", "Families of Random Variables"),
    ]),
    dict(key="ch3", tex="stats_ch3_multiRV.tex", title="Chapter 3: Multivariate Random Variables", pages=[
        ("Multivariate Random Variables", "Multivariate Random Variables"),
        ("Joint PDFs and CDFs", "Joint PDFs and CDFs"),
        ("Marginal PDFs", "Marginal PDFs"),
        ("Conditional PDFs and CDFs", "Conditional PDFs and CDFs"),
        ("Independence", "Independence"),
        ("Conditional Events", "Conditional Events"),
        ("Summarizing Statistics", "Summarizing Statistics"),
        ("Multivariate Gaussian", "Multivariate Gaussian Model"),
    ]),
    dict(key="ch4", tex="stats_ch4_estimationRV.tex", title="Chapter 4: Estimation for Random Variables", pages=[
        ("Estimation Techniques", "Estimation Techniques"),
        ("Method of Moments", "Method of Moments"),
        ("Least-Squares Method", "Least-Squares Method"),
        ("Maximum Likelihood Estimation", "Maximum Likelihood Estimation"),
        ("Data Collection and Asymptotic Properties", "Data Collection and Asymptotic Properties"),
        ("Monte Carlo Approximations", "Monte Carlo Approximations"),
        ("Central Limit Theorem", "Central Limit Theorem"),
        ("Extreme Value Theorem", "Extreme Value Theorem"),
    ]),
    dict(key="ch5", tex="stats_ch5_estimationstruct.tex", title="Chapter 5: Estimation for Structural Models", pages=[
        ("Estimation Techniques for Structural Models", "Estimation Techniques for Structural Models"),
        ("Scalar Linear Models", "Scalar Linear Models"),
        ("Multidimensional Linear Models", "Multidimensional Linear Models"),
        ("Residual Analysis", "Residual Analysis"),
        ("Uncertainty Quantification", "Uncertainty Quantification"),
        ("Data Sufficiency and Overfitting", "Data Sufficiency and Overfitting"),
        ("Estimation for General Nonlinear Models", "Estimation for General Nonlinear Models"),
        ("Prior Knowledge", "Prior Knowledge"),
        ("Bayesian Estimation", "Bayesian Estimation"),
    ]),
    dict(key="ch6a", tex="stats_ch6_partI.tex", group="Chapter 6: Statistical Data Analysis and Learning",
         title="Part I: Dimensionality Reduction", pages=[
             ("Statistical Data Analysis and Learning", "Statistical Data Analysis and Learning"),
             ("Principal Component Analysis", "Principal Component Analysis (PCA)"),
             ("Principal Component Estimation", "Principal Component Estimation"),
             ("Sparse PCA", "Sparse PCA"),
             ("Singular Value Decomposition", "Singular Value Decomposition (SVD)"),
         ]),
    dict(key="ch6b", tex="stats_ch6_partII.tex", group="Chapter 6: Statistical Data Analysis and Learning",
         title="Part II: Signal Processing", pages=[
             ("Signal Processing", "Signal Processing"),
             ("1D Convolution Operations", "1D Convolution Operations"),
             ("1D Fourier Transforms", "1D Fourier Transforms"),
             ("Properties of Convolution", "Properties of Convolution"),
             ("2D Convolutions", "Convolutions and Fourier Transforms in 2D"),
             ("2D Fourier Transforms", "2D Fourier Transforms"),
         ]),
    dict(key="ch6c", tex="stats_ch6_partIII.tex", group="Chapter 6: Statistical Data Analysis and Learning",
         title="Part III: Machine Learning Models", pages=[
             ("Classification Models", "Classification Models"),
             ("Kernel Models", "Kernel Models"),
             ("Kriging", "Kriging"),
             ("Neural Nets", "Neural Nets"),
             ("Convolutional Neural Nets", "Convolutional Neural Nets"),
         ]),
    dict(key="ch7", tex="stats_ch7_decisionmaking.tex", title="Chapter 7: Decision-Making Under Uncertainty", pages=[
        ("Motivation", "Motivation"),
        ("Deterministic Decision-Making", "Deterministic Decision-Making"),
        ("Attitudes Towards Risk", "Attitudes Towards Risk"),
        ("Risk Measures", "Risk Measures"),
        ("Stochastic Dominance", "Stochastic Dominance"),
        ("Coherent Risk Measures", "Coherent Risk Measures"),
        ("Stochastic Optimization", "Stochastic Optimization"),
        ("Conflict Resolution", "Conflict Resolution"),
        ("Formulations with Recourse", "Stochastic Optimization: Formulations with Recourse"),
        ("Shaping Formulations and Flexibility", "Stochastic Optimization: Shaping"),
        ("Bayesian Optimization", "Bayesian Optimization"),
    ]),
]

# Slide script names that differ from the notebook names.
ALIASES = {}

# Notebooks the slides do not cite: notebook -> (chapter key, frame title prefix,
# occurrence of that title, 1-based). The notebook is placed after that frame.
EXTRA = {
    "ch1_mean_var_descriptors": ("ch1", "Summarizing Statistics (Basic)", 1),
    "ch2_compare_binomial": ("ch2", "Binomial", -1),
    "ch2_example_alarm": ("ch2", "Example: Setting up an Alarm", -1),
    "ch2_example_exponential": ("ch2", "Example: Residence Time", -1),
    "ch2_example_service_time": ("ch2", "Example: Service Time", -1),
    "ch2_compare_chisq": ("ch2", "Chi-Squared", -1),
    "ch2_example_chisquare": ("ch2", "Example: Computing Confidence Intervals", -1),
    "ch3_examples_multidatasets": ("ch3", "Multivariate Random Variables", 1),
    "ch3_gibbs_subdomains": ("ch3", "Domains and Subdomains", -1),
    "ch3_gibbs_conditional": ("ch3", "Empirical Event Probabilities", -1),
    "ch3_thermostat_lifetime": ("ch3", "Example: Lifetime of Thermostat", 1),
    "ch4_weibull_moments_example": ("ch4", "Example: Method of Moments for Weibull", -1),
    "ch5_linear_est_example_wrong": ("ch5", "Residual Analysis", -1),
    "ch5_examples_surfaces": ("ch5", "Data Sufficiency and Overfitting", -1),
    "ch5_heat_capacity_benchmark": ("ch5", "Data Sufficiency and Overfitting", -1),
    "ch5_heat_capacity_mc": ("ch5", "Estimation for General Nonlinear Models", -1),
    "ch5_gibbs_tradeoff": ("ch5", "Prior Knowledge", -1),
    "ch5_heat_capacity_bayes": ("ch5", "Example: Bayesian Estimation", -1),
    "ch5_data_reconciliation": ("ch5", "Example: Data Reconciliation", -1),
    "ch6_convolution_filters": ("ch6b", "Example: FT of Pulse and Gaussian", -1),
    "ch6_sysid_co2_est": ("ch6b", "Example: FT of Pulse and Gaussian", -1),
    "ch6_matrix_convolution": ("ch6b", "Example: Convolution of Matrix with Filter", -1),
    "ch6_cnn_sds_caseI": ("ch6c", "Example: Classification of Flow Cytometry", -1),
    "ch6_cnn_sds_caseII": ("ch6c", "Example: Classification of Flow Cytometry", -1),
    "ch7_risk_measures_norms": ("ch7", "Example: Risk Measures as Norms", -1),
    "ch7_example_SAA": ("ch7", "Example: Deterministic vs. Stochastic", -1),
}

# Notebooks shown collapsed (they only reproduce a slide figure).
COLLAPSED_PREFIXES = ("ch2_compare_", "ch3_compare_")
