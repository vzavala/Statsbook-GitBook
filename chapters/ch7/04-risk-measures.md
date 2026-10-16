# Risk Measures

* Risk measure is a scalar function $$\rho(Y(u))$$ that maps RV $$Y(u)$$ to a number.
* Risk measure is a *summarizing statistic*.
* Risk measure aims to extract key aspects/features of RV.
* We have mentioned some common risk measures that capture risk attitudes.
* We now present more sophisticated measures and analyze them in detail.

* Mean-Variance:

  $$
  \mathbb{MV}_\kappa[Y(u)]=\mathbb{E}[Y(u)]+\kappa\cdot \mathbb{V}[Y(u)]
  $$

   for some weight $$\kappa \in \mathbb{R}_+$$; for $$\kappa\to 0$$ we prefer mean, for $$\kappa\gg 0$$ we prefer variance.

* Probability of Loss:

  $$
  \mathbb{P}(Y(u)>\bar{y})
  $$

   for some threshold $$\bar{y}\in \mathbb{R}$$. This is also often expressed as $$\mathbb{P}(Y(u)\leq \bar{y})$$.

* Event probability:

  $$
  \mathbb{P}(Y(u)\in \mathcal{A})
  $$

   where the event $$\mathcal{A}$$ capture conditions/constraints that $$Y(u)$$ should satisfy.

* Value-at-Risk:

  $$
  \mathbb{Q}_{Y(u)}(\alpha)
  $$

   for probability $$\alpha\in [0,1]$$. This is the $$\alpha$$-quantile of $$Y(u)$$.

* Conditional Value-at-Risk (CVaR):

  $$
  \mathbb{E}[Y(u)|Y(u)\geq \mathbb{Q}_{Y(u)}(\alpha)]
  $$

   for some probability $$\alpha\in [0,1]$$. This is also known as “expected loss”.

* Mean Absolute Deviation:

  $$
  \mathbb{E}\left[|Y(u)-\mathbb{E}[Y(u)]|\right]
  $$

* Worst-Case and Best-Case Value:

  $$
  \max Y(u)\quad \textrm{and} \quad  \min Y(u)
  $$

* Probability of loss resembles our colloquial definition of risk. But this fails to capture important aspects (magnitude of losses).
* There is a broad selection of risk measures that one can select.
* In fact, any statistic can be used as risk measure (e.g., mode, moments,  median.
* It is also important to highlight that the selection of a risk measure should be based on  nature of RV to be compared.
* e.g., comparing Gaussian RVs (symmetric RVs with left and right tails) is not the same as comparing exponential RVs (asymmetric RVs with only right tails).

## Example: Measuring Risk for Investment Optimization

We compute risk measures for investment example.

* Expected Value:

   $$\mathbb{E}[Y(u)]=-\$70,891$$ and $$\mathbb{E}[Y(u')]=-\$109,048$$

* Standard Deviation:

   $$\mathbb{SD}[Y(u)]=\$19,284$$ and $$\mathbb{V}[Y(u')]=\$79,288$$

* Probability of Loss:

   $$\mathbb{P}(Y(u)>0)=0\%$$ and $$\mathbb{P}(Y(u')>0)=9\%$$

* Probability of Gain:

   $$\mathbb{P}(Y(u)\leq \bar{y})=6.4\%$$ and $$\mathbb{P}(Y(u')\leq\bar{y})=55.90\%$$ for $$\bar{y}=-\$100,000$$.

* Value-at-Risk:

   $$\mathbb{Q}_{Y(u)}(\alpha)=-\$46,699$$ and $$\mathbb{Q}_{Y(u')}(\alpha)=-\$4,935$$ for $$\alpha=0.9$$.

* Conditional Value-at-Risk:

   $$\textrm{CVaR}_\alpha[Y(u)]=-\$37,530$$ and $$\textrm{CVaR}_\alpha[Y(u')]=\$34,810$$  for $$\alpha=0.9$$

* Worst-Case:

   $$\max Y(u)=-\$9,129$$ and $$\max Y(u')=\$181,413$$.

* Best-Case:

   $$\min Y(u)=-\$140,609$$ and $$\min Y(u')=-\$363,708$$.

#### Python implementation: Risk measures for the two options

$$\mathbb{E}$$, $$\mathbb{SD}$$, probability of loss $$\mathbb{P}(Y>0)$$, probability of gain $$\mathbb{P}(Y\leq\bar{y})$$
with $$\bar{y}=-100$$ kUSD, $$\mathbb{Q}_Y(0.9)$$, $$\mathrm{CVaR}_{0.9}$$, worst case $$\max Y$$ and best case $$\min Y$$.

```python
def risk_measures(Y, alpha=0.9, y_bar=-100):
    Q = np.quantile(Y, alpha)
    return {'E': Y.mean(), 'SD': Y.std(ddof=1), 'P(loss)': np.mean(Y > 0), 'P(gain)': np.mean(Y <= y_bar),
            'VaR': Q, 'CVaR': Y[Y >= Q].mean(), 'max': Y.max(), 'min': Y.min()}

rm_I, rm_II = risk_measures(NPV_I), risk_measures(NPV_II)
print(f"{'measure':10s}{'Option I':>12s}{'Option II':>12s}")
for k in rm_I:
    fmt = '{:12.1%}' if k.startswith('P') else '{:12.1f}'
    print(f"{k:10s}" + fmt.format(rm_I[k]) + fmt.format(rm_II[k]))
```

```
measure       Option I   Option II
E                -70.6      -105.5
SD                19.1        74.8
P(loss)           0.0%        8.7%
P(gain)           6.7%       53.8%
VaR              -46.4        -8.5
CVaR             -37.4        24.3
max              -12.7       126.8
min             -124.8      -349.3
```
