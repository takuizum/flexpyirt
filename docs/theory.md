# Theory

English | [日本語](theory.ja.md)

The ideas behind flexpyirt, in brief: what the models are, what is maximized, how the EM algorithm
does it, and where standard errors, scores, fit statistics and linking constants come from. Each
section ends with a pointer to the [user guide](guide.md) section that shows the code. For full
treatments see the references at the end (e.g. Baker & Kim, 2004; Chalmers, 2012).

1. [Item response models](#1-item-response-models)
2. [Marginal likelihood](#2-marginal-likelihood)
3. [Identification](#3-identification)
4. [The Bock–Aitkin EM algorithm](#4-the-bockaitkin-em-algorithm)
5. [Latent distributions](#5-latent-distributions)
6. [Multiple groups and FIPC](#6-multiple-groups-and-fipc)
7. [Priors and MAP estimation](#7-priors-and-map-estimation)
8. [Missing responses](#8-missing-responses)
9. [Standard errors](#9-standard-errors)
10. [Scoring respondents](#10-scoring-respondents)
11. [Model fit and DIF](#11-model-fit-and-dif)
12. [Scale linking](#12-scale-linking)

**Notation.** Respondents $i=1,\dots,N$, items $j=1,\dots,J$, response $y_{ij}\in\{0,\dots,K_j-1\}$,
latent trait $\theta$ (a vector $\boldsymbol\theta$ in multidimensional models), groups
$g=1,\dots,G$. $\sigma(z)=1/(1+e^{-z})$ is the logistic function. $\xi_j$ are the parameters of
item $j$ and $\eta$ the vector of all free parameters.

## 1. Item response models

An item response model gives the probability of each response category as a function of
$\theta$. flexpyirt uses the **slope–intercept** form (as mirt does): the linear predictor is
$z=a\theta+d$ with slope $a$ and intercept $d$.

| Model | Category probabilities | Notes |
|---|---|---|
| 2PL (Birnbaum, 1968) | $P(y=1\mid\theta)=\sigma(a\theta+d)$ | |
| Rasch (Rasch, 1960) | $P(y=1\mid\theta)=\sigma(\theta+d)$ | $a=1$; the latent variance carries the scale |
| 3PL / 4PL | $P(y=1\mid\theta)=g+(u-g)\,\sigma(a\theta+d)$ | lower asymptote $g$ ("guessing"), upper asymptote $u$ ($u=1$ in 3PL) |
| GRM (Samejima, 1969) | $P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$, $P(y=k)=P(y\ge k)-P(y\ge k+1)$ | ordered intercepts $d_1>\dots>d_{K-1}$ |
| GPCM (Muraki, 1992) | $P(y=k\mid\theta)\propto\exp(k\,a\theta+d_k)$, $d_0=0$ | adjacent-category logits $\log\frac{P(y=k)}{P(y=k-1)}=a\theta+d_k-d_{k-1}$ |
| PCM (Masters, 1982) | GPCM with $a=1$ | |
| Nominal (Bock, 1972) | $P(y=k\mid\theta)\propto\exp(a\,\mathrm{ak}_k\,\theta+d_k)$ | unordered categories; scoring coefficients $\mathrm{ak}_0=0$, $\mathrm{ak}_{K-1}=K-1$, $d_0=0$ (Thissen, Cai & Bock, 2010) |

**IRT (difficulty) parameterization.** Writing $a\theta+d=a(\theta-b)$ gives the familiar
difficulty $b=-d/a$: the $\theta$ at which $P(y=1)=1/2$ in the 2PL. Likewise $b_k=-d_k/a$ is the
GRM category boundary, and $b_k=-(d_k-d_{k-1})/a$ is the GPCM/PCM step difficulty (where adjacent
categories are equally likely). flexpyirt estimates $(a,d)$, which behaves better numerically
(it is defined at $a=0$), and reports $b$ for display with delta-method standard errors
([section 9](#9-standard-errors)).

**The constant $D=1.7$.** Some programs write $\sigma(1.7\,a(\theta-b))$ so that the logistic curve
approximates the normal ogive. flexpyirt (like mirt) uses no such constant; slopes from a
"$D=1.7$" program are $1.7$ times smaller than flexpyirt's.

**Multidimensional models.** With $D$ factors, $z=\mathbf a^\top\boldsymbol\theta+d$ with a slope
per factor ($a_1,\dots,a_D$), and $\boldsymbol\theta$ follows a multivariate normal distribution.
In a *confirmatory* model each item loads only on the factors you declare; the other slopes are
fixed at 0.

**Information.** The Fisher information of an item tells how precisely it measures at each
$\theta$; for the 2PL it is $I_j(\theta)=a_j^2P_j(\theta)\{1-P_j(\theta)\}$, largest at
$\theta=b_j$. The test information $I(\theta)=\sum_jI_j(\theta)$ appears in the WLE
([section 10](#10-scoring-respondents)).

Guide: [section 3](guide.md#3-specifying-a-model).

## 2. Marginal likelihood

**Local independence.** Given $\theta$, responses to different items are independent, so the
likelihood of respondent $i$'s response pattern $\mathbf y_i$ is
$$L(\mathbf y_i\mid\theta)=\prod_j P_j(y_{ij}\mid\theta;\xi_j).$$

**Marginal maximum likelihood (MML).** The person parameters $\theta_i$ are not estimated
together with the items (that would give inconsistent item estimates as $N$ grows). Instead
$\theta$ is treated as a random draw from a population distribution $f(\theta)$ and integrated out
(Bock & Lieberman, 1970; Bock & Aitkin, 1981):
$$\ell(\eta)=\sum_{i=1}^N\log\int L(\mathbf y_i\mid\theta)\,f(\theta)\,d\theta .$$
The item parameters and the parameters of $f$ (its mean and variance, or its shape) are estimated
by maximizing $\ell$.

**Quadrature.** The integral is replaced by a sum over nodes $\theta_1,\dots,\theta_Q$ with weights
$w_q$:
$$\ell(\eta)\approx\sum_i\log\sum_{q=1}^Q L(\mathbf y_i\mid\theta_q)\,w_q .$$
The default grid is 61 equally spaced nodes on $[-6,6]$, with $w_q$ proportional to the density
$f(\theta_q)$ and normalized to sum to 1 on the grid. **This discretized likelihood is the
quantity flexpyirt maximizes exactly** and reports as `loglik`. Respondents with the same
response pattern contribute identical terms, so patterns are counted once with their frequency.
Gauss–Hermite quadrature is an alternative for normal $f$. Multidimensional models use the
Cartesian product of a one-dimensional grid.

Because the weights are normalized on a finite grid, a normal distribution whose mean is far from
0 or whose variance is large is cut off at the grid ends. The reported `var` is the parameter of
the normal density before the cut; the distribution on the grid has a slightly smaller variance.
Widening the grid makes the cut-off negligible (guide
[section 13](guide.md#13-choosing-the-quadrature-grid)).

## 3. Identification

The location and unit of $\theta$ are arbitrary. For any $m$ and $s>0$, replacing $\theta$ by
$\theta^*=s\theta+m$ and the item parameters by $a^*=a/s$, $d^*=d-am/s$ leaves every response
probability unchanged, because $a^*\theta^*+d^*=a\theta+d$. Something has to fix the scale:

- **Default**: the latent distribution is $N(0,1)$ (fixed mean and variance).
- **Rasch / PCM**: slopes are fixed at 1, which fixes the unit, so the latent **variance is
  estimated** and only the mean is fixed at 0.
- **Fixed items (FIPC)**: items fixed at known values carry the scale of the item bank, so the
  latent mean and variance can be estimated.
- **Multiple groups**: the reference group is fixed; a focal group's mean and variance are
  identified through anchor items whose parameters are equal across groups.

The general rule: a free latent variance needs at least one fixed nonzero slope (in the same
connected set of groups), and a free latent mean needs at least one fixed intercept. flexpyirt
checks this before fitting and raises `IdentificationError` otherwise. In confirmatory
multidimensional models, the zero loadings also remove the rotational indeterminacy of
$\boldsymbol\theta$.

Guide: [section 8](guide.md#8-latent-distributions-and-identification).

## 4. The Bock–Aitkin EM algorithm

Maximizing $\ell(\eta)$ directly is awkward because of the log of a sum. The EM algorithm
(Dempster, Laird & Rubin, 1977) applied to IRT by Bock & Aitkin (1981) treats the node at which
each respondent "sits" as missing data and alternates two steps.

**E-step.** With the current parameters $\eta'$, compute each pattern's posterior weights over the
nodes,
$$h_{iq}=\frac{L(\mathbf y_i\mid\theta_q)\,w_q}{\sum_{q'}L(\mathbf y_i\mid\theta_{q'})\,w_{q'}},$$
and from them the **expected counts** ("artificial data"): the expected number of respondents at
node $q$ and the expected number of them choosing category $k$ of item $j$,
$$\bar n_q=\sum_i h_{iq},\qquad \bar r_{jkq}=\sum_i h_{iq}\,\mathbf 1[y_{ij}=k].$$

**M-step.** Maximize the expected complete-data log-likelihood
$$Q(\eta\mid\eta')=\sum_j\sum_q\sum_k\bar r_{jkq}\log P_{jk}(\theta_q;\xi_j)+\sum_q\bar n_q\log w_q .$$
It separates into one small problem per item (the first term) and one for the latent
distribution (the second term). flexpyirt solves each item's problem with a few Newton steps,
using gradients and Hessians from automatic differentiation, and a line search that never lets
$Q$ decrease.

**Monotonicity.** Since $\ell(\eta)=Q(\eta\mid\eta')-H(\eta\mid\eta')$ with $H$ maximized at
$\eta=\eta'$, any step that increases $Q$ also increases $\ell$ (a *generalized* EM). The
log-likelihood therefore never decreases from one iteration to the next; `history["loglik"]`
lets you check this.

**The latent distribution's M-step.** With a normal $f$ the weights are
$w_q(\mu,\sigma^2)\propto\phi(\theta_q;\mu,\sigma^2)$ normalized on the grid. A common shortcut is
to set $\mu$ and $\sigma^2$ to the mean and variance of the expected counts $\bar n_q$; that
maximizes the *continuous* normal likelihood, not the discretized one, and so is not an exact EM
step. flexpyirt maximizes $\sum_q\bar n_q\log w_q(\mu,\sigma^2)$ exactly. The two agree when the
grid comfortably covers the distribution, and differ slightly when the grid cuts it off (this is
the source of small differences from mirt; see [Accuracy and validation](accuracy.md)).

**Convergence.** EM stops when the largest absolute change in any free parameter between two
iterations falls below `tol` (default $10^{-4}$, the same rule as mirt's `TOL`).

**SQUAREM.** EM converges linearly and can be slow when the latent distribution is weakly
determined (free variance, several groups, histograms). SQUAREM (Varadhan & Roland, 2008)
extrapolates along the EM path: with $F$ the EM map, $\mathbf r=F(\eta)-\eta$ and
$\mathbf v=F(F(\eta))-2F(\eta)+\eta$, it moves to $\eta-2\alpha\mathbf r+\alpha^2\mathbf v$ with
$\alpha=-\lVert\mathbf r\rVert/\lVert\mathbf v\rVert$, then applies one stabilizing EM step. If the
extrapolated point is infeasible or lowers the log-likelihood, flexpyirt falls back to the plain
EM point, so monotonicity is kept and the fixed point is the same.

Guide: [section 4](guide.md#4-fitting).

## 5. Latent distributions

The latent distribution $f(\theta)$ is a model for the population, not only a device for
integration. If the true distribution is skewed or bimodal, assuming normality can bias the item
parameters (Woods, 2007). flexpyirt offers four forms.

- **Normal** $N(\mu,\sigma^2)$: two parameters.
- **Empirical histogram (EH)** (Bock & Aitkin, 1981; Woods, 2007): the weights $w_1,\dots,w_Q$ are
  themselves free parameters. The M-step maximizes $\sum_q\bar n_q\log w_q$ over the probability
  simplex, giving $w_q=\bar n_q/N$ (with a tiny floor). Because a histogram has a free location and
  scale, flexpyirt constrains the reference group's histogram to mean 0 and variance 1 *within* the
  M-step (an exact constrained maximization, which keeps EM monotone) unless fixed items set the
  scale. EH is flexible but has $Q-1$ parameters, so it needs large samples.
- **Davidian curves** (Zhang & Davidian, 2001; Woods & Lin, 2009): a smooth
  seminonparametric density $h_k(z)=P_k(z)^2\varphi(z)$, a squared polynomial of degree $k$
  times the standard normal density. The polynomial coefficients are constrained to the unit
  sphere (so $h_k$ integrates to 1) and parameterized by $k$ angles; all angles 0 is the normal
  distribution. flexpyirt standardizes the curve analytically so that its mean and variance are
  exactly $\mu$ and $\sigma^2$ for any shape, which separates shape from location and scale. The
  degree $k$ (1–10) is usually chosen by AIC or BIC.
- **Normal mixtures** (Mislevy, 1984): $f(\theta)=\sum_c\pi_c\,N(\mu_c,\sigma_c^2)$ with $k$
  components and one common set of item parameters, suited to populations made of
  subpopulations. The mixture is standardized to mean $\mu$ and variance $\sigma^2$ in the same
  way. Components are exchangeable (their numbering is arbitrary) and the likelihood can have
  several local maxima, so refitting from several starts is advisable.

Near the normal distribution the shape parameters of Davidian curves and mixtures are poorly
determined, and likelihood-ratio tests against the normal model do not follow their usual
$\chi^2$ distribution; compare models by AIC / BIC instead.

Guide: [section 8](guide.md#8-latent-distributions-and-identification).

## 6. Multiple groups and FIPC

**Multiple-group calibration** (Bock & Zimowski, 1997) fits several populations at once, each with
its own latent distribution $f_g$. The marginal log-likelihood is the sum over groups,
$\ell=\sum_g\sum_{i\in g}\log\sum_q L(\mathbf y_i\mid\theta_q)w_{gq}$. Item parameters may differ by
group, except for **anchor items** constrained equal across groups; the anchors put all groups on
the reference group's scale, so the focal groups' means and variances become estimable.
Equality constraints are implemented by letting several parameters share one free parameter, so
the item-wise M-step is kept.

**Differential item functioning (DIF).** An item shows DIF when respondents with the same $\theta$
but from different groups have different response probabilities. With anchors linking the scales,
DIF is tested by comparing a model in which the item's parameters are free across groups with one
in which they are equal ([section 11](#11-model-fit-and-dif)).

**Fixed item parameter calibration (FIPC)** (Kim, 2006) puts new items on an existing item bank's
scale: the bank items are fixed at their known values and the new items and the latent mean and
variance are estimated in one EM run. Keeping the latent distribution at $N(0,1)$ would contradict
the fixed items whenever the new sample differs from the bank's calibration sample, so the latent
distribution is updated at every iteration (Kim's "multiple weights updating, multiple EM cycles"
method).

Guide: [section 9](guide.md#9-multiple-groups), [section 10](guide.md#10-fixed-item-parameter-calibration-fipc).

## 7. Priors and MAP estimation

Small samples, or items whose responses barely constrain a parameter (e.g. the 3PL lower
asymptote), can give unstable or boundary estimates. Bayes modal estimation (Mislevy, 1986)
adds log prior densities to the objective and maximizes the log posterior
$$\ell(\eta)+\log\pi(\eta).$$
In EM, the prior simply enters the M-step: each item maximizes $Q_j(\xi_j)+\log\pi(\xi_j)$, and
EM is monotone in the log posterior. Priors are defined on the natural parameter scale (e.g. a
Beta prior on $g$ itself, a log-normal prior on $a$), without a Jacobian term, as in mirt. The
result is a posterior mode, so the log-likelihood at that point is not the maximum; AIC and BIC
then lose their usual meaning, and flexpyirt warns if you request them.

Guide: [section 11](guide.md#11-priors-and-map-estimation).

## 8. Missing responses

A missing response simply drops out of the likelihood: the product
$L(\mathbf y_i\mid\theta)=\prod_jP_j(y_{ij}\mid\theta)$ runs over the observed items only. This is valid when responses are **missing at random**
(Rubin, 1976): whether a response is missing may depend on observed quantities (including the
group or which test form was administered), but not on the unobserved response itself given those.
Planned missingness from incomplete designs (different forms with common items) satisfies this.
Omitted or not-reached responses that depend on the respondent's ability may not, and are better
handled by scoring them explicitly or with a model for the missingness.

## 9. Standard errors

Asymptotic standard errors come from the inverse of the **observed information matrix**
$I(\hat\eta)=-\partial^2\ell/\partial\eta\,\partial\eta^\top$ at the estimate (plus
$-\partial^2\log\pi/\partial\eta\,\partial\eta^\top$ under MAP).

- **Hessian** (`se="hessian"`): the second derivatives of the marginal log-likelihood, computed
  directly by automatic differentiation.
- **Oakes' identity** (Oakes, 1999; `se="oakes"`, the default): the same matrix computed from the
  EM objective,
  $$I(\hat\eta)=-\left[\frac{\partial^2Q(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}+\frac{\partial^2Q(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\right]_{\eta=\eta'=\hat\eta}.$$
  The first term is the complete-data information, the second corrects for the information lost
  because $\theta$ is unobserved (Louis, 1982, gives an equivalent form). With automatic
  differentiation both terms are exact, so the two methods agree up to rounding.

Both are only meaningful at a converged interior solution. If the matrix is not positive definite
(an estimate on a boundary, weak identification), flexpyirt reports NaN.

**Cross-product and sandwich.** Let $s_i=\partial\log L_i/\partial\eta$ be the score of
respondent $i$ at the estimate and $B=\sum_i s_is_i^\top$ their cross-product. When the model is
correct, $B$ and $I$ estimate the same matrix (the information equality), so $B^{-1}$ is another
covariance estimate (`se="crossprod"`). When the model is not correct the two differ, the
estimator converges to the parameter that best approximates the truth, and its covariance is the
**sandwich** (Huber, 1967; White, 1982; `se="sandwich"`)
$$\operatorname{Var}(\hat\eta)\approx I(\hat\eta)^{-1}\,B\,I(\hat\eta)^{-1},$$
which reduces to $I^{-1}$ under the information equality. Under MAP the prior's curvature
$-\partial^2\log\pi/\partial\eta\,\partial\eta^\top$ is added to $B$ as it is to $I$, so the three
estimates still coincide for a correct model. The scores are exact (automatic differentiation of
each response pattern's marginal likelihood). Respondents are assumed to be independent.

**Delta method.** For a function $b=h(\eta)$ such as $b=-d/a$,
$\operatorname{Var}(\hat b)\approx\nabla h^\top\,\operatorname{Var}(\hat\eta)\,\nabla h$. flexpyirt uses
this for the displayed IRT thresholds and for the standardized mixture components.

Guide: [section 6](guide.md#6-standard-errors).

## 10. Scoring respondents

Once the items are calibrated, each respondent's $\theta$ is estimated with the item parameters
treated as known.

| Method | Estimate | Standard error |
|---|---|---|
| EAP (Bock & Mislevy, 1982) | posterior mean $\hat\theta=\sum_q\theta_q h_q$ on the quadrature grid | posterior SD |
| MAP | posterior mode: maximizes $\log L(\mathbf y\mid\theta)+\log\phi(\theta;\mu,\sigma^2)$ | $\{-\partial^2/\partial\theta^2\}^{-1/2}$ at the mode |
| ML | maximizes $\log L(\mathbf y\mid\theta)$ | $I(\hat\theta)^{-1/2}$ from the observed information |
| WLE (Warm, 1989) | solves $\partial\log L/\partial\theta+I'(\theta)/\{2I(\theta)\}=0$ | from the curvature of the weighted likelihood |
| PV (Mislevy, 1991) | random draws from each respondent's posterior | — |

- **Bayesian scores** (EAP, MAP) use the estimated population distribution as the prior. They have
  the smallest mean squared error for individuals but are shrunk toward the group mean, more so
  for short tests; their variance underestimates the population variance.
- **ML** uses no prior, so it is not shrunk, but it has a bias of order $1/J$ in short tests and is
  infinite for all-lowest or all-highest patterns. **WLE**
  (equivalently, maximizing $\log L+\tfrac12\log I(\theta)$; Firth, 1993) removes the leading
  $O(1/J)$ bias of ML and stays finite.
- **Plausible values** are for population-level analyses (means, regressions): analysing each set
  of draws and combining the results with Rubin's rules propagates measurement error correctly,
  which point estimates do not.

Guide: [section 7](guide.md#7-scoring).

## 11. Model fit and DIF

**Information criteria.** $\mathrm{AIC}=-2\ell+2k$ and $\mathrm{BIC}=-2\ell+k\log N$, with $k$ free
parameters, compare models fitted to the same data (smaller is better); BIC penalizes complexity
more.

**Likelihood-ratio tests.** For nested models,
$G^2=2(\ell_{\text{larger}}-\ell_{\text{smaller}})$ is approximately $\chi^2$ with degrees of freedom
equal to the difference in free parameters, provided the smaller model is not on the boundary of
the larger one (which fails, for example, when testing a latent shape against the normal
distribution). DIF tests use this, one item at a time.

**Item fit: S-X2** (Orlando & Thissen, 2000). Respondents are grouped by summed score $s$. For each
score and category, the observed proportion choosing category $k$ of item $j$ is compared with
the model-implied proportion
$$E_{jk}(s)=\frac{\sum_qP_{jk}(\theta_q)\,L_{-j}(s-k\mid\theta_q)\,w_q}{\sum_qL(s\mid\theta_q)\,w_q},$$
where $L(s\mid\theta)$ is the probability of summed score $s$ (from the Lord–Wingersky recursion)
and $L_{-j}$ the same without item $j$. Grouping by the observed summed score, instead of by
estimated $\theta$, makes the statistic approximately $\chi^2$ after sparse cells are collapsed.

**Limited-information fit: $M_2$** (Maydeu-Olivares & Joe, 2005, 2006). Full-information
goodness-of-fit tests over all $\prod_jK_j$ response patterns break down because most patterns are
never observed. $M_2$ uses only the first- and second-order margins (item means and pairwise
cross-products). With $\mathbf e_2$ the residuals of those margins, $\Xi_2$ their asymptotic
covariance and $\Delta_2$ their derivatives with respect to the parameters,
$$M_2=N\,\mathbf e_2^\top C_2\,\mathbf e_2,\qquad C_2=\Xi_2^{-1}-\Xi_2^{-1}\Delta_2(\Delta_2^\top\Xi_2^{-1}\Delta_2)^{-1}\Delta_2^\top\Xi_2^{-1},$$
which is approximately $\chi^2$ with (number of margins − number of parameters) degrees of freedom.
The RMSEA derived from it measures approximate fit on a sample-size-free scale.

Guide: [section 5](guide.md#5-reading-the-results), [section 9](guide.md#9-multiple-groups).

## 12. Scale linking

When two test forms are calibrated separately, each on its own $N(0,1)$ scale, their parameters
are on different scales that differ by a linear transformation $\theta^*=A\theta+B$ (section 3).
Common items let you estimate $A$ and $B$. Under the transformation, the source form's
slope–intercept parameters become $a^*=a/A$ and $d^*=d-aB/A$ (for GPCM, $d_k^*=d_k-k\,aB/A$).

**Characteristic-curve methods** choose $(A,B)$ so that the transformed source items reproduce
the target items' response curves over the target population:

- **Haebara** (1980): the squared differences of category response curves,
  $\sum_qw_q\sum_j\sum_k\{P^{\text{tgt}}_{jk}(\theta_q)-P^{\text{src}}_{jk}((\theta_q-B)/A)\}^2$
  (summing over all categories, as extended to polytomous items by Kim & Lee, 2006).
- **Stocking–Lord** (1983): the squared difference of the test characteristic curves (the
  expected summed score over the common items),
  $\sum_qw_q\{T^{\text{tgt}}(\theta_q)-T^{\text{src}}((\theta_q-B)/A)\}^2$.
- **Symmetric versions** (Kim & Lee, 2006) add the same criterion in the reverse direction, so
  linking B to A gives exactly the inverse transformation of linking A to B.

Characteristic-curve methods are generally more stable than moment methods (mean/mean,
mean/sigma), which flexpyirt uses only as starting values. Standard errors of $(A,B)$ follow from
the delta method applied to the condition that the criterion's gradient is zero (Ogasawara,
2001), propagating the sampling error of both calibrations. A linking is only as good as its
common items: check them for DIF across the two forms first.

Guide: [section 15](guide.md#15-scale-linking).

## References

- Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores*. Addison-Wesley.
- Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two or more nominal categories. *Psychometrika*, 37, 29–51.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6, 431–444.
- Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J. van der Linden & R. K. Hambleton (Eds.), *Handbook of Modern Item Response Theory*. Springer.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B*, 39, 1–38.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80, 27–38.
- Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research*, 22, 144–149.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 221–233). University of California Press.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Kim, S., & Lee, W.-C. (2006). An extension of four IRT linking methods for mixed-format tests. *Journal of Educational Measurement*, 43, 193–213.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings". *Applied Psychological Measurement*, 8, 453–461.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information estimation and goodness-of-fit testing in $2^n$ contingency tables. *Journal of the American Statistical Association*, 100, 1009–1020.
- Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit testing in multidimensional contingency tables. *Psychometrika*, 71, 713–732.
- Mislevy, R. J. (1984). Estimating latent distributions. *Psychometrika*, 49, 359–381.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56, 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Ogasawara, H. (2001). Standard errors of item response theory equating/linking by response function methods. *Applied Psychological Measurement*, 25, 53–67.
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement*, 24, 50–64.
- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*. Danish Institute for Educational Research.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63, 581–592.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement*, 7, 201–210.
- Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of Polytomous Item Response Theory Models* (pp. 43–75). Routledge.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scandinavian Journal of Statistics*, 35, 335–353.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54, 427–450.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Zhang, D., & Davidian, M. (2001). Linear mixed models with flexible distributions of random effects for longitudinal data. *Biometrics*, 57, 795–802.
