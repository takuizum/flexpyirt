# Theory

English | [日本語](theory.ja.md)

The ideas behind flexpyirt, in brief: what the models are, what is maximized, how the EM algorithm
does it, and where standard errors, scores, fit statistics, linking constants, summed-score tables and dimensionality checks
come from. Each
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
13. [Bifactor and two-tier models](#13-bifactor-and-two-tier-models)
14. [Sampling weights](#14-sampling-weights)
15. [Summed scores, information and reliability](#15-summed-scores-information-and-reliability)
16. [Local dependence and dimensionality](#16-local-dependence-and-dimensionality)

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
fixed at 0. Bifactor and two-tier models are confirmatory models with a structure that allows many
factors ([section 13](#13-bifactor-and-two-tier-models)).

**Information.** The Fisher information of an item tells how precisely it measures at each
$\theta$; for the 2PL it is $I_j(\theta)=a_j^2P_j(\theta)\{1-P_j(\theta)\}$, largest at
$\theta=b_j$. The test information $I(\theta)=\sum_jI_j(\theta)$ appears in the WLE
([section 10](#10-scoring-respondents)) and in reliability ([section 15](#15-summed-scores-information-and-reliability)).

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
each response pattern's marginal likelihood). Respondents are assumed to be independent; for
sampling weights see [section 14](#14-sampling-weights).

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

**DIF screening and effect sizes.** The Wald test is the large-sample equivalent of the
likelihood-ratio test computed from one fit: with $\mathbf L\boldsymbol\eta$ the differences of an
item's parameters between groups and $\mathbf V$ their covariance,
$W=(\mathbf L\boldsymbol\eta)^\top(\mathbf L\mathbf V\mathbf L^\top)^{-1}\mathbf L\boldsymbol\eta$
(Lord, 1980). Testing every item needs anchors, and anchors that themselves show DIF bias every
test, so screening purifies the anchor set: items flagged in one round are freed and the rest
retested until the flags stop changing (Candell & Drasgow, 1988; Kopf, Zeileis & Strobl, 2015).
How large a difference is follows from the expected item scores of the two groups,
$D(\theta)=E_f(\theta)-E_r(\theta)$, averaged over the focal latent distribution, signed (where
differences of opposite sign cancel) or unsigned (Chalmers, 2018), or standardized at the focal
respondents' scores (Meade, 2010).

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
slope–intercept parameters become $a^*=a/A$ and $d^*=d-aB/A$ (for GPCM, $d_k^*=d_k-k\,aB/A$; for
the nominal model, $d_k^*=d_k-\mathrm{ak}_k\,aB/A$ with the scoring coefficients $\mathrm{ak}_k$
unchanged). Asymptotes of the 3PL and 4PL do not change.

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

**Multidimensional models.** With $D$ factors the scales differ by
$\boldsymbol\theta^*=\mathbf A\boldsymbol\theta+\boldsymbol\beta$, under which
$\mathbf a^*=\mathbf A^{-\top}\mathbf a$ and $d^*=d-\mathbf a^{*\top}\boldsymbol\beta$ (Oshima, Davey &
Lee, 2000). The same two criteria are summed over the $D$-dimensional grid, with source curves
evaluated at $\mathbf A^{-1}(\boldsymbol\theta_q-\boldsymbol\beta)$. What $\mathbf A$ may be depends
on how the two calibrations were identified. Two confirmatory calibrations with the same
structure of zero slopes and unit factor variances differ only by a unit and an origin per
factor, so $\mathbf A$ is diagonal. If the factor axes themselves may differ, $\mathbf A$ is a
rotation times a dilation (Li & Lissitz, 2000) or a general nonsingular matrix (Oshima, Davey &
Lee, 2000), and the transformed items no longer have zero slopes where the source items did.

Guide: [section 15](guide.md#15-scale-linking).

## 13. Bifactor and two-tier models

**The models.** Tests are often built from item bundles (testlets): several questions about one
reading passage, or items sharing a scenario. Responses within a bundle are then more alike than
the main trait explains. The **bifactor model** gives every item a slope on one general factor
$\theta_0$ and, if it belongs to a bundle, on one specific factor $\theta_s$:
$$z_j=a_{j0}\theta_0+a_{js}\theta_{s(j)}+d_j ,$$
with all factors independent $N(0,1)$. Items in no bundle load on $\theta_0$ only, and each
specific factor needs at least two items. The **two-tier model** (Cai, 2010) replaces the
general factor by $P$ primary factors $\boldsymbol\theta_0$ (flexpyirt allows $P=1$ or $2$) that
may correlate with each other; the specific factors stay orthogonal to everything. A bifactor
model is the two-tier model with one primary factor on which every item loads.

**Dimension reduction.** With $S$ specific factors the model has $1+S$ dimensions, and a product
quadrature grid with $n$ points per dimension has $n^{1+S}$ nodes, which is out of reach for more
than two or three bundles. Because each item depends on $\theta_0$ and at most one $\theta_s$, and
the specific factors are independent, the marginal likelihood factorizes given the general factor
(Gibbons & Hedeker, 1992):
$$L(\mathbf y_i)=\sum_g w^{(0)}_g\prod_{s=1}^{S}\Big[\sum_t w^{(s)}_t\prod_{j\in s}P_j(y_{ij}\mid\theta_g,\theta_t)\Big]\prod_{j\in s_\varnothing}P_j(y_{ij}\mid\theta_g),$$
where $\{\theta_g\}$ and $\{\theta_t\}$ are one-dimensional grids with normalized weights
$w^{(0)}$ and $w^{(s)}$, and $s_\varnothing$ is the set of general-only items. The cost per
response pattern is $O(S\,n^2)$ instead of $O(n^{1+S})$. For a two-tier model the general node is
replaced by a node of the $P$-dimensional primary grid, with multivariate normal weights that
depend on the primary correlation, and the cost is $O(S\,n^{P+1})$ (Cai, 2010). Fewer points are
needed per dimension than for a full product grid: the default is 31 for $P=1$ and 15 for $P=2$,
as in mirt's `bfactor()`. flexpyirt fits both models for a single group.

**EM and standard errors.** The posterior of a respondent factorizes in the same way, so the
E-step gives each item expected counts on its own two-dimensional (general × specific) block grid,
and the item-wise M-step of [section 4](#4-the-bockaitkin-em-algorithm) is unchanged. In a
two-tier model the primary correlation is updated by maximizing the expected log weights of the
primary grid, which keeps EM monotone. Standard errors come from the Hessian of the reduced
marginal log-likelihood, or from Oakes' identity differentiated through the reduced E-step; both
are exact by automatic differentiation. Respondents are scored by EAP on the factorized posterior.

**Reading the result.** A specific factor absorbs the dependence within its bundle, so the
general slopes are not inflated by it. Local-dependence statistics for a bifactor fit
([section 16](#16-local-dependence-and-dimensionality)) test pairs *beyond* the specific factor: a
bundle left out of the specification shows up as large positive residual association within it.

Guide: [bifactor models](guide.md#bifactor-models), [two-tier models](guide.md#two-tier-models).

## 14. Sampling weights

In a survey, respondents may be sampled with unequal probabilities, and each carries a weight
$w_i$ (roughly the number of population members it represents). flexpyirt then maximizes the
**weighted marginal log-likelihood**, a pseudo-likelihood (Skinner, 1989):
$$\ell_w(\eta)=\sum_iw_i\log L(\mathbf y_i;\eta)=\sum_pc_p\log L_p(\eta),\qquad c_p=\sum_{i:\,\mathbf y_i=p}w_i .$$
Respondents with the same pattern are combined by summing their weights instead of counting them,
so EM, multiple groups, FIPC and every latent distribution work unchanged. Weights are used as
given (not rescaled): with integer weights the fit is exactly the fit to the data with row $i$
repeated $w_i$ times (*frequency weights*). The log-likelihood is on the scale of the weights, and
BIC uses $\log\sum_iw_i$ in place of $\log N$, as mirt does.

**Standard errors.** The inverse of the information matrix $-\nabla^2\ell_w$ is the right
covariance for frequency weights, but not for sampling weights. Under a sampling design the
estimator's variance is the sandwich
$$\operatorname{Var}(\hat\eta)\approx H^{-1}\Big(\sum_iw_i^2s_is_i^\top\Big)H^{-1},\qquad H=-\nabla^2\ell_w(\hat\eta),$$
with $s_i$ the score of respondent $i$ (Binder, 1983), and with clustered samples the outer
products are summed within clusters. These design-consistent standard errors are not implemented
yet, so flexpyirt warns whenever weights are combined with standard errors. Item fit, $M_2$,
local dependence and DIF assume unweighted counts and are not available for weighted fits.

Guide: [sampling weights](guide.md#sampling-weights).

## 15. Summed scores, information and reliability

**Information.** For any item model the Fisher information about $\theta$ is
$$I_j(\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\theta)\Big(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\Big)^2$$
(Birnbaum, 1968; Samejima, 1969), which is $a_j^2P_j(1-P_j)$ for the 2PL. flexpyirt computes the
derivative by automatic differentiation of each item model's category probabilities, so the same
code serves every model. The test information $I(\theta)=\sum_jI_j(\theta)$ gives the
conditional standard error $\mathrm{SE}(\theta)=I(\theta)^{-1/2}$ of an efficient estimate of
$\theta$. With categories scored $0,\dots,K_j-1$, the **expected score**
$E_j(\theta)=\sum_kk\,P_{jk}(\theta)$ and the **test characteristic curve**
$T(\theta)=\sum_jE_j(\theta)$ map $\theta$ to the scale of summed scores.

**The summed-score distribution.** Given $\theta$ the item scores are independent, so the
distribution of the summed score $S=\sum_jX_j$ is a convolution, computed by the
**Lord–Wingersky recursion** (Lord & Wingersky, 1984; Thissen, Pommerich, Billeaud & Williams,
1995, for polytomous items):
$$L_0(0)=1,\qquad L_j(s)=\sum_kL_{j-1}(s-k)\,P_{jk}(\theta),\qquad s=0,\dots,\textstyle\sum_j(K_j-1).$$
For dichotomous items this is the **Poisson-binomial distribution**, the distribution of a sum of
independent Bernoulli variables with different success probabilities (González, Wiberg & von
Davier, 2016). Every term of the recursion is non-negative, so it is accurate in relative terms
even in the tails; a log-space version keeps probabilities smaller than the smallest double. The
alternatives are the discrete Fourier transform of the characteristic function (Hong, 2013),
exact up to an absolute error of about $10^{-16}$ and therefore unsuitable for tail
probabilities, and, for dichotomous items, the normal, refined normal (Volkova, 1996) and Poisson
(Le Cam, 1960) approximations.

**Marginal reliability.** With error variance $1/I(\theta)$ at $\theta$, the conditional
reliability of a score is $\sigma^2I(\theta)/\{\sigma^2I(\theta)+1\}$, and its expectation over the
population is
$$\bar\rho=\int\frac{\sigma^2I(\theta)}{\sigma^2I(\theta)+1}\,\phi(\theta;\mu,\sigma^2)\,d\theta$$
(Green, Bock, Humphreys, Linn & Reckase, 1984; Thissen & Wainer, 2001). This form does not change
under a linear change of the latent scale; with $\mu=0$, $\sigma^2=1$ it equals mirt's
`marginal_rxx()`. The **empirical reliability** of a sample is
$s^2_{\hat\theta}/(s^2_{\hat\theta}+\overline{\mathrm{SE}^2})$ from the EAP scores and their
posterior SDs: the variance of the scores estimates the true-score variance and the mean posterior
variance the error variance.

**Summed-score EAP** (Thissen et al., 1995). When respondents should get the same score for the
same summed score (score tables, reporting), the posterior given the summed score replaces the
posterior given the pattern. With the latent weights $w_q$ of the fit,
$$P(s)=\sum_qL(s\mid\theta_q)\,w_q,\qquad \hat\theta(s)=\frac{\sum_q\theta_q\,L(s\mid\theta_q)\,w_q}{P(s)},$$
and the posterior SD likewise. $P(s)$ is the model-implied distribution of summed scores. A
respondent with a missing response has no summed score on this scale and gets no score.

Guide: [summed-score distribution](guide.md#summed-score-distribution),
[information, expected score and reliability](guide.md#information-expected-score-and-reliability),
[summed-score EAP](guide.md#summed-score-eap).

## 16. Local dependence and dimensionality

Unidimensional IRT assumes that one trait explains the association between items (local
independence, [section 2](#2-marginal-likelihood)). Two different questions check this, and they
are kept apart because one vote over both would be meaningless: **how many dimensions** the
items need (an integer), and whether the data are **essentially unidimensional** (Stout, 1990),
that is, whether one dominant trait remains after minor dependencies (a verdict or an index).

**Polychoric correlations.** Pearson correlations of ordinal item scores are attenuated and
depend on item difficulty, which can create spurious "difficulty factors". The polychoric
correlation assumes that the responses to items $j$ and $j'$ come from cutting a bivariate normal
pair at thresholds $\tau_{jk}$; it is the correlation $\rho$ of that pair (the tetrachoric
correlation for two binary items). In the two-step method (Olsson, 1979) the thresholds come
from the margins, $\hat\tau_{jk}=\Phi^{-1}(\hat P(Y_j\le k))$, and $\rho$ maximizes
$\sum_{ab}n_{ab}\log\pi_{ab}(\rho)$, where $\pi_{ab}$ is the bivariate normal probability of the
rectangle of cell $(a,b)$. Matrices estimated pair by pair need not be positive definite and are
then smoothed through their eigenvalues.

**How many dimensions.** These methods work on the (polychoric) correlation matrix:

- **Parallel analysis** (Horn, 1965): retain the leading eigenvalues that exceed those of data
  without common factors (normal data, or the observed data with each item permuted independently),
  here at the 95th percentile (Glorfeld, 1995; Garrido, Abad & Ponsoda, 2013).
- **Velicer's MAP** (Velicer, 1976; Velicer, Eaton & Fava, 2000): partial out $m$ components and
  choose the $m$ that minimizes the average squared (or fourth-power) partial correlation.
- **Empirical Kaiser criterion** (Braeken & van Assen, 2017): eigenvalues are compared with
  reference values that account for sampling error and for the variance already explained.
- **Hull method** (Lorenzo-Seva, Timmerman & Kiers, 2011): the factor solution at the elbow of the
  convex hull of fit against degrees of freedom.
- **Comparison data** (Ruscio & Roche, 2012): populations with $k$ factors that reproduce the
  observed correlations and marginals are generated, and $k$ is increased while it reproduces the
  observed eigenvalues significantly better.

**Residual association in a fitted model.** These statistics look for pairs of items that remain
associated after the model is fitted:

- **Q3** (Yen, 1984): the correlation over respondents of the residuals
  $d_{ij}=x_{ij}-E(X_j\mid\hat\theta_i)$ of two items. Because Q3 is slightly negative on average
  even when the model holds, **Q3\*** subtracts the mean off-diagonal Q3 (Christensen, Makransky &
  Horton, 2017).
- **LD $X^2$ and $G^2$** (Chen & Thissen, 1997): the observed $K_i\times K_j$ table of a pair is
  compared with the model-implied table $E_{ab}=n_{ij}\sum_qP_{ia}(\theta_q)P_{jb}(\theta_q)w_q$.
  The signed Cramér's $V$ shows whether the pair is more (positive) or less associated than the
  model predicts.
- **Jackknife slope index** (Edwards, Houts & Cai, 2018): refit the model without item $j$; if
  item $j$ is locally dependent with item $i$, the slope of $i$ drops, and
  $(\hat a_i-\hat a_i^{(-j)})/\mathrm{SE}(\hat a_i^{(-j)})$ stands out. flexpyirt also offers
  one-step and Newton approximations to the refit, accurate from about 15 items on.
- **Model-based parallel analysis** (Drasgow & Lissak, 1983): the second eigenvalue of the
  observed polychoric matrix is compared with its distribution in data simulated from the fitted
  unidimensional model. Because the parameters come from the same data, the $p$-value is
  conservative.

**Conditional covariances.** Under essential unidimensionality, two items have covariance close
to 0 among respondents with the same value of the trait. The nonparametric methods replace the
trait by the summed score (Stout, 1987):

- **DETECT** (Zhang & Stout, 1999; Zhang, 2007): for a partition of the items into clusters,
  $\mathrm{DETECT}=100\cdot\overline{\delta_{ij}\,\widehat{\mathrm{Cov}}(X_i,X_j\mid S)}$ over item
  pairs, with $\delta_{ij}=+1$ for pairs in the same cluster and $-1$ otherwise. It is large when
  items in the same cluster covary positively and items in different clusters negatively; values
  below about 0.2 indicate essential unidimensionality (Jang & Roussos, 2007). The exploratory
  version finds the partition by clustering on one half of the sample and evaluates it on the
  other half.
- **DIMTEST** (Stout, 1987; Nandakumar & Stout, 1993): a test of essential unidimensionality that
  compares the variance of an assessment subtest, within groups of equal score on the rest of the
  test, with its value under unidimensionality; bias is removed with unidimensional data
  resampled from the observed item response functions (Stout, Froelich & Gao, 2001). The test is
  conservative (type I error below the nominal level in flexpyirt's simulations): a small
  $p$-value is strong evidence of multidimensionality, a large one weak evidence for
  unidimensionality.

No single method is reliable on its own (Auerswald & Moshagen, 2019). `dimensionality()` runs
several of each kind and reports a separate consensus for each question.

Guide: [section 16](guide.md#16-polychoric-correlations), [section 17](guide.md#17-number-of-dimensions),
[section 18](guide.md#18-local-dependence-and-the-dimensionality-report), [section 19](guide.md#19-detect).

## References

- Auerswald, M., & Moshagen, M. (2019). How to determine the number of factors to retain in exploratory factor analysis: A comparison of extraction methods under realistic conditions. *Psychological Methods*, 24(4), 468–491.
- Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker.
- Binder, D. A. (1983). On the variances of asymptotically normal estimators from complex surveys. *International Statistical Review*, 51(3), 279–292.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores*. Addison-Wesley.
- Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two or more nominal categories. *Psychometrika*, 37, 29–51.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6, 431–444.
- Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J. van der Linden & R. K. Hambleton (Eds.), *Handbook of Modern Item Response Theory*. Springer.
- Braeken, J., & van Assen, M. A. L. M. (2017). An empirical Kaiser criterion. *Psychological Methods*, 22(3), 450–466.
- Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika*, 75(4), 581–612.
- Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking metrics and assessing item bias in item response theory. *Applied Psychological Measurement*, 12(3), 253–260.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Chalmers, R. P. (2018). Model-based measures for detecting and quantifying response bias. *Psychometrika*, 83(3), 696–732.
- Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item pairs using item response theory. *Journal of Educational and Behavioral Statistics*, 22(3), 265–289.
- Christensen, K. B., Makransky, G., & Horton, M. (2017). Critical values for Yen's Q3: Identification of local dependence in the Rasch model using residual correlations. *Applied Psychological Measurement*, 41(3), 178–194.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS B*, 39, 1–38.
- Drasgow, F., & Lissak, R. I. (1983). Modified parallel analysis: A procedure for examining the latent dimensionality of dichotomously scored item responses. *Journal of Applied Psychology*, 68(3), 363–373.
- Edwards, M. C., Houts, C. R., & Cai, L. (2018). A diagnostic procedure to detect departures from local independence in item response theory models. *Psychological Methods*, 23(1), 138–149.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80, 27–38.
- Garrido, L. E., Abad, F. J., & Ponsoda, V. (2013). A new look at Horn's parallel analysis with ordinal variables. *Psychological Methods*, 18(4), 454–474.
- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis. *Psychometrika*, 57(3), 423–436.
- Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement*, 55(3), 377–393.
- González, J., Wiberg, M., & von Davier, A. A. (2016). A note on the Poisson's binomial distribution in item response theory. *Applied Psychological Measurement*, 40(4), 302–310.
- Green, B. F., Bock, R. D., Humphreys, L. G., Linn, R. L., & Reckase, M. D. (1984). Technical guidelines for assessing computerized adaptive tests. *Journal of Educational Measurement*, 21(4), 347–360.
- Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research*, 22, 144–149.
- Hong, Y. (2013). On computing the distribution function for the Poisson binomial distribution. *Computational Statistics & Data Analysis*, 59, 41–51.
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30(2), 179–185.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 221–233). University of California Press.
- Jang, E. E., & Roussos, L. (2007). An investigation into the dimensionality of TOEFL using conditional covariance-based nonparametric approach. *Journal of Educational Measurement*, 44(1), 1–21.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Kim, S., & Lee, W.-C. (2006). An extension of four IRT linking methods for mixed-format tests. *Journal of Educational Measurement*, 43, 193–213.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- Kopf, J., Zeileis, A., & Strobl, C. (2015). Anchor selection strategies for DIF analysis: Review, assessment, and new approaches. *Educational and Psychological Measurement*, 75(1), 22–56.
- Le Cam, L. (1960). An approximation theorem for the Poisson binomial distribution. *Pacific Journal of Mathematics*, 10(4), 1181–1197.
- Li, Y. H., & Lissitz, R. W. (2000). An evaluation of the accuracy of multidimensional IRT linking. *Applied Psychological Measurement*, 24, 115–138.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Erlbaum.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings". *Applied Psychological Measurement*, 8, 453–461.
- Lorenzo-Seva, U., Timmerman, M. E., & Kiers, H. A. L. (2011). The Hull method for selecting the number of common factors. *Multivariate Behavioral Research*, 46(2), 340–364.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information estimation and goodness-of-fit testing in $2^n$ contingency tables. *Journal of the American Statistical Association*, 100, 1009–1020.
- Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit testing in multidimensional contingency tables. *Psychometrika*, 71, 713–732.
- Meade, A. W. (2010). A taxonomy of effect size measures for the differential functioning of items and scales. *Journal of Applied Psychology*, 95(4), 728–743.
- Mislevy, R. J. (1984). Estimating latent distributions. *Psychometrika*, 49, 359–381.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56, 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Nandakumar, R., & Stout, W. (1993). Refinements of Stout's procedure for assessing latent trait unidimensionality. *Journal of Educational Statistics*, 18(1), 41–68.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Ogasawara, H. (2001). Standard errors of item response theory equating/linking by response function methods. *Applied Psychological Measurement*, 25, 53–67.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. *Psychometrika*, 44(4), 443–460.
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement*, 24, 50–64.
- Oshima, T. C., Davey, T. C., & Lee, K. (2000). Multidimensional linking: Four practical approaches. *Journal of Educational Measurement*, 37, 357–373.
- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*. Danish Institute for Educational Research.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63, 581–592.
- Ruscio, J., & Roche, B. (2012). Determining the number of factors to retain in an exploratory factor analysis using comparison data of known factorial structure. *Psychological Assessment*, 24(2), 282–292.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Skinner, C. J. (1989). Domain means, regression and multivariate analysis. In C. J. Skinner, D. Holt & T. M. F. Smith (Eds.), *Analysis of Complex Surveys* (pp. 59–87). Wiley.
- Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement*, 7, 201–210.
- Stout, W. (1987). A nonparametric approach for assessing latent trait unidimensionality. *Psychometrika*, 52(4), 589–617.
- Stout, W. (1990). A new item response theory modeling approach with applications to unidimensionality assessment and ability estimation. *Psychometrika*, 55(2), 293–325.
- Stout, W., Froelich, A. G., & Gao, F. (2001). Using resampling methods to produce an improved DIMTEST procedure. In A. Boomsma, M. A. J. van Duijn, & T. A. B. Snijders (Eds.), *Essays on Item Response Theory* (pp. 357–375). Springer.
- Thissen, D., & Wainer, H. (Eds.). (2001). *Test Scoring*. Lawrence Erlbaum Associates.
- Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of Polytomous Item Response Theory Models* (pp. 43–75). Routledge.
- Thissen, D., Pommerich, M., Billeaud, K., & Williams, V. S. L. (1995). Item response theory for scores on tests including polytomous items with ordered responses. *Applied Psychological Measurement*, 19(1), 39–49.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scandinavian Journal of Statistics*, 35, 335–353.
- Velicer, W. F. (1976). Determining the number of components from the matrix of partial correlations. *Psychometrika*, 41(3), 321–327.
- Velicer, W. F., Eaton, C. A., & Fava, J. L. (2000). Construct explication through factor or component analysis: A review and evaluation of alternative procedures for determining the number of factors or components. In R. D. Goffin & E. Helmes (Eds.), *Problems and Solutions in Human Assessment* (pp. 41–71). Kluwer.
- Volkova, A. Y. (1996). A refinement of the central limit theorem for sums of independent random indicators. *Theory of Probability and Its Applications*, 40(4), 791–794.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54, 427–450.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Yen, W. M. (1984). Effects of local item dependence on the fit and equating performance of the three-parameter logistic model. *Applied Psychological Measurement*, 8(2), 125–145.
- Zhang, D., & Davidian, M. (2001). Linear mixed models with flexible distributions of random effects for longitudinal data. *Biometrics*, 57, 795–802.
- Zhang, J. (2007). Conditional covariance theory and DETECT for polytomous items. *Psychometrika*, 72(1), 69–91.
- Zhang, J., & Stout, W. (1999). The theoretical DETECT index of dimensionality and its application to approximate simple structure. *Psychometrika*, 64(2), 213–249.
