# flexpyequate design (DESIGN)

English | [日本語](DESIGN.ja.md)

> Status: draft v0.3 (2026-11-25). Milestones E1–E5 are implemented and validated against R
> (§7); E6 and later are plans. Decisions are marked "Decision".

## 1. Purpose and non-goals

### Purpose
- Equating of **score distributions**: given the score distributions of two test forms X and Y,
  find the function $e_Y(x)$ that puts form-X scores on the form-Y scale.
- Cover the observed-score methods of Kolen & Brennan (2014) and the kernel method of von Davier,
  Holland & Thayer (2004), each implemented from the published formulas and checked numerically
  against R (`equate`, `kequate`, `SNSequate`).
- A light distribution: NumPy, SciPy and pandas only.

### Non-goals
- **Anything that needs item parameters**: IRT true-score and observed-score equating, scale
  linking and raw-to-scale tables stay in flexpyirt. flexpyequate never imports flexpyirt or JAX
  (decision of issue #87).
- Item-level data. The input is always a score distribution (or the scores it is tabulated from).
- Reproducing the argument names or object structure of the R packages.

### Principles (shared with flexpyirt)
1. Correctness over speed over number of features. A method is "done" when its comparison test
   against an R reference passes; only validated methods appear in the README.
2. Formulas first: the code is written from the formulas in §4, and the R packages are reference
   values only. Where an R package deviates from the published formula, the deviation is recorded
   in §7 and the published formula wins.
3. Pure functions on arrays inside, validation and DataFrames in the public layer.

## 2. Data collection designs

| Design | Data | Container |
|---|---|---|
| Equivalent groups (EG) | X from sample 1, Y from an independent sample 2 | two `ScoreDistribution` |
| Single group (SG) | X and Y from the same examinees | one `BivariateScoreDistribution` |
| Counterbalanced (CB) | two SG samples taking the forms in opposite orders | two `BivariateScoreDistribution` (E6) |
| Non-equivalent groups with anchor test (NEAT) | (X, A) from population P and (Y, A) from population Q | two `BivariateScoreDistribution`, anchor as the second variable (E4, E5; kernel equating in E6) |

For NEAT, two families of methods exist and both are planned: **chained** equating (X → A on P,
then A → Y on Q) and **post-stratification** (a synthetic population
$T = wP + (1 - w)Q$; Tucker, Levine, Braun–Holland, frequency estimation, kernel PSE).

## 3. Notation

| Symbol | Meaning |
|---|---|
| $x_j,\ j = 1..J$; $y_k,\ k = 1..K$ | possible scores of X and Y (equally spaced, increment $\delta$) |
| $n_j$, $N = \sum_j n_j$ | frequency of $x_j$, sample size |
| $r_j = n_j / N$, $s_k$ | score probabilities of X and Y |
| $F$, $G$ | discrete cumulative distributions of X and Y |
| $\mu_X, \sigma_X^2$ | mean and variance of X |
| $e_Y(x)$ | equating function, X to the Y scale |

## 4. Methods

Formulas are cited by author, year and chapter. Equation numbers are deliberately not given yet:
they are to be added after checking them against the printed books.

### 4.1 Score distributions (E1)

`ScoreDistribution` holds the equally spaced possible scores and their (non-negative, possibly
non-integer) frequencies. Moments use the population form
$\sigma_X^2 = \sum_j r_j (x_j - \mu_X)^2$ unless `ddof=1` is asked for.

**Percentile rank** (Kolen & Brennan, 2014, ch. 2; written here as a proportion, the public
methods use the 0–100 scale of the book). A discrete score is
treated as uniform on $[x_j - \delta/2,\, x_j + \delta/2)$. With $x^*$ the possible score nearest
to $x$ (halves rounded up),

$$P(x) = F(x^* - \delta) + \frac{x - (x^* - \delta/2)}{\delta}\,\bigl[F(x^*) - F(x^* - \delta)\bigr]$$

for $x_1 - \delta/2 \le x < x_J + \delta/2$, with $P = 0$ below and $P = 1$ above.

**Percentile** (inverse; Kolen & Brennan, 2014, ch. 2). For a proportion $p$,

$$P^{-1}_U(p) = x^*_U - \frac{\delta}{2} + \delta\,\frac{p - F(x^*_U - \delta)}{F(x^*_U) - F(x^*_U - \delta)},
\qquad x^*_U = \min\{x_j : F(x_j) > p\},$$

$$P^{-1}_L(p) = x^*_L + \frac{\delta}{2} + \delta\,\frac{p - F(x^*_L)}{F(x^*_L + \delta) - F(x^*_L)},
\qquad x^*_L = \max\{x_j : F(x_j) < p\},$$

with $P^{-1}_U(1) = x_J + \delta/2$ and $P^{-1}_L(0) = x_1 - \delta/2$, and
$P^{-1}(p) = \tfrac12\,[P^{-1}_U(p) + P^{-1}_L(p)]$. The two differ only where a run of zero
frequencies makes $F$ flat; the average is then the middle of the flat stretch.

**Decision**: frequencies may be non-integer, because presmoothed distributions are passed through
the same container.

### 4.2 Identity, mean and linear equating (E1)

- Identity: $e_Y(x) = x$.
- Mean (Kolen & Brennan, 2014, ch. 2): $e_Y(x) = x - \mu_X + \mu_Y$.
- Linear (ch. 2): $e_Y(x) = \dfrac{\sigma_Y}{\sigma_X}(x - \mu_X) + \mu_Y$.

For SG data the same formulas apply to the two margins (the dependence between X and Y matters
only for standard errors, E7).

**Decision**: the population variance (`ddof=0`) is the default because that is the formula in
Kolen & Brennan; `ddof=1` reproduces R `equate`, which divides by $N - 1$ (§7).

### 4.3 Equipercentile equating (E1)

$$e_Y(x) = Q^{-1}\bigl(P(x)\bigr)$$

(Kolen & Brennan, 2014, ch. 2), with $P$ the percentile-rank function of X and $Q^{-1}$ the
percentile function of Y from §4.1. Equated scores lie in
$[y_1 - \delta_Y/2,\ y_K + \delta_Y/2]$.

### 4.4 Log-linear presmoothing (E2)

Polynomial log-linear model for the score frequencies (Holland & Thayer, 2000):

$$\log(N r_j) = \beta_0 + \sum_{i=1}^{C} \beta_i\, x_j^{\,i},$$

and for a bivariate distribution

$$\log(N p_{jk}) = \beta_0 + \sum_{i=1}^{C_X} \beta_{Xi}\, x_j^{\,i} + \sum_{i=1}^{C_Y} \beta_{Yi}\, y_k^{\,i}
+ \sum_{(a,b)} \beta_{ab}\, x_j^{\,a} y_k^{\,b}.$$

Fitted by maximum likelihood (Poisson = multinomial likelihood) with Newton's method. The fitted
distribution reproduces the first $C$ sample moments of the observed one (and the cross moments
$E[X^aY^b]$ for the included cross terms); this is a unit test.

**Decision**: the design matrix is orthonormalized (QR of the centred, scaled powers) before
fitting. Fitted frequencies do not depend on the basis, and raw powers of scores up to 40–100 with
degree 6–10 are too ill-conditioned for Newton's method. Coefficients are therefore not reported in
the raw-power basis; the result exposes the fitted distribution, the design matrix and the fit
statistics (Poisson log-likelihood, likelihood-ratio $G^2$, Pearson $\chi^2$, Freeman–Tukey
$\chi^2$, AIC and degrees of freedom). BIC is left out for now: R's `BIC()` on the Poisson `glm`
uses the number of cells as the sample size, the equating literature uses the number of
examinees, and there is no reference to validate a choice against.

The asymptotic covariance of $\sqrt{N}(\hat r - r)$ is $C C^\top$ with the "C-matrix" of Holland &
Thayer (1987): with $B$ the design matrix (no intercept column) and
$M = (D_{\sqrt r} - \sqrt r\, r^\top) B = QR$,

$$C = D_{\sqrt r}\, Q, \qquad \operatorname{Cov}(\hat r) = \tfrac1N\, C C^\top .$$

Without presmoothing (saturated model) $C = D_{\sqrt r} - r \sqrt r^\top$, which gives the
multinomial covariance $D_r - r r^\top$.

### 4.5 Kernel equating (E3 for EG; E6 for the other designs)

Five steps (von Davier, Holland & Thayer, 2004): presmoothing (§4.4), estimation of the score
probabilities $r, s$ through the design function, continuization, equating, and the standard error
of equating.

**Continuization** with a Gaussian kernel (ch. 4). With bandwidth $h_X > 0$ and
$a_X^2 = \sigma_X^2 / (\sigma_X^2 + h_X^2)$,

$$F_{h}(x) = \sum_j r_j\, \Phi\bigl(R_{jX}(x)\bigr), \qquad
R_{jX}(x) = \frac{x - a_X x_j - (1 - a_X)\mu_X}{a_X h_X},$$

$$f_{h}(x) = \frac{1}{a_X h_X} \sum_j r_j\, \phi\bigl(R_{jX}(x)\bigr).$$

The continuized variable has mean $\mu_X$ and variance $\sigma_X^2$ for every $h_X$. As
$h_X \to \infty$ the kernel equating function tends to the linear one.

**Bandwidth** (ch. 4): minimize

$$\mathrm{PEN}(h) = \sum_j \bigl(r_j - f_h(x_j)\bigr)^2 + K \sum_j A_j (1 - B_j),$$

where $A_j = 1$ if $f_h'(x_j - w) < 0$ and $B_j = 0$ if $f_h'(x_j + w) > 0$ ($w = 1/4$): the second
term counts the scores at which the density is U-shaped. $K = 1$ by default.

**Decision**: the minimizer is found by a grid search on $h$ (400 points, log-spaced between
0.05 and 10 score increments) followed by a bounded scalar minimization around the best grid
point, because the penalty is not unimodal in general and is discontinuous when $K > 0$.

**Decision**: the penalty is evaluated with scores measured in units of their increment
$\delta$ (the book assumes integer scores, where a density and a probability are comparable), and
the bandwidth is reported on the score scale. Kernel equating is then equivariant under linear
transformations of either score scale (a unit test).

**Equating** (ch. 4): $e_Y(x) = G_{h_Y}^{-1}\bigl(F_{h_X}(x)\bigr)$, solved by a safeguarded
Newton iteration to machine precision.

**Standard error of equating** (ch. 5). With $J_{e_Y} = (\partial e_Y/\partial r,\,
\partial e_Y/\partial s)$,

$$\frac{\partial e_Y(x)}{\partial r_j} = \frac{1}{G'(e_Y(x))} \frac{\partial F_h(x)}{\partial r_j}, \qquad
\frac{\partial e_Y(x)}{\partial s_k} = -\frac{1}{G'(e_Y(x))} \frac{\partial G_h(y)}{\partial s_k}\Big|_{y = e_Y(x)},$$

$$\frac{\partial F_h(x)}{\partial r_j} = \Phi\bigl(R_{jX}(x)\bigr) - M_{jX}(x)\, f_h(x), \qquad
M_{jX}(x) = \tfrac12 (x - \mu_X)(1 - a_X^2)\Bigl(\frac{x_j - \mu_X}{\sigma_X}\Bigr)^2 + (1 - a_X)\, x_j,$$

and, for EG with independent samples of sizes $N$ and $M$,

$$\mathrm{SEE}_Y(x) = \sqrt{\tfrac1N \bigl\lVert \tfrac{\partial e_Y}{\partial r}^{\!\top} C_r \bigr\rVert^2
+ \tfrac1M \bigl\lVert \tfrac{\partial e_Y}{\partial s}^{\!\top} C_s \bigr\rVert^2 } .$$

The derivative formulas are checked against finite differences in the unit tests, and the SEE
against a parametric bootstrap. Without presmoothing the multinomial covariance is used
($C = D_{\sqrt r} - r\sqrt r^\top$, §4.4).

**Percent relative error** (ch. 4): $\mathrm{PRE}(p) = 100\,[\mu_p(e_Y(X)) - \mu_p(Y)]/\mu_p(Y)$
for the $p$-th raw moments, $p = 1..10$, with $e_Y(X)$ taking the value $e_Y(x_j)$ with
probability $r_j$.

**SEED** (E6): the standard error of the difference between two equating functions, e.g. kernel
versus linear, $\lVert (J_{e_1} - J_{e_2}) J_{DF} C \rVert$ (ch. 5).

### 4.6 NEAT, linear methods (E4)

Population 1 takes X and the anchor V, population 2 takes Y and V (Kolen & Brennan, 2014, ch. 4).
Each population is one `BivariateScoreDistribution` with the anchor as the second variable.

**Synthetic population** $T = w_1 P_1 + w_2 P_2$, $w_1 + w_2 = 1$. The Tucker and the Levine
observed-score methods equate linearly in $T$,
$l_Y(x) = \dfrac{\sigma_s(Y)}{\sigma_s(X)}[x - \mu_s(X)] + \mu_s(Y)$, with

$$\mu_s(X) = \mu_1(X) - w_2 \gamma_1 [\mu_1(V) - \mu_2(V)], \qquad
\mu_s(Y) = \mu_2(Y) + w_1 \gamma_2 [\mu_1(V) - \mu_2(V)],$$

$$\sigma_s^2(X) = \sigma_1^2(X) - w_2 \gamma_1^2 [\sigma_1^2(V) - \sigma_2^2(V)] + w_1 w_2 \gamma_1^2 [\mu_1(V) - \mu_2(V)]^2,$$

$$\sigma_s^2(Y) = \sigma_2^2(Y) + w_1 \gamma_2^2 [\sigma_1^2(V) - \sigma_2^2(V)] + w_1 w_2 \gamma_2^2 [\mu_1(V) - \mu_2(V)]^2 .$$

| Method | $\gamma_1$ ($\gamma_2$ likewise from Y and V in population 2) |
|---|---|
| Tucker (regression of X on V is linear and the same in both populations) | $\sigma_1(X, V) / \sigma_1^2(V)$ |
| Levine, internal anchor (classical congeneric model) | $\sigma_1^2(X) / \sigma_1(X, V)$ |
| Levine, external anchor | $[\sigma_1^2(X) + \sigma_1(X, V)] / [\sigma_1^2(V) + \sigma_1(X, V)]$ |

**Levine true-score** equating, applied to observed scores, and **chained linear** equating do
not use a synthetic population:

$$l_Y(x) = \frac{\gamma_2}{\gamma_1}[x - \mu_1(X)] + \mu_2(Y) + \gamma_2[\mu_1(V) - \mu_2(V)],$$

$$l_Y(x) = \mu_2(Y) + \frac{\sigma_2(Y)}{\sigma_2(V)}\Bigl[\mu_1(V) + \frac{\sigma_1(V)}{\sigma_1(X)}(x - \mu_1(X)) - \mu_2(V)\Bigr].$$

**Braun–Holland**: linear equating with the means and standard deviations of the
frequency-estimation (post-stratification) distributions (Kolen & Brennan, 2014, ch. 5),

$$f_s(x) = w_1 f_1(x) + w_2 \sum_v f_1(x \mid v)\, h_2(v), \qquad
g_s(y) = w_1 \sum_v g_2(y \mid v)\, h_1(v) + w_2\, g_2(y),$$

with $h_1, h_2$ the anchor distributions. The same synthetic distributions serve frequency
estimation in E5.

**Decision**: the default weight is proportional to the sample sizes, $w_1 = N_1/(N_1 + N_2)$
(`synthetic_weight=None`).

**Decision**: the Levine methods require `anchor="internal"` or `"external"` explicitly. There
is no default, because the wrong choice gives a wrong answer without any warning.

**Decision**: where an anchor score occurs in one population only, $f_1(x \mid v)$ (or
$g_2(y \mid v)$) is not estimable. That part of the other population is left out, the synthetic
distribution is renormalized, and a `SparseDataWarning` reports the probability left out.
Presmoothing the joint distributions avoids the situation.

**Decision**: `x` and `y` must have the same possible anchor scores. The anchor is one test
taken by both populations; different anchor scales are rejected rather than silently compared.

**Decision**: `ddof` applies to the variances and covariances of the Tucker and Levine
observed-score methods. It cancels in $\gamma$, in Levine true-score and in chained linear
equating (ratios of moments of one population), and the Braun–Holland method uses the moments of
the synthetic distributions, which have no sample size of their own.

### 4.7 NEAT, equipercentile methods (E5)

Kolen & Brennan (2014, ch. 5). Both methods reuse the percentile-rank and percentile functions of
§4.1.

**Frequency estimation** (post-stratification on the anchor): equipercentile equating of the
synthetic distributions $f_s$ and $g_s$ of §4.6,

$$e_Y(x) = Q_s^{-1}\bigl(P_s(x)\bigr).$$

**Chained equipercentile** equating: X is equated to the anchor in population 1 and the anchor to
Y in population 2,

$$e_Y(x) = Q_2^{-1}\Bigl(H_2\bigl(H_1^{-1}(P_1(x))\bigr)\Bigr),$$

with $P_1$, $H_1$ the percentile-rank functions of X and V in population 1 and $H_2$, $Q_2$ those
of V and Y in population 2. $H_2$ is evaluated at the non-integer anchor equivalents with the
continuous percentile-rank function of §4.1.

Presmoothing is not an option of `equate()`: pass the presmoothed joint distributions
(`presmooth(xa, degree=(…, …), cross_terms=…).distribution`).

**Decision**: for frequency estimation, `EquatingResult.x` and `.y` are the synthetic
distributions (total frequency $N_1 + N_2$), because that is the population in which the
equating is defined; `moments` then describes the synthetic population. For chained equating
they are X in population 1 and Y in population 2, and the two anchor distributions are kept in
`anchor_distributions` so that `transform()` can equate arbitrary scores.

**Decision** (all anchor methods): `x` and `y` must have the same possible anchor scores. The
anchor is one test taken by both populations; different anchor scales are rejected rather than
silently compared.

## 5. Public API sketch

```python
import flexpyequate as fe

x = fe.ScoreDistribution.from_scores(x_scores, score_range=(0, 40))
y = fe.ScoreDistribution.from_frequencies(scores=range(41), frequencies=y_counts)
xy = fe.BivariateScoreDistribution.from_scores(
    x_scores, y_scores, x_range=(0, 40), y_range=(0, 40)
)  # single group

result = fe.equate(x, y, method="equipercentile")  # EG
result = fe.equate(xy, method="linear")  # SG: one bivariate distribution
result.table  # DataFrame: score, equated
result.moments  # DataFrame: mean, sd, skewness, kurtosis of X, Y and equated X
result.transform(x_new)  # equate arbitrary (also non-integer) scores

smooth_x = fe.presmooth(x, degree=6)  # LoglinearResult
smooth_xy = fe.presmooth(xy, degree=(4, 4), cross_terms=[(1, 1)])
smooth_x.distribution  # ScoreDistribution with the fitted frequencies
smooth_x.fit_statistics  # Series: likelihood_ratio, pearson, freeman_tukey, aic, ...
smooth_x.c_matrix  # C with Cov(r_hat) = C C' / N

ke = fe.kernel_equate(smooth_x, smooth_y)  # Gaussian kernel, penalty bandwidths
ke = fe.kernel_equate(x, y, bandwidth_x=0.6, bandwidth_y=0.6)  # observed distributions
ke.table  # DataFrame: score, equated, see
ke.continuized  # DataFrame: form, score, probability, cdf, density
ke.bandwidth_x, ke.bandwidth_y
ke.percent_relative_error
fe.penalty_bandwidth(smooth_x, penalty_weight=1.0)

# Non-equivalent groups: form X with the anchor, form Y with the anchor.
xa = fe.BivariateScoreDistribution.from_scores(x_scores, x_anchor, x_range=(0, 36), y_range=(0, 12))
ya = fe.BivariateScoreDistribution.from_scores(y_scores, y_anchor, x_range=(0, 36), y_range=(0, 12))
tucker = fe.equate(xa, ya, method="tucker")  # w1 = N1 / (N1 + N2)
levine = fe.equate(xa, ya, method="levine_observed", anchor="internal", synthetic_weight=1.0)
tucker.gamma, tucker.synthetic_moments, tucker.synthetic_weight
fe.equate(xa, ya, method="frequency_estimation")
fe.equate(xa, ya, method="chained_equipercentile")
```

- All arguments after the data are keyword-only. Options are `Literal` strings in lower
  snake_case: `method="identity" | "mean" | "linear" | "equipercentile"` (EG, SG) and
  `"tucker" | "levine_observed" | "levine_true" | "braun_holland" | "chained_linear" |
  "frequency_estimation" | "chained_equipercentile"` (NEAT).
- Results are frozen dataclasses (`EquatingResult`, `LoglinearResult`, `KernelEquatingResult`)
  holding NumPy arrays (read-only) and exposing DataFrames as properties.
- Errors are `flexpyequate.errors` classes: `ScoreDistributionError` (invalid scores or
  frequencies), `SpecificationError` (inconsistent arguments), `ConvergenceWarning`,
  `SparseDataWarning`.
- Names are Python names; the R name is given in the docstring where it helps (`ddof=1` ↔ R
  `equate`'s variance, `penalty_weight` ↔ `kequate`'s `KPEN`).
- Planned (E6): `kernel_equate(..., design="single_group" | "counterbalanced" |
  "chained" | "post_stratification")`, `KernelEquatingResult.seed(other)`.

## 6. Layout

```
packages/flexpyequate/
  pyproject.toml        hatchling + hatch-vcs (version from the repository tag, root = "../..")
  README.md, README.ja.md
  docs/DESIGN.md, DESIGN.ja.md
  src/flexpyequate/
    distributions.py    ScoreDistribution, BivariateScoreDistribution
    equating.py         equate(), EquatingResult
    _neat.py            formulas for non-equivalent groups with an anchor test
    presmoothing.py     presmooth(), LoglinearResult
    kernel.py           kernel_equate(), KernelEquatingResult
    errors.py
  tests/                unit tests, R comparison tests, reference/*.json
validation/r/equate_*.R reference generators (repository root; not run in CI)
```

## 7. Validation plan

Reference values come from R and are committed as JSON under `tests/reference/`; each file records
the R and package versions. CI does not run R. To regenerate, run from the repository root:

```bash
Rscript validation/r/equate_e1_reference.R
Rscript validation/r/equate_e2_reference.R
Rscript validation/r/equate_e3_reference.R
Rscript validation/r/equate_e4_reference.R
Rscript validation/r/equate_e5_reference.R
```

### Cases

| ID | Contents | R reference | MS |
|---|---|---|---|
| Q1 | Mean and linear equating, EG (`ACTmath`, simulated forms with different lengths and increments) and SG (bivariate simulated) | `equate::equate(type = "mean", sx = 1, sy = 1)`, `equate(type = "linear")` | E1 |
| Q2 | Percentile ranks and equipercentile equating, EG and SG, including distributions with zero frequencies inside and at the ends, and non-unit score increments | `equate::px`, `equate::equate(type = "equipercentile")`; an independent R implementation of Kolen & Brennan ch. 2 without rounding | E1 |
| Q3 | Log-linear presmoothing: univariate (degrees 2–8), bivariate with and without cross terms; equipercentile equating of the smoothed distributions | `stats::glm(family = poisson)` with a strict tolerance; `equate::presmoothing(smoothmethod = "loglinear")`; `equate(..., smoothmethod = "loglinear")` | E2 |
| Q4 | Kernel equating, EG: continuized CDFs, equated scores, SEE, PRE with given bandwidths; with presmoothing (three data sets) and without (one) | `kequate::kequate("EG", ...)`; `SNSequate::ker.eq`; an independent R implementation with a strict root tolerance and a finite-difference SEE | E3 |
| Q5 | Penalty function values and penalty bandwidths ($K = 0$ and $K = 1$) | `kequate` (`KPEN = 0`), `SNSequate::bandwidth`, independent R implementation | E3 |
| Q6 | Tucker, Levine observed score, Braun–Holland (four synthetic weights each), Levine true score, chained linear; internal anchor (`KBneat`) and external anchor (simulated, unequal sample sizes and form lengths) | `equate::equate(type = "linear", method = "tucker" / "levine" / "braun/holland" / "chained", lts = )`; the Kolen & Brennan formulas written out in R with population moments | E4 |
| Q7 | Frequency estimation (four synthetic weights) and chained equipercentile equating on `KBneat` and the simulated external-anchor data; both methods on `KBneat` presmoothed with a bivariate log-linear model | `equate::equate(type = "equipercentile", method = "frequency estimation" / "chained", smoothmethod = )`; Kolen & Brennan written out in R without rounding | E5 |
| Q8 | Kernel equating SG / CB / NEAT CE / NEAT PSE, SEED | `kequate`, `SNSequate` | E6 |

### Tolerances (maximum absolute difference unless stated)

| Quantity | Reference | Tolerance |
|---|---|---|
| Moments, mean / linear equated scores, slope, intercept | `equate` (`ddof=1`) | 1e-10 |
| Percentile ranks | `equate::px` | 1e-12 |
| Equipercentile equated scores | independent R implementation | 1e-10 |
| Equipercentile equated scores (ranks strictly between 0 and 1) | `equate` | 1e-7 |
| NEAT linear methods: slope, intercept, $\gamma$, synthetic means and SDs | `equate` (`ddof=1`); independent R implementation (`ddof=0`) | 1e-10 |
| Frequency-estimation synthetic probabilities | independent R implementation | 1e-13 |
| Frequency estimation and chained equipercentile equated scores | independent R implementation | 1e-10 |
| Frequency estimation and chained equipercentile equated scores (ranks strictly between 0 and 1) | `equate` | 5e-7 |
| The same on presmoothed joint distributions (tables smoothed by `equate`, and end to end) | `equate` | 5e-6 |
| Presmoothed frequencies, relative | `glm` (`epsilon = 1e-13`) | 1e-8 |
| Presmoothed frequencies, relative | `equate::presmoothing` (default `glm` tolerance) | 1e-5 |
| $G^2$, Pearson and Freeman–Tukey $\chi^2$, log-likelihood, AIC (relative) | `glm` | 1e-7 |
| Equipercentile equating of presmoothed distributions | `equate` | 1e-5 |
| Covariance of the fitted probabilities $CC^\top/N$ (relative to its largest entry) | `kequate` | 1e-9 |
| Kernel CDFs $F_h$, $G_h$; density | `kequate`; independent R implementation | 1e-12 |
| Kernel equated scores | independent R implementation | 1e-9 |
| Kernel equated scores | `kequate` without its variance factor | 1e-6 |
| SEE (relative) | `kequate` without its variance factor; independent finite-difference SEE | 1e-6 |
| PRE | `kequate` without its variance factor | 1e-5 |
| Penalty values (relative) | independent R implementation | 1e-10 |
| Bandwidth, $K = 0$ | `kequate` without its variance factor | 1e-6 |
| Bandwidth, $K = 0$ and $K = 1$ | independent R implementation | 1e-6 |
| Bandwidth, $K = 0$ and $K = 1$ | `SNSequate` | 1e-3 |
| Kernel equated scores / SEE (relative) | `SNSequate::ker.eq` | 1e-4 / 1e-4 |
| Kernel equated scores / SEE (relative) / bandwidth | ordinary `kequate` call | 5e-3 / 1e-3 / 1e-4 |

Observed differences are well inside these: about 1e-8 for the SEE and 2e-7 for the equated
scores against `kequate` (the latter is `kequate`'s root tolerance).

### Known differences from the R packages

- **`equate` divides variances by $N - 1$**; Kolen & Brennan use $N$. `ddof=1` reproduces
  `equate`; the default is `ddof=0`.
- **`equate`'s `type = "mean"`** has slope (range of Y) / (range of X), which is 1 only when the
  two forms have the same score range. flexpyequate's mean equating has slope 1 (ch. 2 of
  Kolen & Brennan); the reference values are generated with `sx = 1, sy = 1`.
- **`equate`'s Braun–Holland (and frequency-estimation) synthetic distributions** mix the two
  populations as counts, $w\,n_1(x, v) + (1 - w)\,n_1(x, v)\,n_2(v)/n_1(v)$, so the weight of
  population 1 is in effect $wN_1/(wN_1 + (1 - w)N_2)$ instead of $w$ (with the default $w$:
  $N_1^2/(N_1^2 + N_2^2)$). flexpyequate mixes probabilities, as in Kolen & Brennan; `equate` is
  reproduced by passing the effective weight as `synthetic_weight` (intercept difference 0.02 for
  $N_1 = 1200$, $N_2 = 2000$). Its synthetic standard deviations also divide by the total count
  minus 1, which cancels in the slope.
- **`equate` rounds** percentile ranks and cumulative proportions to 10 decimals inside its
  equipercentile routine. The rounding error (up to 5e-11) is divided by a relative frequency, so
  agreement is limited to about 1e-8; hence the unrounded independent R implementation for the
  tight check.
- **Percentile rank exactly 0 or 1**: `equate` returns the end of the Y range
  ($y_1 - \delta/2$ or $y_K + \delta/2$). Kolen & Brennan's definition, followed here, is the mean
  of the upper and lower percentiles, which differs when Y has no observations at that end (ACT
  mathematics, $x = 0$: 0.0 instead of −0.5). The comparison with `equate` leaves these scores
  out; the independent implementation covers them.
- **`kequate` multiplies the variance by $N/(N - 1)$** in the continuization
  ($a_X^2 = \sigma_X^2/(\sigma_X^2 + h_X^2)$) while keeping the derivative formulas of von Davier
  et al., which assume $\sigma_X^2 = \sum_j r_j (x_j - \mu_X)^2$. flexpyequate follows the book
  (as `SNSequate` does). For the tight comparison `kequate` is called with probability vectors and
  sample sizes multiplied by $10^{8}$, which makes the factor negligible, and its SEE is rescaled
  by $10^{4}$ (the SEE is exactly proportional to $N^{-1/2}$). A second comparison uses the
  ordinary `kequate` call with tolerances that cover the factor (it moved the equated scores by
  up to 1.2e-3 with $N \approx 600$ and a bandwidth of 2).
- **`kequate` solves $G_h(y) = F_h(x)$ with `uniroot(tol = 1e-6)`**, which limits agreement of the
  equated scores to about 1e-6.
- **`kequate`'s second penalty term** also counts scores at which the density has a local maximum
  ($f' > 0$ on the left and $< 0$ on the right), unlike ch. 4 of the book and `SNSequate`.
  flexpyequate follows the book; with $K = 0$ (`kequate`'s default) the term does not enter.
  flexpyequate's default is $K = 1$, as in the book.
- **`kequate` and `SNSequate` assume scores 0, 1, 2, …**; other score scales are covered by the
  equivariance unit test only.
- **`SNSequate::ker.eq`** searches the equated score in $(-1, 101)$ and fails when a score
  equates below −1 (one of the two cases tried); its bandwidths come from `optim()` started at
  0.65.
- A two-point finite difference with step 1e-6 is not accurate enough for the independent SEE
  where a score probability is about 5e-6 (error 1.4e-3); the generator uses a five-point stencil
  with step 1e-7.

## References

- Holland, P. W., & Thayer, D. T. (1987). *Notes on the use of log-linear models for fitting
  discrete probability distributions* (ETS RR-87-31).
- Holland, P. W., & Thayer, D. T. (2000). Univariate and bivariate loglinear models for discrete
  test score distributions. *Journal of Educational and Behavioral Statistics, 25*, 133–183.
- Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking* (3rd ed.). Springer.
- von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method of test equating*.
  Springer.
- Albano, A. D. (2016). equate: An R package for observed-score linking and equating. *Journal of
  Statistical Software, 74*(8).
- Andersson, B., Bränberg, K., & Wiberg, M. (2013). Performing the kernel method of test equating
  with the package kequate. *Journal of Statistical Software, 55*(6).
- González, J. (2014). SNSequate: Standard and nonstandard statistical models and methods for test
  equating. *Journal of Statistical Software, 59*(7).
