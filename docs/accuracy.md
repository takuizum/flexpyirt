# Accuracy and validation

English | [日本語](accuracy.ja.md)

How flexpyirt's results are checked, how closely they agree with R `mirt`, and where the two
programs deliberately differ.

## How results are checked

flexpyirt is implemented from the formulas (see [Theory](theory.md)), not translated from mirt.
mirt (Chalmers, 2012) serves as an independent reference:

- Reference values are computed with R `mirt` and stored with the source code; the test suite
  compares against them on every change, so a feature is listed as supported only after it agrees
  with mirt (or, where mirt has no equivalent, with an independent R implementation).
- Both programs are run under **the same conditions**: the same quadrature grid, a much stricter
  convergence criterion on the mirt side, no acceleration, and the same priors.
- Data are public data sets bundled with mirt (`LSAT7`, `Science`) and simulated data with fixed
  seeds, including multiple groups, missing responses, fixed items and boundary-prone models.

Besides agreement with mirt, the tests check properties that must hold whatever the reference:

- **EM monotonicity**: in every case, the log-likelihood (the log posterior under MAP) never
  decreases between iterations, with and without SQUAREM.
- **Internal consistency of standard errors**: Oakes' identity and the direct Hessian of the
  marginal log-likelihood agree to rounding error.
- **Invariance**: results do not change when respondents or items are reordered.

## Tolerances

The largest differences allowed by the tests. In practice most cases agree far more closely
(parameters, log-likelihoods and standard errors typically to about $10^{-8}$).

| Quantity | Tolerance |
|---|---|
| Log-likelihood | $10^{-6}$ (relative) |
| Item parameters (slope–intercept scale) | $10^{-4}$ (absolute) |
| Latent mean and variance | $10^{-4}$ (absolute) |
| Standard errors | $10^{-3}$ (relative) |
| EAP scores | $10^{-5}$ (absolute) |
| MAP / ML / WLE scores, and their SEs | $10^{-4}$ (absolute), $10^{-3}$ (relative) |

## What is compared

| Feature | Compared with |
|---|---|
| 2PL, Rasch, 3PL, 4PL, GRM, GPCM, PCM, Nominal | `mirt(..., itemtype = ...)`: estimates, log-likelihood, Oakes SEs |
| Missing responses (MAR) | `mirt()` on data with `NA` |
| Sampling weights | `mirt(..., survey.weights = w)` and `multipleGroup(..., survey.weights = w)`: estimates, log-likelihood, AIC, BIC and Oakes SEs (2PL, graded, two groups with missing responses, fixed item parameter calibration), and `bfactor(..., survey.weights = w)` for a bifactor model |
| Multiple groups with anchor items | `multipleGroup(..., invariance = c(<anchors>, "free_means", "free_var"))` |
| Equality constraints across items | `mirt.model(... CONSTRAIN = ...)` |
| Fixed item parameter calibration | `mirt()` with fixed parameters (`pars = "values"`, `est = FALSE`) |
| Priors and MAP | `mirt.model(... PRIOR = ...)` |
| Display thresholds and delta-method SEs | `coef(m, IRTpars = TRUE, printSE = TRUE)` |
| EAP, MAP, ML, WLE scores | `fscores(m, method = ...)` |
| Confirmatory 2–3 factor models (also with groups) | `mirt()` / `multipleGroup()` with a `mirt.model` and `COV = F1*F2` |
| Empirical histogram | `mirt(..., dentype = "EH")` with fixed items |
| Davidian curves | the exact standardized Davidian likelihood fitted inside mirt (`createGroup()`), and mirt's built-in `dentype = "Davidian-k"` (see below) |
| Normal mixtures | `multipleGroup(..., dentype = "mixture-k")` with all items invariant, plus an independent R implementation |
| S-X2 item fit | `itemfit(m, fit_stats = "S_X2")` |
| Item and test information, expected scores | `iteminfo()`, `testinfo()`, `expected.item()`, `expected.test()` at mirt's estimates (2PL, 3PL, GRM, GPCM, nominal, mixed formats, two groups): to $10^{-9}$ relative |
| Multidimensional information | Information matrices (`iteminfo(..., multidim_matrix = TRUE)`), directional item and test information (`iteminfo()` / `testinfo()` with `degrees =`) and expected scores of two- and three-factor models with every item model and cross-loadings, a bifactor and a two-tier model: to $10^{-10}$ relative |
| Reliability | `marginal_rxx()` (also recomputed with a tight integration tolerance, and for a group with mean and variance other than 0 and 1 on the equivalent standardized model) and `empirical_rxx()`: to $10^{-10}$ |
| Summed-score EAP | `fscores(m, method = "EAPsum")`, the table and the scores of each respondent, including `NA` for incomplete response patterns: to $10^{-9}$ |
| $M_2$ / $M_2^*$ | `M2(m)`, plus an exact R computation where mirt differs (see below) |
| DIF tests | `DIF(mg, scheme = "add" / "drop")` |
| Robust standard errors | `mirt(..., SE.type = "crossprod")` and `SE.type = "sandwich"`: whole covariance matrices at mirt's estimates, and standard errors from a full fit |
| SQUAREM | every comparison above is repeated with `BAEM(squarem=True)` |
| Unconstrained M-step scale | the comparisons of 3PL, 4PL, priors and MAP, missing responses, multiple groups, fixed item parameter calibration and confirmatory 2–3 factor models are repeated with `BAEM(m_step_scale="unconstrained")` at the same tolerances |
| `LogitNormalPrior` | the default `mode_scale="natural"` has no mirt equivalent, so it is checked independently: the density against SciPy (it integrates to 1, and logit $X$ has the stated mean and variance), and the MAP estimate as a stationary point of the log posterior written from the formula, to $10^{-6}$. `mode_scale="logit"` matches mirt's `norm` prior on `g` (log-likelihood and log posterior to $10^{-9}$ relative, $a$, $d$, $g$ to $10^{-6}$, Oakes SEs to $10^{-5}$ relative) and flexMIRT's logit-metric mode computed independently |
| Polychoric correlations | lavaan's `lavCor()` on ordered variables to $10^{-6}$, psych's `polychoric()` to $10^{-3}$ (its optimizer's tolerance) and psych's `cor.smooth()` for smoothing |
| Parallel analysis, MAP, empirical Kaiser criterion | EFAtools' `EKC()` and `PARALLEL()`, psych's `vss()`, EFA.dimensions' `MAP()` and an independent R implementation of the fourth-power MAP, on the same correlation matrices to $10^{-11}$; reference eigenvalues of normal data within Monte Carlo error |
| Q3 and LD $X^2$ / $G^2$ | `residuals(m, type = "Q3" / "LD" / "LDG2")` on the same models fitted by mirt |
| Jackknife slope index | `residuals(m, type = "JSI")` on the same models: the default (refit) to $10^{-5}$. The approximations without refitting are compared with the refit: within about 0.02 from 15 items on, but off by more than 1 on 4–5 item tests |
| Model-based parallel analysis | ltm's `unidimTest()` with the same item parameters: the observed statistic, and the simulated distribution within Monte Carlo error |
| Factor analysis (ULS, ML) and the Hull method | EFAtools' `EFA()` and `HULL()` and psych's `fa()` on the same matrices: implied loadings to $10^{-4}$, fit indices to $2\times10^{-4}$, the same Hull selections |
| Comparison data | EFAtools' `CD()`: the same population from the same random draws, the same rank-sum tests and the same suggested numbers |
| DIMTEST | No implementation outside the stand-alone DIMPACK exists, so it is checked by simulation: type I error 0.5–1% at the 5% level (conservative) and power 99–100% against two dimensions correlated 0.7–0.8 |
| DETECT (confirmatory, exploratory) | sirt's `conf.detect(use_sum_score = TRUE, smooth = FALSE)` and `expl.detect(use_sum_score = TRUE)` with the same split: conditional covariances, indices, partitions, to $10^{-12}$ |
| Observed-score DIF (Mantel–Haenszel, Mantel, generalized MH, logistic regression) | difR's `difMH()`, `difGMH()` and `difLogistic()` (statistics, odds ratios and their variances, $\Delta R^2$, coefficients, purification paths and flags) and R's `mantelhaen.test()`: to $10^{-10}$, logistic regression to $10^{-8}$ (standard errors and Wald statistics $10^{-6}$); Mantel's statistic and the cumulative odds ratio against `difMantel.poly()` / `liu_agresti_ccor()` and a direct R implementation |
| Scale linking (Haebara, Stocking–Lord; unidimensional and confirmatory multidimensional models) | independent R implementations of the criteria using mirt's response functions (mirt itself has no characteristic-curve linking) |
| Classical item analysis (scoring with a key, alpha, item–total and item–rest correlations, point-biserial / biserial / polyserial, omega total, option statistics) | psych's `alpha()`, `score.multiple.choice()`, `polyserial()`, `biserial()` and `omega()`, ltm's `cronbach.alpha()` and `biserial.cor()`, polycor's `polyserial()` and base R: closed forms to $10^{-12}$, omega to $10^{-6}$ |

Polychoric correlations, parallel analysis, MAP, the empirical Kaiser criterion, factor analysis,
the Hull method, comparison data, DETECT, DIMTEST, classical item analysis and observed-score DIF are
functions of the companion package flexpyctt (NumPy / SciPy, no JAX). flexpyirt's own `polychoric()` is a JAX implementation of the
same estimator: it passes the same checks, and the two agree to $10^{-9}$ on every check case.

## Validation cases

Every case below is a test in the repository. Its reference values were generated in R (V30: by a
simulation study) and stored with the source, and the test suite compares against them. Fitted
models are compared with mirt within the tolerances above; quantities with tolerances of their own
are listed in the table after this one. The
[validation plan](https://github.com/takuizum/flexpyirt/blob/main/docs/dev/VALIDATION.md) has the
exact R calls, settings and the largest differences measured in each case. V26 has a second part
added in M7.

| ID | Feature | Models and data | Compared with |
|---|---|---|---|
| V1 | 2PL | `LSAT7`, single group | `mirt(..., "2PL")`, Oakes SEs |
| V2 | Rasch | single group, latent variance estimated | `mirt(..., "Rasch")` |
| V3 | GRM | `Science` | `mirt(..., "graded")` |
| V4 | Multiple groups | 2PL, two groups, partial anchors, focal mean and variance free | `multipleGroup(..., invariance = c(<anchors>, "free_means", "free_var"))` |
| V5 | FIPC | fixed items, latent distribution estimated | `mirt()` with `pars = "values"`, `est = FALSE` |
| V6 | Empirical histogram | under FIPC, one and two groups | `mirt(..., dentype = "EH")`, `multipleGroup(...)` |
| V7 | Priors and MAP | 3PL with priors | `mirt.model(... PRIOR = ...)` |
| V8 | GPCM, PCM | `Science` and simulated | `mirt(..., "gpcm")`, `mirt(..., "Rasch")` |
| V9 | Missing responses | MAR data | `mirt()` on data with `NA` |
| V10 | Standard errors | Oakes' identity (robust SEs are V35) | `mirt(..., SE.type = "Oakes")` |
| V11 | 3PL, 4PL | simulated, single group | `mirt(..., "3PL" / "4PL")` |
| V12 | IRT thresholds, delta-method SEs | 2PL, Rasch, GRM, GPCM, PCM, two-group 2PL | `coef(m, IRTpars = TRUE, printSE = TRUE)` |
| V13 | MAP, ML, WLE scores | 2PL, GRM, GPCM, 3PL, two groups, missing responses | `fscores(m, method = ...)` |
| V14 | Confirmatory 2–3 factor models | 2PL simple structure and cross-loading, GRM, three-factor 2PL | `mirt()` with `mirt.model(... COV = ...)` |
| V15 | S-X2 item fit | GRM, GPCM, PCM, 3PL, FIPC; minimum cell 1 and 3; per group for two groups | `itemfit(m, fit_stats = "S_X2")` |
| V16 | Equality constraints across items | common slope; 2PL + GRM with cross-model constraints; two groups | `mirt.model(... CONSTRAIN = ...)` |
| V17 | DIF tests | V4 data, `add` and `drop` schemes | `DIF(mg, scheme = ...)` |
| V18 | $M_2$ / $M_2^*$ | 2PL, 3PL, Rasch with free variance, GRM, GPCM | `M2(m)`, plus an exact R computation |
| V19 | Davidian curves | 2PL, GRM, FIPC 2PL with free mean and variance | `mirt(..., dentype = "Davidian-k")` and the exact likelihood inside mirt |
| V20 | Scale linking | 2PL, GRM, GPCM, 3PL, mixed 2PL + GPCM; Haebara and Stocking–Lord, both symmetric forms; SEs | separate `mirt()` calibrations + independent R criteria |
| V21 | Multiple-group confirmatory models | two factors, anchors on both, focal means and variances free | `multipleGroup()` with `COV = F1*F2` |
| V22 | Normal mixtures | 2PL and GRM with two components, GPCM with three | `multipleGroup(..., dentype = "mixture-k")` with all items invariant, plus an independent R implementation |
| V23 | Nominal model | `Science`, unordered scoring, mixed with 2PL, two factors | `mirt(..., "nominal")`, `itemfit()` |
| V24 | Polychoric correlations | tetrachoric, 2–5 categories, 10% MCAR, empty cells, `LSAT7`, `Science`; smoothing | lavaan `lavCor()`, psych `polychoric()`, `cor.smooth()` |
| V25 | Number of dimensions | EKC, MAP, eigenvalues on the V24 matrices; parallel-analysis references | EFAtools `EKC()`, `PARALLEL()`; psych `vss()`; EFA.dimensions `MAP()` |
| V26 | Local dependence, model-based parallel analysis | `LSAT7` 2PL, `Science` GRM, simulated 2PL with a dependent pair and MCAR | `residuals(m, type = "LD" / "LDG2" / "Q3")`; ltm `unidimTest()` |
| V26 (M7) | Local dependence beyond one group | two groups; EH, Davidian and mixture latent distributions; two and three factors | `residuals()` on the corresponding mirt fits |
| V27 | DETECT | confirmatory (two and one dimensions, `Science`) and exploratory | sirt `conf.detect()`, `expl.detect()` |
| V28 | EFA, Hull method | ULS and ML, every number of factors; CAF, CFI, RMSEA | EFAtools `EFA()`, `HULL()`; psych `fa()` |
| V29 | Comparison data | population generator, rank-sum tests, three data sets | EFAtools `CD()` and its internal steps |
| V30 | DIMTEST | 2PL, $N$ 1000–2000, $J$ 20–30, 200 replications per condition | simulation study (no reference implementation outside DIMPACK) |
| V31 | Jackknife slope index | `LSAT7` 2PL, `Science` GRM, simulated 2PL and GPCM | `residuals(m, type = "JSI")` |
| V32 | Bifactor models | 2PL with three specific factors and general-only items; GRM with three testlets | `bfactor()`, `fscores()` |
| V33 | Two-tier models | 2PL with two correlated primary factors; GRM with a cross-loading item | `bfactor(..., model2 = )` |
| V34 | Sampling weights | 2PL, GRM, two groups with missing responses, FIPC, bifactor 2PL | `mirt()`, `multipleGroup()`, `bfactor()` with `survey.weights` |
| V35 | Robust standard errors | misspecified 2PL, mixed formats, MAP, two groups, FIPC, two and three factors, EH | `mirt(..., SE.type = "crossprod" / "sandwich")` |
| V36 | Summed-score distribution | Poisson-binomial with $J$ 10–300; GRM and mixed formats | poibin `ppoibin()`, `dpoibin()`; `dpois()`; `fscores(method = "EAPsum")` |
| V37 | Information, expected score, reliability, summed-score EAP | 2PL (also with missing), 3PL, GRM, GPCM, nominal, mixed formats, two groups | `iteminfo()`, `testinfo()`, `expected.item()`, `marginal_rxx()`, `empirical_rxx()`, `fscores(method = "EAPsum")` |
| V38 | Local dependence for bifactor and two-tier models | bifactor 2PL with MCAR, bifactor GRM, two-tier 2PL | `residuals()` on `bfactor()` fits |
| V39 | $M_2$ and S-X2 for confirmatory models | the four V14 models | `M2()`, `itemfit()`, plus an exact R $M_2^*$ |
| V40 | Linking: 4PL, nominal, multidimensional | 4PL, nominal, mixed 2PL + nominal; two- and three-factor models with diagonal, orthogonal or oblique $\mathbf A$ | separate `mirt()` calibrations + independent R criteria |

Tolerances of the cases that are not plain comparisons of fitted models with mirt:

| Quantity | Tolerance |
|---|---|
| Scale linking $(A,B)$ / criterion, on identical item estimates | $10^{-9}$ / $10^{-9}$ (relative) |
| Scale linking SEs, on identical estimates and covariance | $10^{-5}$ (relative) |
| Scale linking end to end (calibration + linking): $(A,B)$ / criterion / SE | $10^{-6}$ / $10^{-5}$ (relative) / $10^{-3}$ (relative) |
| Polychoric correlations vs lavaan / psych; thresholds; smoothing | $10^{-6}$ / $10^{-3}$; $10^{-12}$; $10^{-12}$ |
| EKC, MAP, observed eigenvalues; normal reference mean / 95th percentile | $10^{-11}$; $\lvert z\rvert\le4.5$ / $6\times10^{-3}$ |
| LD $X^2$ / $G^2$; signed $V$, Q3; model-based PA observed / simulated mean / 95th percentile | $10^{-5}$ (relative); $10^{-6}$; $2\times10^{-4}$ / $\lvert z\rvert\le4.5$ / 0.02 |
| DETECT conditional covariances, indices, Ward heights | $10^{-12}$ |
| EFA implied $\Lambda\Lambda^\top$ / communalities; CAF, CFI, RMSEA; $\chi^2$; Hull fit / $st$ | $10^{-4}$; $2\times10^{-4}$; $10^{-4}$ (relative); $10^{-5}$ / $10^{-2}$ (relative) |
| Comparison data: PAF; population from the same draws; rank-sum $p$; mean RMSE | $10^{-12}$; exact; $10^{-10}$ (relative); 15% (relative) for the retained $k$, 60% for the next |
| DIMTEST (small rerun of the simulation study): rejections at .05 under the null / power | at most 3 of 20 / at least 8 of 10 |
| Cross-product / sandwich covariance at mirt's estimates (entries relative to $\sqrt{v_{ii}v_{jj}}$); parameters / SEs end to end | $10^{-9}$ / $10^{-6}$; $10^{-6}$ / $10^{-5}$ (relative) |

## Known differences from mirt

These differences are intentional: flexpyirt maximizes exactly the quantity it reports. They are
small or absent in typical use, but you may see them when comparing the two programs.

- **Latent mean and variance on a finite grid.** mirt updates a normal latent distribution with
  the mean and variance of the expected counts; flexpyirt maximizes the discretized likelihood
  exactly ([Theory, section 4](theory.md#4-the-bockaitkin-em-algorithm)). When the grid cuts off
  the distribution (a group with a large variance or a mean far from 0), the estimates differ
  slightly (for example a Rasch variance by about $10^{-4}$), and flexpyirt's log-likelihood is the
  higher of the two. Widening the grid on both sides
  (`Quadrature.equispaced(81, bounds=(-8, 8))` and mirt's `technical = list(theta_lim = c(-8, 8))`)
  makes them agree.
- **Reported log-likelihood.** mirt reports the log-likelihood from its last E-step, which need not
  equal the value at the final estimates. flexpyirt reports the value at the estimates.
- **3PL / 4PL asymptotes.** mirt estimates $g$ and $u$ on the logit scale and reports
  $\mathrm{logit}(g)$ with logit-scale SEs; flexpyirt estimates and reports $g$ and $u$ on their
  natural scale. The estimates agree after transformation ($g=\sigma(\mathrm{logit}\,g)$,
  $SE_g=g(1-g)\,SE_{\mathrm{logit}\,g}$).
- **Empirical histogram.** mirt re-estimates the histogram without standardizing it, so with all
  items free the scale is not identified and EM drifts. flexpyirt constrains the reference group's
  histogram to mean 0 and variance 1 within the M-step. The two agree when fixed items set the
  scale (FIPC).
- **Davidian curves.** mirt's `dentype = "Davidian-k"` standardizes the expected counts by
  interpolation at every cycle, which breaks EM monotonicity and can end at a lower log-likelihood.
  flexpyirt maximizes the exact discretized likelihood, so its estimates differ from mirt's
  built-in Davidian fits, with a higher log-likelihood in the tested cases. mirt's built-in
  Davidian curves also offer no standard errors and no free latent mean or variance.
- **Normal mixtures.** mirt's `dentype = "mixture-k"` estimates class-specific item parameters by
  default (a mixture IRT model). flexpyirt's `NormalMixture` has one common set of items, which
  corresponds to mirt with every item constrained equal across classes, up to the scale of
  $\theta$. mirt's Oakes SEs for mixtures are not observed-information SEs; flexpyirt's are.
- **Nominal model.** mirt's default starting values can stop at a saddle point (e.g. on
  `Science`); flexpyirt's default starts reached the best solution in every tested case. For ML
  scoring, mirt returns $\pm\infty$ for every all-lowest / all-highest *category* pattern, whereas
  flexpyirt decides by the category slopes $a\,\mathrm{ak}_c$, so a pattern such as all-0 with
  $\mathrm{ak}_1<0$ gets a finite estimate.
- **$M_2$.** flexpyirt includes free latent parameters in the derivative matrix $\Delta_2$ (mirt
  leaves them out, for example the Rasch variance or the factor correlations of a confirmatory
  multidimensional model, where flexpyirt's $M_2$ is smaller by up to 0.07% in the tested cases) and uses exact derivatives for 3PL/4PL and GPCM
  items, where mirt 1.46's are inaccurate. For these models flexpyirt is compared with an exact R
  computation, and its $M_2$ differs from mirt's (by 5–11% in the tested 3PL and GPCM cases). TLI and
  CFI are reported for dichotomous models only, because the independence null model is
  rank-deficient for polytomous items.
- **Multidimensional ML scoring.** When one dimension of an ML estimate diverges, flexpyirt reports
  $\pm\infty$ with SE NaN, whereas mirt can stop near its bound ($|\theta|\approx20$) with SE 0.

The complete validation plan, with every case, mirt call and measured difference, is in the
[validation plan](https://github.com/takuizum/flexpyirt/blob/main/docs/dev/VALIDATION.md) on GitHub.
## Known differences from other packages

- **Fourth-power MAP in EFA.dimensions.** EFA.dimensions (0.1.9.3) computes its "4th power" MAP statistic with a matrix power, $(\operatorname{tr}(R^{*4})-p)/(p(p-1))$, not by raising each partial correlation to the fourth power as Velicer, Eaton & Fava (2000) do (its values can exceed 1). flexpyirt uses the elementwise power; the squared version agrees with EFA.dimensions and psych.
- **Sum-score DETECT in sirt's `conf.detect()`.** With `use_sum_score = TRUE` and the default
  `smooth = TRUE`, sirt 4.2 standardizes the total score before subtracting the two item scores,
  so the "rest score" groups are not rest-score groups and the indices differ (DETECT 6.73
  instead of 1.60 on one test case). With `smooth = FALSE`, and in `expl.detect()`, sirt and
  flexpyirt agree.
- **Missing data in psych's `alpha()` and ltm's `cronbach.alpha()`.** With missing responses
  psych's `raw.r` correlates each item with the sum of the observed scores (a missing response
  counts as 0), while its alpha and `r.drop` use pairwise covariances; flexpyctt's `item_total`
  is the correlation implied by those pairwise covariances, so it differs from `raw.r` (by up to
  0.014 on a test case with 6% missing). ltm's raw alpha with `na.rm = TRUE` takes the total
  variance from complete rows and the item variances from all responses; flexpyctt's
  `missing="listwise"` uses complete rows for both. On complete data all of them agree.
- **psych's `biserial()`** divides by the $n-1$ standard deviation of the total score, so it is
  smaller than the two-step biserial correlation (polycor's) by the factor $\sqrt{(n-1)/n}$.
- **Standard error of the Liu–Agresti odds ratio in difR.** difR 6.1's `liu_agresti_ccor()`
  (used by `difMantel.poly()`) reports as `SE_log_Psi` a delta-method value whose derivative with
  respect to the focal counts has a sign error and which is divided by the wrong sum, so it is
  neither the SE of $\ln\hat\psi$ nor of $\hat\psi$. flexpyctt's `mantel()` uses Liu & Agresti's
  (1996) own variance estimator (their eq. 3), which stays valid when the matching score makes
  many small strata. It reproduces the paper's worked example (SE .571), matches the spread of
  $\ln\hat\psi$ in simulations with four to ten respondents per stratum, and for 0 / 1 items with
  large strata agrees with the Mantel–Haenszel SE of Phillips & Holland. The odds ratio itself
  agrees (as its reciprocal: difR's $\hat\psi$ above 1 favours the focal group).
- **Continuity correction in `mantelhaen.test()`.** R's `mantelhaen.test()` omits the
  correction of $\tfrac12$ when $\lvert\sum A-\sum E(A)\rvert<\tfrac12$; difR, Holland & Thayer
  (1988) and flexpyctt always apply it, so the corrected $\chi^2_{MH}$ of such an item (one with no
  DIF) differs from `mantelhaen.test()`.
- **Purification in difR's `difMantel.poly()`.** It removes a flagged item from its own matching
  score as well; flexpyctt keeps the studied item in its matching score for every method, as
  difR's other functions do.

## References

- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel–Haenszel procedure. In H. Wainer & H. I. Braun (Eds.), *Test Validity* (pp. 129–145). Erlbaum.
- Liu, I.-M., & Agresti, A. (1996). Mantel–Haenszel-type inference for cumulative odds ratios with a stratified ordinal response. *Biometrics*, 52(4), 1223–1234.
- Phillips, A., & Holland, P. W. (1987). Estimators of the variance of the Mantel–Haenszel log-odds-ratio estimate. *Biometrics*, 43(2), 425–431.
