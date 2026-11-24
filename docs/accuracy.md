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
| $M_2$ / $M_2^*$ | `M2(m)`, plus an exact R computation where mirt differs (see below) |
| DIF tests | `DIF(mg, scheme = "add" / "drop")` |
| SQUAREM | every comparison above is repeated with `BAEM(squarem=True)` |
| Polychoric correlations | lavaan's `lavCor()` on ordered variables to $10^{-6}$, psych's `polychoric()` to $10^{-3}$ (its optimizer's tolerance) and psych's `cor.smooth()` for smoothing |
| Parallel analysis, MAP, empirical Kaiser criterion | EFAtools' `EKC()` and `PARALLEL()`, psych's `vss()`, EFA.dimensions' `MAP()` and an independent R implementation of the fourth-power MAP, on the same correlation matrices to $10^{-11}$; reference eigenvalues of normal data within Monte Carlo error |
| Q3 and LD $X^2$ / $G^2$ | `residuals(m, type = "Q3" / "LD" / "LDG2")` on the same models fitted by mirt |
| Jackknife slope index | `residuals(m, type = "JSI")` on the same models: the default (refit) to $10^{-5}$. The approximations without refitting are compared with the refit: within about 0.02 from 15 items on, but off by more than 1 on 4–5 item tests |
| Model-based parallel analysis | ltm's `unidimTest()` with the same item parameters: the observed statistic, and the simulated distribution within Monte Carlo error |
| Factor analysis (ULS, ML) and the Hull method | EFAtools' `EFA()` and `HULL()` and psych's `fa()` on the same matrices: implied loadings to $10^{-4}$, fit indices to $2\times10^{-4}$, the same Hull selections |
| Comparison data | EFAtools' `CD()`: the same population from the same random draws, the same rank-sum tests and the same suggested numbers |
| DIMTEST | No implementation outside the stand-alone DIMPACK exists, so it is checked by simulation: type I error 0–2% at the 5% level (conservative) and power 99–100% against two dimensions correlated 0.7–0.8 |
| DETECT (confirmatory, exploratory) | sirt's `conf.detect(use_sum_score = TRUE, smooth = FALSE)` and `expl.detect(use_sum_score = TRUE)` with the same split: conditional covariances, indices, partitions, to $10^{-12}$ |
| Scale linking (Haebara, Stocking–Lord) | independent R implementations of the criteria using mirt's response functions (mirt itself has no characteristic-curve linking) |

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
  leaves them out, for example the Rasch variance) and uses exact derivatives for 3PL/4PL and GPCM
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

## References

- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
