# flexpyequate

English | [日本語](https://github.com/takuizum/flexpyirt/blob/main/packages/flexpyequate/README.ja.md)

Equating of test score distributions in Python. Given the score distributions of two test forms,
flexpyequate finds the function that puts scores of form X on the scale of form Y.

- **Checked against R**: every method is compared numerically with R `equate`, `kequate` and
  `SNSequate`, and with independent implementations of the published formulas.
- **Light**: NumPy, SciPy and pandas only.
- **Scope**: observed-score equating of score distributions. Equating that needs item parameters
  (IRT true-score and observed-score equating, scale linking) is in
  [flexpyirt](https://github.com/takuizum/flexpyirt).

> Under development (0.x). The public API may change before 1.0.0.

## Features

| Feature | Details | Checked against |
|---|---|---|
| Score distributions | `ScoreDistribution` and `BivariateScoreDistribution` from raw scores or frequency tables; equally spaced scores with any increment; moments; percentile ranks and percentiles (Kolen & Brennan, 2014) | R `equate` |
| Designs | Equivalent groups (two distributions), single group (one bivariate distribution) and non-equivalent groups with an anchor test (two bivariate distributions) | R `equate` |
| Mean and linear equating | `equate(x, y, method="mean")`, `method="linear"` (population or sample standard deviations) | R `equate` |
| Equipercentile equating | `equate(x, y, method="equipercentile")` with the percentile-rank definition of Kolen & Brennan (2014), including distributions with zero frequencies | R `equate`; independent R implementation of the textbook formulas |
| Linear equating with an anchor test | `equate(xa, ya, method=...)` with `"tucker"`, `"levine_observed"`, `"levine_true"` (internal or external anchor), `"braun_holland"` and `"chained_linear"`; synthetic population weights | R `equate`; independent R implementation of the textbook formulas |
| Equipercentile equating with an anchor test | `equate(xa, ya, method="frequency_estimation")` (post-stratification, synthetic population weights) and `method="chained_equipercentile"`; presmoothed joint distributions can be passed | R `equate`; independent R implementation of the textbook formulas |
| Log-linear presmoothing | `presmooth()`: polynomial log-linear models for one score or for the joint distribution of two scores (with cross terms); likelihood-ratio, Pearson and Freeman–Tukey chi-squares, AIC | R `glm`, `equate` |
| Kernel equating | `kernel_equate()` for equivalent groups, single group, counterbalanced and anchor-test designs (chained and post-stratification): Gaussian kernel, bandwidths chosen by the penalty of von Davier, Holland & Thayer (2004) or given, standard error of equating (with or without presmoothing), linear version, standard error of the difference between two equating functions (`seed()`), percent relative error | R `kequate`, `SNSequate`; independent R implementation |
| Other kernels and bandwidth rules | `kernel_equate(..., kernel="logistic")` and `kernel="uniform"` (Lee & von Davier, 2011) in every design, with standard errors; `bandwidth_method="rule_of_thumb"`, `"double_smoothing"` and `"cross_validation"` (the leave-one-out criterion of `kequate`) for the Gaussian kernel | R `kequate`, `SNSequate`; independent R implementation |
| Standard errors of equating | `equate(..., standard_error="analytic")`: delta method for mean, linear, Tucker, Levine, chained linear and Braun–Holland equating (distribution-free, or `"analytic_normal"`), Lord's (1982) formula for equipercentile equating of equivalent groups; `standard_error="bootstrap"` with `n_replications` and `rng` for every method | R `equate` (analytic); independent R implementation; bootstrap: statistical agreement with `equate::bootstrap` and with the analytic values |

## Installation

Python 3.11 or later.

```bash
pip install flexpyequate  # or: uv add flexpyequate
```

## Example

```python
import flexpyequate as fe

# Frequencies of the scores 0..4 on two forms taken by equivalent groups.
x = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[4, 6, 6, 3, 1])
y = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[2, 4, 6, 4, 4])

result = fe.equate(x, y, method="equipercentile")
print(result.table)
#    score   equated
# 0    0.0  0.500000
# 1    1.0  1.666667
# 2    2.0  2.750000
# 3    3.0  3.875000
# 4    4.0  4.375000

result.transform([1.5, 2.5])  # equate scores between the possible scores
result.moments  # mean, sd, skewness, kurtosis of X, Y and the equated scores

linear = fe.equate(x, y, method="linear")
linear.slope, linear.intercept
```

From one score per examinee, and for a single group taking both forms:

```python
x = fe.ScoreDistribution.from_scores(x_scores, score_range=(0, 40))
xy = fe.BivariateScoreDistribution.from_scores(x_scores, y_scores, x_range=(0, 40), y_range=(0, 40))
fe.equate(xy, method="equipercentile")  # equates the two margins
```

Non-equivalent groups with an anchor test (each group takes one form and the anchor):

```python
xa = fe.BivariateScoreDistribution.from_scores(x_total, x_anchor, x_range=(0, 36), y_range=(0, 12))
ya = fe.BivariateScoreDistribution.from_scores(y_total, y_anchor, x_range=(0, 36), y_range=(0, 12))

tucker = fe.equate(xa, ya, method="tucker")  # weights proportional to the sample sizes
tucker.slope, tucker.intercept
tucker.synthetic_moments  # mean and sd of X and Y in the synthetic population

fe.equate(xa, ya, method="levine_observed", anchor="internal", synthetic_weight=1.0)
fe.equate(xa, ya, method="chained_linear")

fe.equate(xa, ya, method="frequency_estimation")
fe.equate(xa, ya, method="chained_equipercentile")

# With presmoothed joint distributions:
xa_smooth = fe.presmooth(xa, degree=(6, 6), cross_terms=[(1, 1)]).distribution
ya_smooth = fe.presmooth(ya, degree=(6, 6), cross_terms=[(1, 1)]).distribution
fe.equate(xa_smooth, ya_smooth, method="frequency_estimation")
```

Presmoothing and kernel equating:

```python
smooth_x = fe.presmooth(x, degree=6)  # log-linear model with the powers x^1..x^6
smooth_y = fe.presmooth(y, degree=6)
smooth_x.distribution  # the smoothed ScoreDistribution
smooth_x.fit_statistics  # likelihood ratio, Pearson, Freeman-Tukey, AIC, ...

fe.equate(smooth_x.distribution, smooth_y.distribution, method="equipercentile")

kernel = fe.kernel_equate(smooth_x, smooth_y)  # Gaussian kernel, penalty bandwidths
kernel.table  # score, equated, see (standard error of equating)
kernel.bandwidth_x, kernel.bandwidth_y
kernel.percent_relative_error  # first ten moments

# Other designs take presmoothed joint distributions.
xa_fit = fe.presmooth(xa, degree=(4, 3), cross_terms=[(1, 1)])
ya_fit = fe.presmooth(ya, degree=(4, 3), cross_terms=[(1, 1)])
chained = fe.kernel_equate(xa_fit, ya_fit, design="chained")
post = fe.kernel_equate(xa_fit, ya_fit, design="post_stratification")
linear = fe.kernel_equate(xa_fit, ya_fit, design="post_stratification", linear=True)
post.seed(linear)  # difference between kernel and linear equating, with its standard error
chained.seed(post)
fe.kernel_equate(fe.presmooth(xy, degree=(4, 4)))  # single group
```

## Notes on agreement with R

- R `equate` divides variances by N − 1; flexpyequate uses the population form of Kolen & Brennan
  by default. Pass `ddof=1` to reproduce `equate`'s linear equating exactly.
- R `equate`'s `type = "mean"` multiplies scores by the ratio of the two score ranges; flexpyequate's
  mean equating always has slope 1 (the two agree for forms with the same score range).
- Where a form-X score has percentile rank exactly 0 or 100, flexpyequate returns the mean of the
  upper and lower percentiles of form Y, as defined by Kolen & Brennan; `equate` returns the end of
  the form-Y range. The two differ only when nobody obtained the lowest (or highest) form-Y scores.

- For the Braun–Holland and frequency-estimation methods, R `equate` mixes the two groups as counts, which changes the
  weight of the first group from w to wN₁ / (wN₁ + (1 − w)N₂) when the sample sizes differ.
  flexpyequate uses w itself, as in Kolen & Brennan; pass that effective weight as
  `synthetic_weight` to reproduce `equate`.
- R `kequate` multiplies the variance used in the continuization by N / (N − 1); flexpyequate
  uses the variance of the score distribution itself, as in von Davier, Holland & Thayer (2004)
  and `SNSequate`. The difference in equated scores shrinks with the sample size (about 1e-3 with
  600 examinees and 1e-5 or less with several thousand in our comparisons).
- In the counterbalanced design R `kequate` has a single weight; its standard errors are
  consistent with its equating function only when that weight is 1/2. flexpyequate has two
  weights (`counterbalance_weights`) and its standard errors hold for any of them.
- `kernel_equate()` selects bandwidths with both penalty terms of the book (`penalty_weight=1`);
  `kequate` uses the first term only by default (`penalty_weight=0` reproduces it).

## References

- Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking* (3rd ed.). Springer.
- von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method of test equating*.
  Springer.
- Holland, P. W., & Thayer, D. T. (2000). Univariate and bivariate loglinear models for discrete
  test score distributions. *Journal of Educational and Behavioral Statistics, 25*, 133–183.
- Albano, A. D. (2016). equate: An R package for observed-score linking and equating. *Journal of
  Statistical Software, 74*(8).
- Andersson, B., Bränberg, K., & Wiberg, M. (2013). Performing the kernel method of test equating
  with the package kequate. *Journal of Statistical Software, 55*(6).
- González, J. (2014). SNSequate: Standard and nonstandard statistical models and methods for test
  equating. *Journal of Statistical Software, 59*(7).

## License

MIT
