# flexpyequate

English | [日本語](https://github.com/takuizum/flexpyirt/blob/main/packages/flexpyequate/README.ja.md)

Equating of test score distributions in Python. Given the score distributions of two test forms,
flexpyequate finds the function that puts scores of form X on the scale of form Y.

- **Checked against R**: every method is compared numerically with R `equate`, and with
  independent implementations of the published formulas.
- **Light**: NumPy, SciPy and pandas only.
- **Scope**: observed-score equating of score distributions. Equating that needs item parameters
  (IRT true-score and observed-score equating, scale linking) is in
  [flexpyirt](https://github.com/takuizum/flexpyirt).

> Under development (0.x). The public API may change before 1.0.0.

## Features

| Feature | Details | Checked against |
|---|---|---|
| Score distributions | `ScoreDistribution` and `BivariateScoreDistribution` from raw scores or frequency tables; equally spaced scores with any increment; moments; percentile ranks and percentiles (Kolen & Brennan, 2014) | R `equate` |
| Designs | Equivalent groups (two distributions) and single group (one bivariate distribution) | R `equate` |
| Mean and linear equating | `equate(x, y, method="mean")`, `method="linear"` (population or sample standard deviations) | R `equate` |
| Equipercentile equating | `equate(x, y, method="equipercentile")` with the percentile-rank definition of Kolen & Brennan (2014), including distributions with zero frequencies | R `equate`; independent R implementation of the textbook formulas |

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

result.transform([1.5, 2.5])   # equate scores between the possible scores
result.moments                 # mean, sd, skewness, kurtosis of X, Y and the equated scores

linear = fe.equate(x, y, method="linear")
linear.slope, linear.intercept
```

From one score per examinee, and for a single group taking both forms:

```python
x = fe.ScoreDistribution.from_scores(x_scores, score_range=(0, 40))
xy = fe.BivariateScoreDistribution.from_scores(
    x_scores, y_scores, x_range=(0, 40), y_range=(0, 40)
)
fe.equate(xy, method="equipercentile")   # equates the two margins
```

## Notes on agreement with R

- R `equate` divides variances by N − 1; flexpyequate uses the population form of Kolen & Brennan
  by default. Pass `ddof=1` to reproduce `equate`'s linear equating exactly.
- R `equate`'s `type = "mean"` multiplies scores by the ratio of the two score ranges; flexpyequate's
  mean equating always has slope 1 (the two agree for forms with the same score range).
- Where a form-X score has percentile rank exactly 0 or 100, flexpyequate returns the mean of the
  upper and lower percentiles of form Y, as defined by Kolen & Brennan; `equate` returns the end of
  the form-Y range. The two differ only when nobody obtained the lowest (or highest) form-Y scores.

## References

- Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking* (3rd ed.). Springer.
- Albano, A. D. (2016). equate: An R package for observed-score linking and equating. *Journal of
  Statistical Software, 74*(8).

## License

MIT
