# flexpyctt

English | [日本語](https://github.com/takuizum/flexpyirt/blob/main/packages/flexpyctt/README.ja.md)

Item and test analysis that needs no fitted IRT model: polychoric correlations, tools for
judging the number of dimensions of a test, and DIF screening on observed scores. It depends on NumPy, SciPy and pandas only (no JAX).

flexpyctt is the model-free companion of [flexpyirt](https://pypi.org/project/flexpyirt/), which
estimates IRT models. Installing flexpyirt installs flexpyctt as well; install flexpyctt alone when
you do not fit models.

- **Checked against R**: results are compared numerically with lavaan, psych, EFAtools,
  EFA.dimensions, sirt and difR (see
  [accuracy and validation](https://takuizum.github.io/flexpyirt/accuracy/)).
- **Light**: no compilation step and no 64-bit configuration.

> Under development (0.x). The public API may change before 1.0.0.

**Documentation: <https://takuizum.github.io/flexpyirt/>**

## Features

| Feature | Details | Checked against |
|---|---|---|
| Polychoric correlations | `polychoric()`: polychoric / tetrachoric correlation matrices by two-step maximum likelihood, available-case missing data, lavaan's 2 x 2 zero-cell correction, smoothing of matrices that are not positive definite | lavaan `lavCor`, psych |
| Parallel analysis | `parallel_analysis()`: polychoric or Pearson correlations, PCA or SMC eigenvalues, permutation or normal reference data, quantile or mean rule | EFAtools |
| MAP and EKC | `velicer_map()` (original and revised), `empirical_kaiser()` | psych, EFA.dimensions, EFAtools |
| Factor analysis and Hull | `efa()` (unrotated ULS / ML), `hull()` | psych, EFAtools |
| Comparison data | `comparison_data()` (Ruscio & Roche, 2012) | EFAtools |
| DETECT | `detect()`, `exploratory_detect()` (conditional covariances, DETECT, ASSI, RATIO) | sirt |
| DIMTEST | `dimtest()` with bootstrap bias correction | simulation study |
| DIF on observed scores | `mantel_haenszel()` (χ², common odds ratio, ETS delta and A / B / C classes), `mantel()` (ordered items, Liu–Agresti odds ratio), `generalized_mantel_haenszel()` (several groups, nominal categories), `logistic_dif()` (uniform / non-uniform, Nagelkerke ΔR²); purification of the matching score | difR, `mantelhaen.test` |

## Installation

Python 3.12 or later.

```bash
pip install flexpyctt  # or: uv add flexpyctt
```

## Quick start

Pass a DataFrame (or array) with one column per item and one row per respondent. Responses are
ordered category codes; missing responses are NaN.

```python
import numpy as np
import pandas as pd

from flexpyctt import empirical_kaiser, parallel_analysis, polychoric, velicer_map

# Simulate 10 binary items measuring two correlated factors.
rng = np.random.default_rng(1)
factors = rng.normal(size=(1000, 2))
factors[:, 1] = 0.3 * factors[:, 0] + np.sqrt(1 - 0.3**2) * factors[:, 1]
loadings = np.zeros((10, 2))
loadings[:5, 0] = loadings[5:, 1] = 0.75
latent = factors @ loadings.T + rng.normal(size=(1000, 10)) * np.sqrt(1 - 0.75**2)
df = pd.DataFrame(
    (latent > np.linspace(-0.8, 0.8, 10)).astype(int),
    columns=[f"q{j:02d}" for j in range(1, 11)],
)

pc = polychoric(df)  # tetrachoric here: every item is binary
print(pc.correlation.round(2))

pa = parallel_analysis(df, rng=0)  # rng: a seed or a numpy.random.Generator
print(pa.n_factors, velicer_map(pc).n_factors, empirical_kaiser(pc).n_factors)
```

Functions that draw random numbers take `rng` (an integer seed or a `numpy.random.Generator`), so
results are reproducible.

See the [user guide](https://takuizum.github.io/flexpyirt/guide/) and the
[API reference](https://takuizum.github.io/flexpyirt/reference/) for every function.

## License

MIT
