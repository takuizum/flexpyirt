# flexpyirt

English | [日本語](README.ja.md)

Item response theory (IRT) estimation in Python, built on JAX. At its core is marginal maximum
likelihood estimation (MMLE) with the Bock–Aitkin EM algorithm, including multiple-group
calibration and fixed item parameter calibration (FIPC).

- **Verified against R `mirt`**: every supported feature is checked numerically against mirt
  (parameters, log-likelihood and standard errors agree to about 1e-8; see
  [validation](docs/VALIDATION.md)).
- **Automatic differentiation**: M-step gradients and Hessians and the standard errors (Oakes'
  identity) all come from JAX autodiff. A new item model only needs its log-probability for
  estimation, standard errors and scoring to work.
- **ParameterTable**: a table like mirt's `pars = "values"` for editing fixed values, equality
  constraints and starting values.

> Under development (0.x). The public API may change before 1.0.0.

## Features

| Feature | Details | mirt check |
|---|---|---|
| Item models | 2PL, Rasch (latent variance estimated), 3PL, 4PL, GRM (graded response model), GPCM, PCM (latent variance estimated), Nominal (nominal response model) | V1, V2, V3, V8, V11, V23 |
| Priors and MAP | `NormalPrior`, `LogNormal`, `Beta` on item parameters (`Model.prior()`), natural scale, `FitResult.log_posterior` | V7 |
| Multiple groups | Group-specific item parameters, anchor items equal across groups, focal-group mean and variance estimated | V4 |
| Equality across items | Common parameters across items and item models (`Model.equal_across_items()`, e.g. one common slope), combined with across-group equalities | V16 |
| FIPC | Fix some items at item-bank values; estimate the other items and the latent distribution | V5 |
| Missing data | Responses missing at random (MAR) | V9 |
| Standard errors | Oakes' identity (default) or the Hessian of the marginal log-likelihood / log posterior, plus delta-method SEs for display IRT thresholds (`b_se` / `bk_se`) | V7, V10, V12, V14 |
| Scoring | EAP and MAP (1–3 factors), ML, WLE (with standard errors) and plausible values (`PV`, explicit JAX PRNG `key`), using each group's latent distribution as the prior where applicable | V1–V5, V7–V9, V11, V13, V14 |
| Multidimensional models | Confirmatory 2–3 factor MIRT (`Model.with_factors()`) with cross-loadings, correlated or orthogonal MVN factors, product quadrature grids, Oakes/Hessian SEs, and EAP / MAP scoring | V14 |
| Empirical histogram | `EmpiricalHistogram()` latent distributions (Woods, 2007) for skewed or multimodal traits; reference group standardized to mean 0 / variance 1 by an exact constrained M-step, free under FIPC; weights reported with moments up to the 4th | V6 |
| Saving results | `FitResult.to_json()` / `FitResult.from_json()` (strict schema, no pickle) | — |
| EM acceleration | SQUAREM (scheme S3) with a monotonicity safeguard, `BAEM(squarem=True)`; agreement with mirt holds with and without it | V1–V5, V7–V9, V11, V14 |
| Item fit | Orlando–Thissen S-X2 (`FitResult.item_fit()`) with mirt's cell collapsing, df, RMSEA and p-values | V15 |
| DIF | Multiple-group likelihood-ratio DIF tests (`FitResult.dif()`, mirt's `DIF()` schemes `add` / `drop`, AIC/SABIC/HQ/BIC differences, Holm or BH adjusted p-values) | V17 |
| Identification checks | Unidentified specifications raise `IdentificationError` before fitting | — |

See the [roadmap](docs/ROADMAP.md) for what comes next.

## Installation

Python 3.11 or later. flexpyirt is not on PyPI yet; install it from GitHub
([uv](https://docs.astral.sh/uv/) recommended).

```bash
uv add git+https://github.com/takuizum/flexpyirt
```

To pin a release, add its tag: `git+https://github.com/takuizum/flexpyirt@v0.0.1`.

flexpyirt computes in **64-bit floating point**. The JAX setting belongs to you, so the library
does not change it; enable it before fitting (otherwise `fit()` raises `PrecisionError`).

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## Quick start

As an example, simulate 2PL data and fit it. With real data, pass a DataFrame with one column per
item and one row per respondent (values 0/1, missing as NaN).

```python
import numpy as np
import pandas as pd

from flexpyirt import Model, TwoPL, fit

rng = np.random.default_rng(1)
n_persons, n_items = 1000, 10
a = rng.lognormal(0.0, 0.3, n_items)  # slopes
d = rng.normal(0.0, 1.0, n_items)  # intercepts
theta = rng.normal(0.0, 1.0, n_persons)
p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
df = pd.DataFrame(
    (rng.random(p.shape) < p).astype(int),
    columns=[f"q{j:02d}" for j in range(1, n_items + 1)],
)

spec = Model.uniform(TwoPL(), items=df.columns)
res = fit(spec, df)

print(res)  # log-likelihood, number of parameters, convergence
print(res.items.head())  # item, model, a, d, a_se, d_se, b (= -d/a), b_se
print(res.loglik, res.aic, res.bic)
scores = res.score(df)  # EAP (theta) and posterior SD (se) per respondent
```

Parameters use mirt's slope–intercept form $P(y=1\mid\theta)=\sigma(a\theta+d)$; `items` also
shows the difficulty $b=-d/a$ for display.

## Examples

See the [user guide](docs/guide.md) for details and the [API reference](docs/reference.md) for
each function.

**Rasch model** (slopes fixed at 1, latent variance estimated)

```python
from flexpyirt import Rasch

rasch = fit(Model.uniform(Rasch(), items=df.columns), df)
print(rasch.latent)  # group, mean (fixed at 0), var (estimated), mean_se, var_se
```

**Multiple groups (linked by anchor items)**

```python
from flexpyirt import Normal

group = np.repeat(["ref", "focal"], n_persons // 2)
spec2 = (
    Model.uniform(TwoPL(), items=df.columns)
    .with_groups(ref=Normal.standard(), focal=Normal.free())  # the first group is the reference
    .equal_across_groups(items=["q01", "q02", "q03", "q04"])  # anchor items
)
res2 = fit(spec2, df, group=group)
print(res2.latent)  # the focal group's mean and variance are estimated
```

**FIPC (fix items at bank values and calibrate new items)**

```python
bank = pd.DataFrame({"item": ["q01", "q02", "q03"], "a": a[:3], "d": d[:3]})
spec3 = Model.uniform(TwoPL(), items=df.columns).fix_from(bank).with_latent(Normal.free())
res3 = fit(spec3, df)
```

## Performance

Wall-time benchmarks against R `mirt` (default settings, `TOL = 1e-4`, 61 quadrature points on CPU)
are in [benchmarks/](benchmarks/README.md) with committed Apple M5 results in
[benchmarks/results/2026-11-21-apple-m5.md](benchmarks/results/2026-11-21-apple-m5.md).
After the first-call JAX/XLA compilation (a cold EM fit takes ~0.8–1.1 s), warm CPU fits are faster
than `mirt` across all tested cases (`0.12×–0.60×` of `mirt` time), while `mirt` is faster for a
single cold fit without standard errors.

## Documentation

- [User guide](docs/guide.md) — specifying models, multiple groups, FIPC, editing the
  ParameterTable, estimation options
- [API reference](docs/reference.md) — public classes and functions and their arguments
- [Validation](docs/VALIDATION.md) — how results are checked against R mirt, tolerances, known
  differences
- [Design](docs/DESIGN.md) — formulas, algorithms and implementation decisions
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md) — branches, reviews, releases

Every document is also available in Japanese (`*.ja.md`).

## Development

```bash
uv sync --all-extras --dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch flow and releases.

## License

MIT
