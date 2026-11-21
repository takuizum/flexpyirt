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
| Item models | 2PL, Rasch (latent variance estimated), 3PL, 4PL, GRM (graded response model), GPCM, PCM (latent variance estimated) | V1, V2, V3, V8, V11 |
| Multiple groups | Group-specific item parameters, anchor items equal across groups, focal-group mean and variance estimated | V4 |
| FIPC | Fix some items at item-bank values; estimate the other items and the latent distribution | V5 |
| Missing data | Responses missing at random (MAR) | V9 |
| Standard errors | Oakes' identity (default) or the Hessian of the marginal log-likelihood | V10 |
| Scoring | EAP and posterior SD, using each group's latent distribution as the prior | V1–V5, V8, V9, V11 |
| EM acceleration | SQUAREM (scheme S3) with a monotonicity safeguard, `BAEM(squarem=True)`; agreement with mirt holds with and without it | V1–V5, V8, V9, V11 |
| Identification checks | Unidentified specifications raise `IdentificationError` before fitting | — |

Not yet available ([roadmap](docs/ROADMAP.md)): empirical histogram (EH) latent distributions,
priors and MAP estimation (M3); MAP / ML / WLE / plausible-value scoring and multidimensional models
(M4).

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
print(res.items.head())  # item, model, a, d, a_se, d_se, b (= -d/a)
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
