# flexpyirt

English | [日本語](index.ja.md)

Item response theory (IRT) estimation in Python, built on JAX: marginal maximum likelihood with
the Bock–Aitkin EM algorithm, multiple-group calibration, fixed item parameter calibration (FIPC),
and more. Every supported feature is checked numerically against R `mirt`
(see [accuracy and validation](accuracy.md)).

> Under development (0.x). The public API may change before 1.0.0.

## Installation

Python 3.12 or later.

```bash
pip install flexpyirt  # or: uv add flexpyirt
```

Installing flexpyirt also installs **flexpyctt**, its companion package for analyses that need no
fitted model (polychoric correlations, number of dimensions, DETECT, DIMTEST). flexpyctt depends
on NumPy, SciPy and pandas only; install it alone with `pip install flexpyctt` where JAX is not
wanted.

**flexpyequate**, a third package from the same repository, equates score distributions (linear,
equipercentile and kernel equating). It is independent of flexpyirt and is not installed with it:
`pip install flexpyequate`. See [Score equating](equating.md).

For polars input, install `flexpyirt[polars]`. For GPUs, install a CUDA build of `jax` first
(see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

flexpyirt computes in **64-bit floating point**. The JAX setting belongs to you, so the library
does not change it; enable it before fitting (otherwise `fit()` raises `PrecisionError`).

```python
import jax

jax.config.update("jax_enable_x64", True)
```

## A first fit

```python
import numpy as np
import pandas as pd

from flexpyirt import Model, TwoPL, fit

rng = np.random.default_rng(1)
a, d = rng.lognormal(0.0, 0.3, 10), rng.normal(0.0, 1.0, 10)
theta = rng.normal(0.0, 1.0, 1000)
p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
df = pd.DataFrame((rng.random(p.shape) < p).astype(int), columns=[f"q{j:02d}" for j in range(10)])

res = fit(Model.uniform(TwoPL(), items=df.columns), df)
print(res.items.head())  # item, model, a, d, a_se, d_se, b (= -d/a), b_se
scores = res.score(df)  # EAP and posterior SD per respondent
```

## Where to go next

- [User guide](guide.md): specifying models, multiple groups, FIPC, scoring, item fit, DIF,
  linking and estimation options.
- [Theory](theory.md): the models and algorithms behind flexpyirt, in brief (IRT models, marginal
  likelihood, the EM algorithm, identification, standard errors, scoring, fit and linking).
- [API reference](reference.md): every public class and function.
- [Accuracy and validation](accuracy.md): how results are checked against R mirt, tolerances and
  known differences from mirt.
- Source code, issues and releases: [GitHub](https://github.com/takuizum/flexpyirt).
