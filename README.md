# flexpyirt

English | [日本語](https://github.com/takuizum/flexpyirt/blob/main/README.ja.md)

Item response theory (IRT) estimation in Python, built on JAX. At its core is marginal maximum
likelihood estimation (MMLE) with the Bock–Aitkin EM algorithm, including multiple-group
calibration and fixed item parameter calibration (FIPC).

- **Verified against R `mirt`**: every supported feature is checked numerically against mirt
  (parameters, log-likelihood and standard errors agree to about 1e-8; see
  [accuracy and validation](https://takuizum.github.io/flexpyirt/accuracy/)).
- **Exact derivatives**: gradients, Hessians and standard errors (Oakes' identity) come from JAX
  automatic differentiation rather than numerical approximation. Estimation is compiled with JAX
  and also runs on GPUs.
- **ParameterTable**: a table like mirt's `pars = "values"` for editing fixed values, equality
  constraints and starting values.

> Under development (0.x). The public API may change before 1.0.0.

**Documentation: <https://takuizum.github.io/flexpyirt/>**

## Features

| Feature | Details | Checked against mirt |
|---|---|---|
| Item models | 2PL, Rasch (latent variance estimated), 3PL, 4PL, GRM (graded response model), GPCM, PCM (latent variance estimated), Nominal (nominal response model; checked against mirt for one and two factors) | ✓ |
| Priors and MAP | `NormalPrior`, `LogNormalPrior`, `BetaPrior` on item parameters (`Model.prior()`), natural scale, `FitResult.log_posterior` | ✓ |
| Multiple groups | Group-specific item parameters, anchor items equal across groups, focal-group mean and variance estimated | ✓ |
| Equality across items | Common parameters across items and item models (`Model.equal_across_items()`, e.g. one common slope), combined with across-group equalities | ✓ |
| FIPC | Fix some items at item-bank values; estimate the other items and the latent distribution | ✓ |
| Missing data | Responses missing at random (MAR) | ✓ |
| Sampling weights | Survey weights `fit(..., weights=...)`: weighted marginal likelihood for every model, multiple groups and FIPC (mirt's `survey.weights`). Oakes / Hessian SEs equal mirt's but are valid for frequency weights only; item fit, M2, local dependence and DIF are not available for weighted fits | ✓ |
| Standard errors | Oakes' identity (default) or the Hessian of the marginal log-likelihood / log posterior, plus delta-method SEs for display IRT thresholds (`b_se` / `bk_se`); robust sandwich and cross-product SEs (`se="sandwich"` / `"crossprod"`; not with Davidian curves, normal mixtures, bifactor or two-tier models) | ✓ |
| Scoring | EAP, MAP, ML and WLE (1–3 factors, with standard errors) and plausible values (`PV`, 1–3 factors, explicit JAX PRNG `key`), using each group's latent distribution as the prior where applicable | ✓ |
| Multidimensional models | Confirmatory 2–3 factor MIRT (`Model.with_factors()`, also with multiple groups) with cross-loadings, correlated or orthogonal MVN factors, product quadrature grids, Oakes/Hessian SEs, and EAP / MAP / ML / WLE scoring; bifactor models with any number of specific factors by dimension reduction (`Model.with_bifactor()`, single group, EAP scoring); two-tier models with one or two correlated primary factors (`Model.with_two_tier()`) | ✓ |
| Empirical histogram | `EmpiricalHistogram()` latent distributions (Woods, 2007) for skewed or multimodal traits; reference group standardized to mean 0 / variance 1 by an exact constrained M-step, free under FIPC; weights reported with moments up to the 4th | ✓ |
| Davidian curves | `DavidianCurve(degree)` seminonparametric latent distributions (Zhang & Davidian, 2001; Woods & Lin, 2009) for single-group unidimensional models; analytically standardized to exact continuous `mean`/`var` (optional `free_mean`/`free_var` for Rasch, PCM, FIPC) with joint Oakes/Hessian SEs and EAP/PV scoring | ✓ |
| Normal mixtures | `NormalMixture(n_components)` latent densities (Mislevy, 1984) with one common set of items for single-group unidimensional models; mirt's `dentype = "mixture-k"` class parameterization analytically standardized to exact continuous `mean`/`var`, standardized components (`pi_c`, `mean_c`, `var_c`) with delta-method SEs, joint Oakes/Hessian SEs, EAP/PV scoring, degenerate-solution warnings | ✓ |
| Saving results | `FitResult.to_json()` / `FitResult.from_json()` (strict schema, no pickle) | — |
| EM acceleration | SQUAREM (scheme S3) with a monotonicity safeguard, `BAEM(squarem=True)`; agreement with mirt holds with and without it | ✓ |
| Item fit | Orlando–Thissen S-X2 (`FitResult.item_fit()`, per group for multiple groups; also confirmatory 2–3 factor models, single group) with mirt's cell collapsing, df, RMSEA and p-values | ✓ |
| Summed scores | Distribution of the summed score given the item category probabilities (`summed_score_distribution()`): Lord–Wingersky recursion (Poisson-binomial distribution for dichotomous items), also in log space, an FFT method, and normal / refined normal / Poisson approximations; works with `jit`, `vmap` and autodiff | ✓ |
| Information and reliability | Item and test information for every item model by automatic differentiation (`item_information()`, `test_information()` with the conditional standard error), expected scores and the test characteristic curve (`expected_score()`), marginal and empirical reliability (`reliability()`), the summed-score EAP table (`summed_score_table()`) and `score(method="EAPsum")`; unidimensional models, single and multiple groups | ✓ |
| Limited-information fit | Maydeu-Olivares & Joe M2 / M2* (`FitResult.m2()`; unidimensional and confirmatory 2–3 factor models, single group) with df, p-value, RMSEA + CI, SRMSR and dichotomous TLI / CFI | ✓ |
| DIF | Multiple-group likelihood-ratio DIF tests (`FitResult.dif()`, mirt's `DIF()` schemes `add` / `drop`, AIC/SABIC/HQ/BIC differences, Holm or BH adjusted p-values). Observed-score DIF without a model: Mantel–Haenszel with ETS delta and A / B / C classes (`mantel_haenszel()`), Mantel's test for ordered items (`mantel()`), generalized MH for several groups or nominal categories (`generalized_mantel_haenszel()`) and logistic regression with $\Delta R^2$ (`logistic_dif()`), all with purification; checked against difR. Provided by flexpyctt | ✓ |
| Scale linking | Differentiable characteristic-curve linking (`link()`, `LinkResult`: Haebara and Stocking–Lord, non-symmetric and symmetric, Ogasawara delta-method SEs, transformed items and $\theta$ scores; checked against independent R implementations for 2PL, 3PL, 4PL, GRM, GPCM, nominal and mixed-format common items, and for confirmatory two- and three-factor models with a diagonal, orthogonal or oblique transformation $\boldsymbol\theta^* = \mathbf A\boldsymbol\theta + \boldsymbol\beta$) | ✓ |
| Polychoric correlations | Polychoric / tetrachoric correlation matrices (`polychoric()`: two-step ML, available-case missing data, lavaan's 2 x 2 zero-cell correction, smoothing of non-positive-definite matrices; checked against lavaan's `lavCor`). Provided by the companion package [flexpyctt](https://pypi.org/project/flexpyctt/) (no JAX), which is installed with flexpyirt | ✓ |
| Number of dimensions | Parallel analysis (`parallel_analysis()`: polychoric or Pearson, PCA or SMC eigenvalues, permutation or normal reference data, quantile or mean rule), Velicer's MAP (`velicer_map()`, original and revised) the empirical Kaiser criterion (`empirical_kaiser()`), unrotated ULS / ML factor analysis (`efa()`), the Hull method (`hull()`) and comparison data (`comparison_data()`); checked against EFAtools, psych and EFA.dimensions. Provided by flexpyctt | ✓ |
| Local dependence and unidimensionality | Q3, Q3\* and Chen & Thissen LD $X^2$ / $G^2$ (`FitResult.local_dependence()`, as mirt's `residuals()`), the jackknife slope index (`FitResult.jsi()`, as mirt, with faster approximations), model-based parallel analysis (`model_parallel_analysis()`, as ltm's `unidimTest`), confirmatory and exploratory DETECT (`detect()`, `exploratory_detect()`, as sirt), DIMTEST with bootstrap bias correction (`dimtest()`, checked by simulation), and a `dimensionality()` report that runs several methods with separate consensus lines. DETECT and DIMTEST are provided by flexpyctt | ✓ |
| Identification checks | Unidentified specifications raise `IdentificationError` before fitting | — |

## Installation

Python 3.12 or later.

```bash
pip install flexpyirt  # or: uv add flexpyirt
```

This also installs [flexpyctt](https://pypi.org/project/flexpyctt/), the companion package for
analyses that need no fitted model. It depends on NumPy, SciPy and pandas only; where JAX is not
wanted, install it alone with `pip install flexpyctt`.

[flexpyequate](https://pypi.org/project/flexpyequate/), a third package from the same repository,
equates score distributions (linear, equipercentile and kernel equating). It is independent of
flexpyirt and is not installed with it: `pip install flexpyequate`
([documentation](https://takuizum.github.io/flexpyirt/equating/)).

For polars input, install `flexpyirt[polars]`. For GPUs, install a CUDA build of `jax` first
(see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

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

See the [user guide](https://takuizum.github.io/flexpyirt/guide/) for details and the [API reference](https://takuizum.github.io/flexpyirt/reference/) for
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

In wall-time comparisons with R `mirt` (default settings, `TOL = 1e-4`, 61 quadrature points on
CPU; [results on an Apple M5](https://github.com/takuizum/flexpyirt/blob/main/benchmarks/results/2026-11-21-apple-m5.md)),
warm CPU fits are faster than `mirt` in every tested case (`0.12×–0.60×` of `mirt` time) once
JAX/XLA has compiled the model on the first call (a cold EM fit takes ~0.8–1.1 s); `mirt` is
faster for a single cold fit without standard errors.

## Documentation

- [User guide](https://takuizum.github.io/flexpyirt/guide/) — specifying models, multiple groups, FIPC, editing the
  ParameterTable, estimation options
- [Theory](https://takuizum.github.io/flexpyirt/theory/) — the models, the estimation algorithm, standard errors, scoring, fit
  statistics and linking, in brief
- [API reference](https://takuizum.github.io/flexpyirt/reference/) — public classes and functions and their arguments
- [Accuracy and validation](https://takuizum.github.io/flexpyirt/accuracy/) — how results are checked against R mirt, tolerances,
  known differences from mirt

Every document is also available in Japanese ([日本語サイト](https://takuizum.github.io/flexpyirt/ja/)).

## Feedback

Bug reports and questions are welcome on [GitHub Issues](https://github.com/takuizum/flexpyirt/issues).

## License

MIT
