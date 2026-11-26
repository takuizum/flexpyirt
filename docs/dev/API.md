# flexpyirt public API sketch

English | [日本語](API.ja.md)

> **For the implemented API, see the [API reference](../reference.md) and the [user guide](../guide.md).**
> This file is the design-stage sketch and still includes a few features that are not implemented yet
> (EH, Gauss–Hermite, `TruncNormal`, M4 scoring/MIRT, ...).

> Goal: separate "specification (Spec), estimation (fit), result (FitResult)". A specification is
> an immutable object, and method chaining returns a new specification.
> The signatures here are proposals. Update this file first when the implementation changes them.

## 1. Minimal example (single-group 2PL)

```python
import jax

jax.config.update("jax_enable_x64", True)

import pandas as pd
from flexpyirt import Model, TwoPL, fit

df = pd.read_csv("responses.csv")  # columns = items, 0/1, missing NaN
spec = Model.uniform(TwoPL(), items=df.columns)
res = fit(spec, df)

res  # a table in Jupyter (_repr_html_)
res.items  # DataFrame: item, a, b(=-d/a), a_se, b_se, ...
res.loglik, res.aic, res.bic
res.score(df, method="EAP")  # DataFrame: theta, se
```

## 2. Mixed formats + priors (MAP)

```python
from flexpyirt import Model, ThreePL, GRM, LogNormalPrior, BetaPrior, fit

spec = (
    Model.from_items(
        {
            **{f"mc{i:02d}": ThreePL() for i in range(1, 31)},
            **{f"cr{i}": GRM(n_categories=4) for i in range(1, 6)},
        }
    )
    .prior(param="a", dist=LogNormalPrior(0.0, 0.5))
    .prior(param="g", dist=BetaPrior(5, 17), items=lambda name: name.startswith("mc"))
)
res = fit(spec, df)
res.log_posterior, res.loglik
```

## 3. Multiple groups (partial anchors)

```python
from flexpyirt import Model, TwoPL, Normal, EmpiricalHistogram, fit

anchors = [f"q{i:02d}" for i in range(1, 11)]

spec = (
    Model.uniform(TwoPL(), items=df.columns.drop("form"))
    .with_groups(
        ref=Normal.standard(),  # reference group fixed at N(0,1)
        focal=Normal.free(),  # mean and variance estimated
    )
    .equal_across_groups(items=anchors)  # anchor items: all parameters equal across groups
)
res = fit(spec, df.drop(columns="form"), group=df["form"])
res.latent  # DataFrame: group, mean, sd, se
res.items  # with a group column (equal items have equal values)
```

For EH use `focal=EmpiricalHistogram()`. For an EH reference group as well,
`ref=EmpiricalHistogram(standardize=True)`.

## 4. Fixed item parameter calibration (FIPC)

```python
bank = pd.read_csv("item_bank.csv")  # item, a, d (or a, b)
spec = (
    Model.uniform(TwoPL(), items=df.columns)
    .fix_from(bank, items=anchors, parameterization="irt")  # a, b also accepted
    .with_latent(Normal.free())  # FIPC estimates the latent distribution (DESIGN §4.5)
)
res = fit(spec, df)
```

Fix individually: `.fix(item="q01", a=1.2, d=0.3)`.

## 5. Editing the parameter table directly (advanced, like mirt's pars='values')

```python
table = spec.parameter_table(df)  # long DataFrame (DESIGN §5.1)
table.loc[(table.item == "q05") & (table.param == "a"), ["value", "free"]] = [1.0, False]
res = fit(spec, df, table=table)  # the table overrides spec
```

## 6. Estimation options

```python
from flexpyirt import BAEM, Quadrature

res = fit(
    spec,
    df,
    method=BAEM(tol=1e-4, max_iter=2000, squarem=True, m_step_newton_steps=5),
    quadrature=Quadrature.equispaced(61, bounds=(-6, 6)),  # or Quadrature.gauss_hermite(41)
    se="oakes",  # "oakes" | "hessian" | "crossprod" | "sandwich" | None
    verbose=False,
)
res.converged, res.n_iter, res.history  # history: DataFrame(iter, loglik, max_change)
```

## 6.1 Implemented as of M1
- Item models: `TwoPL`, `Rasch`. Latent distribution: `Normal` (`Normal.standard()`,
  `Normal.free()`, `Normal(free_var=True)`). A single group with only Rasch items frees the
  variance by default (DESIGN §4.5).
- `fit(spec, data, *, group=None, table=None, item_names=None, method=None, quadrature=None, se="oakes")`.
  `se` is `"oakes" | "hessian" | None`. `group=` raised `NotImplementedError` until M2.
- `BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3)` (`squarem` comes in M3).
  `Quadrature.equispaced(n_points=61, bounds=(-6, 6))`; Gauss–Hermite is `Quadrature.gauss_hermite(n)` (DESIGN §15.44).
- `Model.uniform`, `Model.from_items`, `.fix(item=..., a=..., d=...)`, `.with_latent(...)`,
  `.parameter_table(data)`.
- `FitResult`: `items`, `latent`, `parameters` (table + `estimate`, `se`), `vcov`, `loglik`, `aic`,
  `bic`, `converged`, `n_iter`, `history`, `score(data, method="EAP")`.
- Input: 0/1 pandas / polars DataFrames or numpy arrays (`item_names=`). Missing data raised
  `SpecificationError` until M2.

## 6.2 Added in M2
- Item model `GRM(n_categories=K)` (categories $0..K-1$). Missing data (NaN, MAR) accepted.
- Multiple groups: `Model.with_groups(ref=Normal.standard(), focal=Normal.free())` (the first is the
  reference group), `.equal_across_groups(items=..., params=None)`, `fit(..., group=labels)`.
  `FitResult.items` has a `group` column; `score(data, group=labels)`.
- FIPC: `Model.fix_from(bank, items=None, parameterization="slope_intercept" | "irt")` +
  `.with_latent(Normal.free())`.
- Unidentified specifications (focal group free without anchors, a free mean without fixed items,
  ...) raise `IdentificationError` before fitting.

## 6.3 Added in M3 (so far)
- Item models: `ThreePL`, `FourPL`, `GPCM(n_categories=K)`, `PCM(n_categories=K)` (`HeywoodWarning` on boundary
  solutions).
- SQUAREM acceleration: `BAEM(squarem=True)` (scheme S3 with monotonicity/feasibility safeguard;
  `FitResult.n_em_evaluations`, `FitResult.n_safeguards`).
- Priors and MAP estimation: `NormalPrior(mean, sd)`, `LogNormalPrior(mean_log, sd_log)`, `BetaPrior(a, b)`
  with `log_pdf`, attached via `Model.prior(*, param, dist, items=None)` or the `prior` column of
  `ParameterTable`. `FitResult.log_posterior` (equal to `loglik` without priors),
  `history["log_posterior"]`, `FlexpyirtWarning` on `aic`/`bic` when priors are active, and
  posterior information in both `se="oakes"` and `se="hessian"`. Priors on latent-distribution
  parameters raise `SpecificationError`.

## 7. Main classes and functions

The tools that need no fitted model (`polychoric` to `detect` in the table below, except
`model_parallel_analysis` and `dimensionality`) are defined in the flexpyctt distribution and
re-exported by flexpyirt; they take `rng` (a seed or `numpy.random.Generator`) instead of a JAX
`key`. `flexpyirt.polychoric` is the JAX implementation; `flexpyctt.polychoric` the NumPy / SciPy
one (DESIGN.md §16.18).

| Name | Kind | Role |
|---|---|---|
| `Model` | frozen dataclass | The whole specification. Built with `uniform`, `from_items`; `.fix`, `.fix_from`, `.prior`, `.equal_across_groups`, `.equal_across_items`, `.with_groups`, `.with_latent`, `.parameter_table` |
| `Rasch`, `TwoPL`, `ThreePL`, `FourPL`, `GRM`, `GPCM`, `PCM`, `Nominal` | frozen dataclass (satisfies `ItemModel`) | Item models |
| `items.ItemModel`, `items.ItemKernel`, `items.register_item_model` | Protocols, function | User-defined item models and their JSON registration (DESIGN §20) |
| `Normal`, `EmpiricalHistogram`, `DavidianCurve`, `NormalMixture` | frozen dataclass (satisfies `LatentDist`) | Latent distributions. `Normal.standard()`, `Normal.free()`, `DavidianCurve(degree)`, `NormalMixture(n_components)` |
| `LogNormalPrior`, `NormalPrior`, `BetaPrior`, `TruncNormal` | frozen dataclass | Priors (with `log_pdf`). The prior is called `NormalPrior` to avoid clashing with the latent `Normal` |
| `BAEM`, `Quadrature` | frozen dataclass | Estimation settings |
| `fit` | function | Estimation; returns a `FitResult` |
| `FitResult` | class | Result (`items`, `latent`, `se`, `vcov`, `loglik`, `score`, `summary`, `item_fit`, `local_dependence`, `jsi`, `dif`, `m2`, `to_json` / `from_json`) |
| `link` | function | Differentiable IRT scale linking (Haebara / Stocking–Lord, non-symmetric & symmetric; `transformation=` for a diagonal, orthogonal or oblique $\mathbf A$ in multidimensional models); returns a `LinkResult` |
| `LinkResult` | class | Scale-linking result (`A`, `B`, `A_se`, `B_se`, `vcov`, `criterion`, `transformation`, `items`, `transform_theta`, `summary`) |
| `screen_common_items` | function | Common-item screening for linking and item parameter drift (robust z, displacement, expected-score RMSD; iterative purification with `link()`); returns a `CommonItemScreeningResult` |
| `CommonItemScreeningResult` | class | Screening result (`table`, `link`, `initial_link`, `kept_items`, `removed_items`, `history`, `flagged_items`, `summary`) |
| `stack_screening_tables` | function | Stacks the screening tables of several administrations with an `administration` column |
| `polychoric` | function | Polychoric / tetrachoric correlation matrix (two-step ML, smoothing); returns a `PolychoricResult` |
| `parallel_analysis`, `velicer_map`, `empirical_kaiser` | function | Number-of-dimensions diagnostics; return `ParallelAnalysisResult`, `MAPResult`, `EmpiricalKaiserResult` (frozen dataclasses with `n_factors`) |
| `comparison_data` | function | Comparison-data method (Ruscio & Roche, 2012); returns `ComparisonDataResult` |
| `efa`, `hull` | function | Unrotated ULS / ML EFA of a correlation matrix with EFAtools' fit indices; the Hull method. Return `EFAResult`, `HullResult` |
| `dimtest` | function | DIMTEST with bootstrap bias correction; returns `DimtestResult` |
| `detect`, `exploratory_detect` | function | Confirmatory / exploratory DETECT; return `DetectResult`, `ExploratoryDetectResult` |
| `model_parallel_analysis`, `dimensionality` | function | Model-based parallel analysis of a `FitResult`; the ensemble report. Return `ModelParallelAnalysisResult`, `DimensionalityResult` |
| `item_information`, `test_information`, `expected_score` | function | Item / test information (with the conditional SE) and expected scores of a `FitResult` at given `theta`; return DataFrames indexed by `theta` (DESIGN §18.2) |
| `information_matrix` | function | Test (or one item's) Fisher information matrix at given `theta`, a `(T, D, D)` array; for multidimensional models `item_information` / `test_information` take `direction=` (directional information and the composite SE) and `theta` of shape `(T, D)`, indexed by a factor-name `MultiIndex` (DESIGN §18.3) |
| `item_probabilities`, `empirical_item_curve`, `wright_map_table` | function | The numbers behind the plots: trace lines (long DataFrame), observed category proportions in equal-count bins of EAP $\theta$ beside the model, and the person scores and Thurstonian item thresholds of a Wright map (`WrightMapTable(persons, items)`) (DESIGN §21) |
| `flexpyirt.plotting` | module | `plot_item_curves` (with observed proportions; one curve per group = DIF plot), `plot_information`, `plot_expected_score`, `plot_wright_map`; matplotlib from the optional extra `flexpyirt[plot]`, imported lazily (`MissingDependencyError` without it) (DESIGN §21) |
| `reliability`, `summed_score_table` | function | Marginal and empirical reliability; the summed-score EAP table (also `FitResult.score(method="EAPsum")`). Return DataFrames |
| `summed_score_distribution` | function | Lord–Wingersky / Poisson-binomial kernel on category probabilities (DESIGN §18.1) |
| `compare_fits` | function | Model-comparison table over fits of the same data: `n_params`, `loglik`, AIC, SABIC, HQ, BIC and likelihood-ratio tests against the previous row (mirt's `anova()`; DESIGN §24.1) |
| `simulate_responses`, `SimulatedData` | function, frozen dataclass | Simulated responses from a `FitResult` or a `Model` + item bank, with a JAX `key`; returns `responses`, `theta`, `group` (DESIGN §24.2) |
| `LocalDependenceResult` | frozen dataclass | Returned by `FitResult.local_dependence()` (`pairs`, `q3`, `q3_mean`, `max_q3_star`) |
| `JSIResult` | frozen dataclass | Returned by `FitResult.jsi()` (`matrix`, `pairs`, `method`, `converged`) |
| `PolychoricResult` | frozen dataclass | `correlation`, `unsmoothed_correlation`, `thresholds`, `categories`, `n_obs`, per-pair `converged` / `at_boundary` / `zero_cell_corrected`, `smoothed`, `min_eigenvalue` |

## 8. Design commitments
- Public functions are keyword-only by default (`fit(spec, data, *, group=None, ...)`).
- String options are typed with `Literal` (`se: Literal["oakes", ...]`).
- Inputs accept pandas / polars / numpy. Column names become item names and NaN means missing; for
  numpy, pass `item_names=`.
- Specification errors raise `SpecificationError` **before** fitting, with a message naming the
  offending item and parameter.
- Anything random (plausible values, ...) takes an explicit `key: jax.Array` (no global random
  state).
