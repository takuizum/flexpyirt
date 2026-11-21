# flexpyirt public API sketch

English | [日本語](API.ja.md)

> **For the implemented API, see the [API reference](reference.md) and the [user guide](guide.md).**
> This file is the design-stage sketch and includes features that are not implemented yet (3PL,
> priors, EH, ...).

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
from flexpyirt import Model, ThreePL, GRM, LogNormal, Beta, fit

spec = (
    Model.from_items(
        {
            **{f"mc{i:02d}": ThreePL() for i in range(1, 31)},
            **{f"cr{i}": GRM(n_cat=4) for i in range(1, 6)},
        }
    )
    .prior(param="a", dist=LogNormal(0.0, 0.5))
    .prior(param="g", dist=Beta(5, 17), items=lambda name: name.startswith("mc"))
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
    quadrature=Quadrature.equispaced(61, lim=(-6, 6)),  # or Quadrature.gauss_hermite(41)
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
  `Quadrature.equispaced(n_points=61, lim=(-6, 6))` (Gauss–Hermite is not implemented).
- `Model.uniform`, `Model.from_items`, `.fix(item=..., a=..., d=...)`, `.with_latent(...)`,
  `.parameter_table(data)`.
- `FitResult`: `items`, `latent`, `parameters` (table + `estimate`, `se`), `vcov`, `loglik`, `aic`,
  `bic`, `converged`, `n_iter`, `history`, `score(data, method="EAP")`.
- Input: 0/1 pandas / polars DataFrames or numpy arrays (`item_names=`). Missing data raised
  `SpecificationError` until M2.

## 6.2 Added in M2
- Item model `GRM(n_cat=K)` (categories $0..K-1$). Missing data (NaN, MAR) accepted.
- Multiple groups: `Model.with_groups(ref=Normal.standard(), focal=Normal.free())` (the first is the
  reference group), `.equal_across_groups(items=..., params=None)`, `fit(..., group=labels)`.
  `FitResult.items` has a `group` column; `score(data, group=labels)`.
- FIPC: `Model.fix_from(bank, items=None, parameterization="slope-intercept" | "irt")` +
  `.with_latent(Normal.free())`.
- Unidentified specifications (focal group free without anchors, a free mean without fixed items,
  ...) raise `IdentificationError` before fitting.

## 7. Main classes and functions

| Name | Kind | Role |
|---|---|---|
| `Model` | frozen dataclass | The whole specification. Built with `uniform`, `from_items`; `.fix`, `.fix_from`, `.prior`, `.equal_across_groups`, `.with_groups`, `.with_latent`, `.parameter_table` |
| `Rasch`, `TwoPL`, `ThreePL`, `FourPL`, `GRM`, `GPCM` | frozen dataclass (satisfies `ItemModel`) | Item models |
| `Normal`, `EmpiricalHistogram` | frozen dataclass (satisfies `LatentDist`) | Latent distributions. `Normal.standard()`, `Normal.free()` |
| `LogNormal`, `NormalPrior`, `Beta`, `TruncNormal` | frozen dataclass | Priors (with `log_pdf`). The prior is called `NormalPrior` to avoid clashing with the latent `Normal` |
| `BAEM`, `Quadrature` | frozen dataclass | Estimation settings |
| `fit` | function | Estimation; returns a `FitResult` |
| `FitResult` | class | Result (`items`, `latent`, `se`, `vcov`, `loglik`, `score`, `summary`, `to_json` / `from_json`) |

## 8. Design commitments
- Public functions are keyword-only by default (`fit(spec, data, *, group=None, ...)`).
- String options are typed with `Literal` (`se: Literal["oakes", ...]`).
- Inputs accept pandas / polars / numpy. Column names become item names and NaN means missing; for
  numpy, pass `item_names=`.
- Specification errors raise `SpecificationError` **before** fitting, with a message naming the
  offending item and parameter.
- Anything random (plausible values, ...) takes an explicit `key: jax.Array` (no global random
  state).
