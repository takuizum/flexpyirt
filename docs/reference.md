# API reference

English | [日本語](reference.ja.md)

The public API of flexpyirt (everything importable with `from flexpyirt import ...`). For the
workflow see the [user guide](guide.md); for formulas and algorithms see the [design](DESIGN.md).
The docstrings carry the same information plus the formulas and sources
(`help(flexpyirt.fit)` and so on).

Apart from the first one or two, arguments of public functions are keyword-only (after `*`).

- Fitting: [`fit`](#fit), [`FitResult`](#fitresult)
- Model: [`Model`](#model)
- Item models: [`TwoPL`](#twopl), [`Rasch`](#rasch), [`GRM`](#grm)
- Latent distribution: [`Normal`](#normal)
- Settings: [`BAEM`](#baem), [`Quadrature`](#quadrature)
- Exceptions and warnings: [`flexpyirt.errors`](#flexpyirterrors)

---

## Fitting

### `fit`

```text
fit(spec, data, *, group=None, table=None, item_names=None, method=None,
    quadrature=None, se="oakes") -> FitResult
```

Marginal maximum likelihood estimation with the Bock–Aitkin EM algorithm (Bock & Aitkin, 1981).

| Argument | Type | Description |
|---|---|---|
| `spec` | `Model` | Model specification |
| `data` | DataFrame / polars DataFrame / 2-D array | One row per respondent, one column per item. Values `0..K-1`, missing as NaN. Columns must be exactly the model's items |
| `group` | array (one label per row), optional | Group labels for a multiple-group model; the group names declared in `Model.with_groups()` |
| `table` | DataFrame, optional | An edited `Model.parameter_table()`. Overrides the fixed values, equalities and starting values of `spec` |
| `item_names` | list[str], optional | Column names when `data` is an array |
| `method` | `BAEM`, optional | EM settings (default `BAEM()`) |
| `quadrature` | `Quadrature`, optional | Quadrature grid (default 61 points on $[-6, 6]$) |
| `se` | `"oakes"` / `"hessian"` / `None` | Standard-error method; `None` skips it |

**Returns**: [`FitResult`](#fitresult)

**Raises**: `PrecisionError` (JAX 64-bit disabled), `SpecificationError` (invalid specification
or data), `IdentificationError` (not identified). All are raised before estimation starts.

**Warns**: `ConvergenceWarning` (`max_iter` reached), `FlexpyirtWarning` (information matrix not
positive definite, or an anchor that is a single parameter).

### `FitResult`

Returned by `fit()`. Immutable.

| Attribute | Type | Description |
|---|---|---|
| `items` | DataFrame | One row per item (per group × item, with a `group` column, for multiple groups): `model`, internal parameters (`a`, `d` / `d1`…), `<param>_se`, display `b` / `bk` ($= -d/a$, $-d_k/a$; no SE) |
| `latent` | DataFrame | One row per group: `group`, `mean`, `var`, `mean_se`, `var_se` (SEs of fixed values are NaN) |
| `parameters` | DataFrame | The validated ParameterTable plus `estimate` (the estimate, or the fixed value) and `se` |
| `loglik` | float | Marginal log-likelihood at the estimates |
| `aic`, `bic` | float | $-2\ell + 2k$ and $-2\ell + k\log N$ ($k$ = `n_params`, $N$ = `n_obs`) |
| `n_params` | int | Number of free parameters (an equality constraint counts once) |
| `n_obs` | int | Number of respondents |
| `converged` | bool | Whether the convergence criterion was met |
| `n_iter` | int | Number of EM iterations |
| `history` | DataFrame | `iter`, `loglik` (at the parameters entering the iteration), `max_change` |
| `vcov` | ndarray / None | Covariance matrix of the free parameters (in `free_id` order); None if `se=None` |
| `se_method` | str / None | `"oakes"` / `"hessian"` / None |

#### `FitResult.score`

```text
score(data, *, method="EAP", group=None, item_names=None) -> DataFrame
```

EAP (`theta`) and posterior standard deviation (`se`) per respondent (Bock & Mislevy, 1982). The
prior is the estimated latent distribution (the respondent's group's distribution with multiple
groups) on the same grid as in fitting. Missing responses are left out of the likelihood.

| Argument | Description |
|---|---|
| `data` | Responses on the fitted items (need not be the fitting data) |
| `method` | `"EAP"` only (MAP / ML / WLE / PV are planned for M4) |
| `group` | Required for a multiple-group model: each row's group label |
| `item_names` | Column names when `data` is an array |

The index of the result follows `data` when it is a DataFrame.

---

## Model

### `Model`

An immutable model specification. Every method returns a new `Model`.

#### `Model.uniform`

```text
Model.uniform(item_model, *, items) -> Model
```

Use the same item model for every name in `items`. Example:
`Model.uniform(TwoPL(), items=df.columns)`.

#### `Model.from_items`

```text
Model.from_items(items) -> Model
```

Build from a `{item name: item model}` mapping (order is kept). Different item models can be
mixed.

#### `Model.fix`

```text
model.fix(*, item, **values) -> Model
```

Fix parameters of `item` at values of the internal (slope–intercept) parameters, in every group
with multiple groups. Example: `model.fix(item="q01", a=1.2, d=0.3)`. Unknown items or parameters
raise `SpecificationError`.

#### `Model.fix_from`

```text
model.fix_from(bank, *, items=None, parameterization="slope-intercept") -> Model
```

Fix item parameters at item-bank values (FIPC). Combine with `.with_latent(Normal.free())` to
estimate the latent distribution.

| Argument | Description |
|---|---|
| `bank` | A DataFrame with an `item` column and parameter columns: `a`, `d` (GRM: `d1`, `d2`, …) for `"slope-intercept"`; `a`, `b` (GRM: `b1`, …) for `"irt"`. Rasch items may omit `a` |
| `items` | Items to fix (a list of names or a `str -> bool` function). Default: every bank item that is also in the model |
| `parameterization` | `"slope-intercept"` or `"irt"` (converted with $d = -ab$) |

#### `Model.with_latent`

```text
model.with_latent(dist) -> Model
```

Set the latent distribution of a single-group model to a `Normal`. Not available with groups
(set them with `with_groups()`).

#### `Model.with_groups`

```text
model.with_groups(**latents) -> Model
```

Make a multiple-group model. Keyword order defines the groups, and **the first group is the
reference group**. At least two groups. Example:
`model.with_groups(ref=Normal.standard(), focal=Normal.free())`.

#### `Model.equal_across_groups`

```text
model.equal_across_groups(*, items, params=None) -> Model
```

Make the parameters of `items` equal across groups (anchor items). Without `params`, all
parameters; with e.g. `params=["a"]`, only those. `items` is a list of names, a `str -> bool`
function, or `None` for every item. Can be called repeatedly.

#### `Model.parameter_table`

```text
model.parameter_table(data, *, group=None, item_names=None) -> DataFrame
```

The default ParameterTable for `data`, with columns `group`, `item`, `model`, `param`, `value`,
`free`, `lower`, `upper`, `free_id`. Edit it and pass it to `fit(..., table=...)`
([guide section 11](guide.md#11-editing-the-parametertable)).

#### Other attributes

| Attribute / method | Description |
|---|---|
| `item_names` | Tuple of item names |
| `is_multigroup` | Whether groups were declared with `with_groups()` |
| `resolved_latent()` | The single-group latent distribution, with the default rule applied |
| `resolved_groups()` | Tuple of `(group name, Normal)`; `(("all", ...),)` for a single group |
| `items`, `fixed`, `latent`, `groups`, `equal` | The raw specification (rarely used directly) |

---

## Item models

### `TwoPL`

```text
TwoPL()
```

Two-parameter logistic model $P(y=1\mid\theta)=\sigma(a\theta+d)$ (Birnbaum, 1968). Internal
parameters `a`, `d`; display $b=-d/a$. Same as mirt's `itemtype = "2PL"`.

### `Rasch`

```text
Rasch()
```

Rasch model $P(y=1\mid\theta)=\sigma(\theta+d)$ (Rasch, 1960). `a` is fixed at 1 and `d` is
estimated. In a single-group model where every item is Rasch, the latent variance is estimated by
default (same as mirt's `itemtype = "Rasch"`).

### `GRM`

```text
GRM(n_cat)
```

Graded response model (Samejima, 1969) with categories `0..n_cat-1`:
$P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$. Internal parameters `a`,
`d1`…`d{K-1}`; display $b_k=-d_k/a$. Same as mirt's `itemtype = "graded"`. GRM items with the same
`n_cat` are estimated together.

---

## Latent distribution

### `Normal`

```text
Normal(mean=0.0, var=1.0, free_mean=False, free_var=False)
Normal.standard()
Normal.free(*, mean=0.0, var=1.0)
```

Normal distribution $N(\mu,\sigma^2)$, with weights from the density normalized on the quadrature
grid. `mean` and `var` are fixed values, or starting values when estimated.

| Construction | Meaning |
|---|---|
| `Normal.standard()` | $N(0,1)$, nothing estimated |
| `Normal(free_var=True)` | Mean fixed at 0, variance estimated |
| `Normal.free()` | Mean and variance estimated |

`var` must be positive and `mean` finite (`SpecificationError` otherwise). An estimated `var` is
the parameter of the normal distribution before the grid cuts it off
([guide section 12](guide.md#12-choosing-the-quadrature-grid)).

---

## Settings

### `BAEM`

```text
BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3)
```

| Argument | Description |
|---|---|
| `tol` | Converged when the largest absolute change of the free parameters is below this (same definition as mirt's `TOL`) |
| `max_iter` | Maximum number of iterations; reaching it gives a `ConvergenceWarning` |
| `m_step_newton_steps` | Newton steps per M-step; no step decreases the expected log-likelihood |

### `Quadrature`

```text
Quadrature.equispaced(n_points=61, *, lim=(-6.0, 6.0))
```

An equally spaced one-dimensional grid (endpoints included). The default, 61 points on
$[-6, 6]$, is mirt's default. `nodes()` returns the grid points. At least 3 points; the limits
must be finite with `lower < upper`.

---

## Exceptions and warnings

### `flexpyirt.errors`

Import with `from flexpyirt.errors import ...`.

#### `FlexpyirtError`

Base class of every exception flexpyirt raises on purpose.

#### `SpecificationError`

The specification and data disagree: missing or extra columns, out-of-range values, categories
that never occur, fixing unknown items or parameters, invalid ParameterTable edits, unknown group
labels and so on. Raised before estimation. Subclass of `ValueError`.

#### `IdentificationError`

The scale is not identified, e.g. a focal group's mean or variance is estimated without anchors,
or a single group's mean is estimated without fixed items. Raised before estimation. Subclass of
`ValueError`.

#### `PrecisionError`

JAX's 64-bit floating point is disabled. Enable it with
`jax.config.update("jax_enable_x64", True)`. Subclass of `RuntimeError`.

#### `FlexpyirtWarning`

Base class of every warning flexpyirt emits (subclass of `UserWarning`). Emitted directly when the
information matrix is not positive definite, or when a focal group's mean is linked by a single
parameter.

#### `ConvergenceWarning`

EM did not converge within `max_iter`.

#### `HeywoodWarning`

Reserved for boundary solutions (guessing parameter near 0, diverging slope). Not emitted by the
current item models (2PL / Rasch / GRM); planned for 3PL / 4PL (M3).
