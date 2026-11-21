# API reference

English | [日本語](reference.ja.md)

The public API of flexpyirt (everything importable with `from flexpyirt import ...`). For the
workflow see the [user guide](guide.md); for formulas and algorithms see the [design](DESIGN.md).
The docstrings carry the same information plus the formulas and sources
(`help(flexpyirt.fit)` and so on).

Apart from the first one or two, arguments of public functions are keyword-only (after `*`).

- Fitting: [`fit`](#fit), [`FitResult`](#fitresult)
- Model: [`Model`](#model)
- Item models: [`TwoPL`](#twopl), [`Rasch`](#rasch), [`ThreePL`](#threepl), [`FourPL`](#fourpl), [`GRM`](#grm), [`GPCM`](#gpcm), [`PCM`](#pcm)
- Priors: [`NormalPrior`](#normalprior), [`LogNormal`](#lognormal), [`Beta`](#beta)
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

Marginal maximum likelihood (or Bayes modal / MAP when priors are attached) estimation with the
Bock–Aitkin EM algorithm (Bock & Aitkin, 1981).

| Argument | Type | Description |
|---|---|---|
| `spec` | `Model` | Model specification |
| `data` | DataFrame / polars DataFrame / 2-D array | One row per respondent, one column per item. Values `0..K-1`, missing as NaN. Columns must be exactly the model's items |
| `group` | array (one label per row), optional | Group labels for a multiple-group model; the group names declared in `Model.with_groups()` |
| `table` | DataFrame, optional | An edited `Model.parameter_table()`. Overrides the fixed values, equalities, priors and starting values of `spec` |
| `item_names` | list[str], optional | Column names when `data` is an array |
| `method` | `BAEM`, optional | EM settings (default `BAEM()`) |
| `quadrature` | `Quadrature`, optional | Quadrature grid (default 61 points on $[-6, 6]$) |
| `se` | `"oakes"` / `"hessian"` / `None` | Standard-error method (includes prior Hessian under MAP); `None` skips it |

**Returns**: [`FitResult`](#fitresult)

**Raises**: `PrecisionError` (JAX 64-bit disabled), `SpecificationError` (invalid specification
or data), `IdentificationError` (not identified). All are raised before estimation starts.

**Warns**: `ConvergenceWarning` (`max_iter` reached, or SQUAREM fell back to plain EM frequently),
`HeywoodWarning` (free asymptote within `1e-4` of its bound or free slope $|a| > 20$),
`FlexpyirtWarning` (information matrix not positive definite, or an anchor that is a single
parameter).

### `FitResult`

Returned by `fit()`. Immutable.

| Attribute | Type | Description |
|---|---|---|
| `items` | DataFrame | One row per item (per group × item, with a `group` column, for multiple groups): `model`, internal parameters (`a`, `d` / `d1`…, `g`, `u`), `<param>_se`, display `b` / `bk` ($-d/a$, GRM $-d_k/a$, GPCM/PCM $-(d_k-d_{k-1})/a$ with $d_0=0$; no SE) |
| `latent` | DataFrame | One row per group: `group`, `mean`, `var`, `mean_se`, `var_se` (SEs of fixed values are NaN) |
| `parameters` | DataFrame | The validated ParameterTable plus `estimate` (the estimate, or the fixed value) and `se` |
| `loglik` | float | Marginal log-likelihood at the estimates (likelihood only, excluding priors) |
| `log_posterior` | float | Log posterior ($\ell(\hat\eta) + \log\pi(\hat\eta)$) at the estimates; equals `loglik` without priors |
| `aic`, `bic` | float | $-2\ell + 2k$ and $-2\ell + k\log N$ ($k$ = `n_params`, $N$ = `n_obs`); emits `FlexpyirtWarning` when priors are active |
| `n_params` | int | Number of free parameters (an equality constraint counts once) |
| `n_obs` | int | Number of respondents |
| `converged` | bool | Whether the convergence criterion was met |
| `n_iter` | int | Number of accepted EM (or SQUAREM) iterations |
| `n_em_evaluations` | int | Total number of EM map evaluations (E-step + M-step); equals `n_iter` for plain EM, and 2 or 3 per accepted iterate for SQUAREM |
| `n_safeguards` | int | Number of SQUAREM iterations on which the monotonicity/feasibility safeguard intervened (0 for plain EM) |
| `history` | DataFrame | `iter`, `loglik`, `log_posterior` (at the parameters entering the iteration; non-decreasing across iterations), `max_change`, `em_evaluations` (cumulative EM map evaluations), `safeguard` (whether the SQUAREM safeguard fired) |
| `vcov` | ndarray / None | Covariance matrix of the free parameters (in `free_id` order); None if `se=None` |
| `se_method` | str / None | `"oakes"` / `"hessian"` / None |

#### `FitResult.log_posterior`

```text
res.log_posterior -> float
```

The log posterior $\ell(\hat\eta) + \sum_i \log\pi_i(\hat\eta_i)$ evaluated on the natural scale
over all free parameters (counted once per `free_id`; fixed parameters ignore their prior). Equal
to `res.loglik` when no priors are active.

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
| `bank` | A DataFrame with an `item` column and parameter columns: `a`, `d` (GRM/GPCM/PCM: `d1`, `d2`, …) for `"slope-intercept"`; `a`, `b` (GRM/GPCM/PCM: `b1`, …) for `"irt"`. Rasch and PCM items may omit `a`. Asymptotes `g` (3PL/4PL) and `u` (4PL) are passed unchanged in both parameterizations; `u` may be omitted for 3PL items |
| `items` | Items to fix (a list of names or a `str -> bool` function). Default: every bank item that is also in the model |
| `parameterization` | `"slope-intercept"` or `"irt"` (converted with $d = -ab$, or $d_k = -a\sum_{c=1}^{k} b_c$ for GPCM/PCM) |

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

#### `Model.prior`

```text
model.prior(*, param, dist, items=None) -> Model
```

Attach a prior distribution `dist` (`NormalPrior`, `LogNormal`, or `Beta`) to internal parameter
`param` of the selected items across all groups (MAP estimation; [guide section 11](guide.md#11-priors-and-map-estimation)).
Priors are evaluated on the natural parameter scale without a Jacobian term. Fixed parameters
ignore their prior, and a parameter shared across groups (`free_id`) is counted once. Priors on
latent-distribution parameters (`"mean"`, `"var"`) raise `SpecificationError`.

| Argument | Description |
|---|---|
| `param` | Internal item parameter name (`"a"`, `"d"`, `"g"`, `"u"`, `"d1"`, …) |
| `dist` | `NormalPrior`, `LogNormal`, or `Beta` |
| `items` | List of item names, a `str -> bool` function, or `None` for every item |

#### `Model.parameter_table`

```text
model.parameter_table(data, *, group=None, item_names=None) -> DataFrame
```

The default ParameterTable for `data`, with columns `group`, `item`, `model`, `param`, `value`,
`free`, `lower`, `upper`, `prior`, `free_id`. Edit it and pass it to `fit(..., table=...)`
([guide section 12](guide.md#12-editing-the-parametertable)).

#### Other attributes

| Attribute / method | Description |
|---|---|
| `item_names` | Tuple of item names |
| `is_multigroup` | Whether groups were declared with `with_groups()` |
| `resolved_latent()` | The single-group latent distribution, with the default rule applied |
| `resolved_groups()` | Tuple of `(group name, Normal)`; `(("all", ...),)` for a single group |
| `items`, `fixed`, `latent`, `groups`, `equal`, `priors` | The raw specification (rarely used directly) |

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

### `ThreePL`

```text
ThreePL()
```

Three-parameter logistic model $P(y=1\mid\theta)=g+(1-g)\sigma(a\theta+d)$ (Birnbaum, 1968).
Internal parameters `a`, `d`, `g` (free, $0 < g < 1$ on the natural scale) and `u` (fixed at 1);
display $b=-d/a$. Same model and parameter names as mirt's `itemtype = "3PL"` (mirt estimates
`g` on the logit scale; flexpyirt works on the natural scale). Shares `FourPLKernel` with
`FourPL`. Fixing `g = 0` reproduces `TwoPL` exactly.

### `FourPL`

```text
FourPL()
```

Four-parameter logistic model $P(y=1\mid\theta)=g+(u-g)\sigma(a\theta+d)$ (Barton & Lord, 1981).
Internal parameters `a`, `d`, `g`, `u` (all free, $0 < g < u < 1$ on the natural scale); display
$b=-d/a$. Same model and parameter names as mirt's `itemtype = "4PL"`. Shares `FourPLKernel`
with `ThreePL`.

### `GRM`

```text
GRM(n_cat)
```

Graded response model (Samejima, 1969) with categories `0..n_cat-1`:
$P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$. Internal parameters `a`,
`d1`…`d{K-1}`; display $b_k=-d_k/a$. Same as mirt's `itemtype = "graded"`. GRM items with the same
`n_cat` are estimated together.

### `GPCM`

```text
GPCM(n_cat)
```

Generalized partial credit model (Muraki, 1992) with categories `0..n_cat-1` (`n_cat >= 2`):
$P(y=k\mid\theta)=\exp(k\,a\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)$ with $d_0=0$ fixed.
Internal parameters `a`, `d1`…`d{K-1}` (no ordering constraint); display step difficulties
$b_k=-(d_k-d_{k-1})/a$ with $d_0=0$. Same as mirt's `itemtype = "gpcm"`. With `n_cat = 2` this
equals `TwoPL`. Shares `GPCMKernel` with `PCM` of the same `n_cat`.

### `PCM`

```text
PCM(n_cat)
```

Partial credit model (Masters, 1982) with categories `0..n_cat-1` (`n_cat >= 2`): GPCM with `a = 1`
fixed, $P(y=k\mid\theta)=\exp(k\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\theta+d_c)$ with $d_0=0$ fixed.
Display step difficulties $b_k=-(d_k-d_{k-1})$. In a single-group model whose items are all `Rasch`
or `PCM`, the latent variance is estimated by default (same as mirt's `itemtype = "Rasch"` on
polytomous data). With `n_cat = 2` this equals `Rasch`. Shares `GPCMKernel` with `GPCM` of the same
`n_cat`.

---

## Priors

Frozen dataclasses representing item-parameter priors on the natural parameter scale without a
Jacobian term (DESIGN §4.7). Each validates its hyperparameters (`SpecificationError`), exposes
`support`, and provides `log_pdf(x) -> Array`.

### `NormalPrior`

```text
NormalPrior(mean, sd)
```

Normal prior on $(-\infty, \infty)$:
$\log\pi(x) = -\tfrac{1}{2}\log(2\pi) - \log\sigma - \frac{(x-\mu)^2}{2\sigma^2}$
(Mislevy, 1986; mirt `norm`). Requires finite `mean` and positive finite `sd`. Named `NormalPrior`
to avoid clashing with the latent distribution `Normal`.

### `LogNormal`

```text
LogNormal(meanlog, sdlog)
```

Log-normal prior on $(0, \infty)$ ($\log X \sim N(\mu, s^2)$):
$\log\pi(x) = -\log x - \log s - \tfrac{1}{2}\log(2\pi) - \frac{(\log x - \mu)^2}{2s^2}$
($-\infty$ for $x \le 0$; Mislevy, 1986; mirt `lnorm`). Requires finite `meanlog` and positive
finite `sdlog`. Commonly used for item slopes `a`.

### `Beta`

```text
Beta(a, b)
```

Beta prior on $(0, 1)$:
$\log\pi(x) = (a - 1)\log x + (b - 1)\log(1 - x) - \log B(a, b)$
($-\infty$ outside $(0, 1)$; Swaminathan & Gifford, 1986; mirt `expbeta` log-prior objective on
the natural scale without a logit Jacobian). Requires positive finite `a` and `b`. Commonly used
for 3PL/4PL asymptotes `g` and `u`.

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
([guide section 13](guide.md#13-choosing-the-quadrature-grid)).

---

## Settings

### `BAEM`

```text
BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3, squarem=False)
```

| Argument | Description |
|---|---|
| `tol` | Converged when the largest absolute change of the free parameters between successive accepted iterates is below this (same definition as mirt's `TOL`) |
| `max_iter` | Maximum number of accepted iterations; reaching it gives a `ConvergenceWarning` |
| `m_step_newton_steps` | Newton steps per M-step; no step decreases the expected log-likelihood |
| `squarem` | Enable SQUAREM scheme S3 acceleration (Varadhan & Roland, 2008) with a monotonicity/feasibility safeguard (default `False`) |

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
information matrix is not positive definite, when a focal group's mean is linked by a single
parameter, or when `FitResult.aic` / `FitResult.bic` is accessed on a model with active priors.

#### `ConvergenceWarning`

EM did not converge within `max_iter`, or SQUAREM fell back to plain EM frequently (at least 5
full fallbacks to $\theta_2$ and more than 50% of iterations).

#### `HeywoodWarning`

Emitted after fitting when a free item parameter ends within `1e-4` of a finite bound (e.g. `g`
near 0 or `u` near 1 in 3PL/4PL) or a free slope has $|a| > 20$, naming every affected item.
