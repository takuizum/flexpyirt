# API reference

English | [日本語](reference.ja.md)

The public API of flexpyirt (everything importable with `from flexpyirt import ...`). For the
workflow see the [user guide](guide.md); for formulas and algorithms see the [design](DESIGN.md).
The docstrings carry the same information plus the formulas and sources
(`help(flexpyirt.fit)` and so on).

Apart from the first one or two, arguments of public functions are keyword-only (after `*`).

- Fitting: [`fit`](#fit), [`FitResult`](#fitresult)
- Scale linking: [`link`](#link), [`LinkResult`](#linkresult)
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
| `quadrature` | `Quadrature`, optional | Quadrature grid (default: 61 points on $[-6, 6]$ for 1 factor, 31 per dimension for 2 factors, 15 per dimension for 3 factors) |
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
| `items` | DataFrame | One row per item (per group × item, with a `group` column, for multiple groups): `model`, internal parameters (`a` or `a1`…`aD`, `d` / `d1`…, `g`, `u`), `<param>_se`, and for unidimensional models display `b` / `bk` ($-d/a$, GRM $-d_k/a$, GPCM/PCM $-(d_k-d_{k-1})/a$ with $d_0=0$) and delta-method `b_se` / `bk_se` (from `vcov` via JAX autodiff; NaN when `se=None` or all parameters entering a threshold are fixed) |
| `latent` | DataFrame | One row per group: `group`, `mean`, `var`, `mean_se`, `var_se` for 1 factor, or `group`, `mean_1`…`mean_D`, `var_1`…`var_D`, `cor_12`… and their `_se` columns for 2–3 factors (SEs of fixed values are NaN); for `EmpiricalHistogram`, `group`, `mean`, `var`, `sd`, `skewness`, `kurtosis` (excess) of each histogram |
| `latent_weights` | DataFrame / None | `EmpiricalHistogram` only: `group`, `node`, `weight` (one row per group × quadrature node); None for `Normal` |
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

#### `FitResult.summary`

```text
summary(*, digits=4) -> str
```

Return a readable plain-text summary containing the item model(s) and estimation mode (`MMLE` or
`MAP`), sample/item/free-parameter counts, group summary, convergence status with iteration and
EM-evaluation counts (plus SQUAREM safeguards when nonzero), SE method, quadrature grid,
log-likelihood, log-posterior, AIC and BIC (with an inline note when priors are active, without
emitting runtime warnings), the latent distribution table, and the item parameter table with
standard errors (including delta-method `b_se` / `bk_se`). Raises `SpecificationError` if `digits`
is not a non-negative integer.

In Jupyter notebooks, `FitResult._repr_html_()` renders the same information in plain,
self-contained HTML/CSS (no external assets, HTML-escaped names), while `repr(res)` remains a
concise one-line summary.

#### `FitResult.score`

```text
score(data, *, method="EAP", group=None, item_names=None, key=None, n_draws=5) -> DataFrame
```

Score respondents on the fitted model ([guide section 7](guide.md#7-scoring)). Missing responses
are dropped from the likelihood (and from $I(\theta)$ in WLE) for every method.

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / polars DataFrame / 2-D array | Responses on the fitted items (need not be the fitting data) |
| `method` | `"EAP"` / `"MAP"` / `"ML"` / `"WLE"` / `"PV"` | Scoring method (default `"EAP"`): EAP (posterior mean & SD on the fitted quadrature grid; Bock & Mislevy, 1982), MAP (posterior mode with continuous normal prior $N(\mu_g,\sigma_g^2)$ & observed curvature SE), ML (maximum likelihood & observed-information SE; $-\infty$/$+\infty$ with SE NaN for patterns with no finite maximum, NaN for all-missing), WLE (Warm, 1989; Firth, 1993: solves $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta))=0$, finite on extreme patterns, NaN for all-missing, SE from observed WLE curvature), or PV (`n_draws` independent posterior draws via piecewise-linear inverse-CDF interpolation on a 501-node grid; Mislevy, 1991) |
| `group` | array, optional | Required for a multiple-group model: each row's group label |
| `item_names` | list[str], optional | Column names when `data` is an array |
| `key` | `jax.Array`, optional | Explicit JAX PRNG key (from `jax.random.key()` or `jax.random.PRNGKey()`). Required for `method="PV"`; must be `None` for other methods |
| `n_draws` | int | Number of plausible values per respondent when `method="PV"` (default `5`, must be `>= 1`) |

**Returns**: `DataFrame` with one row per respondent (index follows `data` when it is a
DataFrame): for unidimensional models, columns `theta`, `se` for `"EAP"`, `"MAP"`, `"ML"`, `"WLE"`
and `pv1`…`pv{n_draws}` for `"PV"`; for multidimensional models, columns `theta_<factor>` and `se_<factor>` in declared factor
order (`"PV"`: `pv<m>_<factor>`).

**Raises**: `SpecificationError` if `method` is unknown, `method="MAP"` with an `EmpiricalHistogram` latent distribution, `key` is missing or invalid for `"PV"`, `key` is supplied for a deterministic method, or
`n_draws` is not a positive integer.

#### `FitResult.item_fit`

```text
item_fit(data, *, mincell=1.0, item_names=None) -> DataFrame
```

Orlando–Thissen S-X2 item fit ([guide section 5](guide.md#item-fit-s-x2)). For each item,
observed and model-implied counts of each category at summed scores $1,\dots,S-1$ are compared;
the expected counts use the Lord–Wingersky recursion with the item left out and the fitted latent
weights on the fitted quadrature grid. Sparse cells are collapsed exactly as mirt's `itemfit()`
(`mincell` is mirt's argument), and $df_j$ = kept cells − rows − free parameters of the item −
free latent parameters (mirt's convention).

| Argument | Description |
|---|---|
| `data` | Complete responses on the fitted items (typically the fitting data) |
| `mincell` | Minimum expected count per cell before collapsing (default `1.0`) |
| `item_names` | Column names when `data` is an array |

**Returns**: `DataFrame` with columns `item`, `s_x2`, `df`, `rmsea`
($\sqrt{\max(0, X^2/df-1)/(N-1)}$) and `p`; items whose collapsed table leaves `df <= 0` get
`df = 0` and NaN statistics.

**Raises**: `SpecificationError` for multidimensional, multiple-group or `EmpiricalHistogram`
models, missing responses, an item with an unobserved category, or a non-positive `mincell`.

#### `FitResult.dif`

```text
dif(data, *, group, items, params=None, scheme="add", p_adjust="holm", item_names=None) -> DataFrame
```

Likelihood-ratio DIF tests across the groups of a multiple-group model
([guide section 9](guide.md#differential-item-functioning-dif)), as mirt's `DIF()`. For each item
the model is refitted (same `BAEM` settings and quadrature, starting from these estimates) with
the item's `params` made equal across groups (`scheme="add"`) or freed in every group
(`scheme="drop"`); $X^2 = 2(\ell_{\text{free}} - \ell_{\text{constrained}})$ with
$df = |k_{\text{base}} - k_{\text{new}}|$.

| Argument | Description |
|---|---|
| `data`, `group` | The responses and group labels this result was fitted to |
| `items` | Items to test (one refit each) |
| `params` | Internal parameters to test (default: all of each item's parameters) |
| `scheme` | `"add"` (constrain; the fitted model is the baseline) or `"drop"` (free) |
| `p_adjust` | `"holm"` (default, mirt's), `"BH"` (Benjamini–Hochberg) or `"none"` for `adj_p` |
| `item_names` | Column names when `data` is an array |

**Returns**: `DataFrame` with columns `item`, `converged`, `aic`, `sabic`, `hq`, `bic` (the model
with the item free minus the constrained one; negative values favour DIF), `x2`, `df`, `p` and
`adj_p`.

**Raises**: `SpecificationError` for single-group models, MAP fits with priors, unknown items or
parameters, or an item with nothing to test (already equal for `"add"`, already free for
`"drop"`); `IdentificationError` if a refitted model is not identified (e.g. freeing the only
anchors).

#### `FitResult.to_json`

```text
to_json(path=None, *, indent=2) -> str
```

Serialize the fitted result to a strict RFC 8259 JSON string, and also write it (UTF-8) to `path`
(`str` or `os.PathLike`) when `path` is given. Uses plain, versioned JSON (no `pickle`) with
top-level `"format": "flexpyirt.FitResult"`, `"schema_version": 3` (files with versions 1 and 2 are
still read), and `"flexpyirt_version"`. Multidimensional results store their factor structure.
Finite float64 values are written as `repr`-exact JSON numbers and non-finite values (`NaN`,
`+inf`, `-inf`) are encoded explicitly as `"NaN"`, `"Infinity"`, and `"-Infinity"` in float-typed
fields so every attribute round-trips bit-for-bit.

| Argument | Type | Description |
|---|---|---|
| `path` | `str` / `os.PathLike`, optional | Output file path to write the UTF-8 JSON to (optional) |
| `indent` | `int` / `None` | Indentation level for `json.dumps` (default `2`; `None` for compact JSON) |

**Returns**: `str` (the JSON text).

**Raises**: `SpecificationError` if `indent` is invalid or the result contains an unregistered
model/prior type.

#### `FitResult.from_json`

```text
FitResult.from_json(source) -> FitResult
```

Load a `FitResult` from a file path (`str` or `os.PathLike`) or a JSON string without refitting.
Item models, latent distributions, priors, quadrature grids, and estimation settings are
reconstructed through explicit name registries (never `eval` or `pickle`). The loaded result
reproduces every attribute, `items`, `latent`, `summary()`, `_repr_html_()`, `aic`/`bic`, and
`score()` identically. Calls `check_x64()` at entry because scoring requires 64-bit floating point.

| Argument | Type | Description |
|---|---|---|
| `source` | `str` / `os.PathLike` | Path to a JSON file written by `to_json()`, or a JSON string |

**Returns**: `FitResult`

**Raises**: `PrecisionError` if JAX 64-bit floating point is disabled; `SpecificationError` if the
JSON is malformed or non-strict (e.g. bare `NaN`/`Infinity` literals), the format tag, schema
version, or a registered type name is unknown, or internal consistency checks fail;
`FileNotFoundError` if `source` points to a nonexistent file.

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

#### `Model.with_factors`

```text
model.with_factors(factors, *, correlated=True) -> Model
```

Declare a confirmatory 2- or 3-dimensional loading pattern ([guide section 14](guide.md#14-confirmatory-multidimensional-models)).
`factors` maps factor names (in order, 2 or 3 entries) to the items loading on each factor
(an iterable of item names or a `str -> bool` callable). Cross-loadings are allowed; each factor
must have at least one item, every model item must load on at least one factor, and unloaded
slopes are fixed at `0`. Item parameters become `a1`…`aD`, `d` (or `d1`…). The latent
distribution is multivariate normal with means fixed at `0`, variances fixed at `1` (or free when
all items are `Rasch` or `PCM`), and correlations free when `correlated=True` (matching mirt's
`COV = F1*F2`) or fixed at `0` when `correlated=False`. Multiple groups (`with_groups()`) and
`with_latent()` are not supported for multidimensional models (`SpecificationError`).

#### `Model.with_latent`

```text
model.with_latent(dist) -> Model
```

Set the latent distribution of a single-group model to a `Normal` or an `EmpiricalHistogram`
(unidimensional models only). Not available with groups
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

#### `Model.equal_across_items`

```text
model.equal_across_items(*, items, params) -> Model
```

Make each parameter in `params` (e.g. `["a"]`, `["a1"]`, `["d"]`, `["d1", "d2"]`) one common
parameter for the selected `items` (at least two), within each group (as mirt's `CONSTRAIN`).
Combine with `equal_across_groups` to share across groups as well; overlapping equalities are
merged. Fixed parameters are not affected. The parameter's default starting value is the mean of
the items' starting values. Equivalent to giving the rows one `free_id` in an edited table.

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
| `is_multidimensional` | Whether confirmatory factors were declared with `with_factors()` |
| `n_factors` | Number of latent factors (`1`, `2`, or `3`) |
| `factor_names` | Tuple of factor names (empty for a unidimensional model) |
| `resolved_latent()` | The single-group latent distribution, with the default rule applied |
| `resolved_groups()` | Tuple of `(group name, Normal)`; `(("all", ...),)` for a single group |
| `items`, `fixed`, `latent`, `groups`, `equal`, `priors`, `factors`, `correlated_factors` | The raw specification (rarely used directly) |

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

### `EmpiricalHistogram`

```text
EmpiricalHistogram(*, min_weight=1e-10)
```

Empirical histogram (Woods, 2007): free weights $w_q \ge$ `min_weight` on the quadrature nodes,
summing to 1, estimated at every EM iteration ([guide section 8](guide.md#8-latent-distributions-and-identification)).
Unless fixed items set the scale (FIPC), the reference group's histogram is constrained to mean 0
and variance 1 (mean only for Rasch/PCM). With multiple groups every group must use EH. The
weights count as parameters in `n_params` ($Q-3$ for a standardized reference group, $Q-2$ for
Rasch/PCM, $Q-1$ under FIPC and for each focal group). Standard errors are conditional on the
fitted histogram. Unidimensional models only.

| Argument | Description |
|---|---|
| `min_weight` | Floor on every node's weight; must be in $(0, 1)$ and below $1/Q$ for the grid used |

**Raises**: `SpecificationError` for an invalid `min_weight`, for mixing `EmpiricalHistogram` and
`Normal` across groups, or for a multidimensional model; `IdentificationError` when intercepts are
fixed but no nonzero slope is (a variance-only constraint is not supported).

---

## Scale linking

### `link`

```text
link(source, target, *, items=None, method="stocking-lord", symmetric=False,
     nodes=None, weights=None, source_group=None, target_group=None,
     tol=1e-10, max_iter=200) -> LinkResult
```

Link a separately calibrated unidimensional `source` (new) scale onto `target` (base) via common
items by finding linear constants $(A, B)$ ($A > 0$) for $\theta^* = A\theta + B$
([guide section 15](guide.md#15-scale-linking), [DESIGN §16.5](DESIGN.md)). Validated (V20) for
2PL, 3PL, GRM, GPCM and mixed 2PL + GPCM common items; Rasch, 4PL and PCM share these kernels and
are covered by unit tests only. Derivatives for Newton optimization and delta-method standard
errors (Ogasawara, 2001) come from JAX automatic differentiation. SEs assume independent
calibrations (the joint covariance is used when one multiple-group fit is both `source` and
`target`), hold the quadrature weights fixed, and are not reported without convergence.

| Argument | Type | Description |
|---|---|---|
| `source` | `FitResult` | Fitted unidimensional model on the source (new) scale |
| `target` | `FitResult` | Fitted unidimensional model on the target (base) scale |
| `items` | iterable[str] / Mapping[str, str], optional | Common items (default `None`: all items sharing the same name). Pass `{source_item: target_item}` when names differ. Common pairs must share the same kernel and category count |
| `method` | `"stocking-lord"` / `"haebara"` | Criterion: Stocking & Lord (1983) squared TCC difference (default) or Haebara (1980) squared category-curve differences (polytomous forms: Kim & Lee, 2006) |
| `symmetric` | bool | If `True`, add the reverse-direction squared difference (Kim & Lee, 2006); default `False` |
| `nodes` | 1-D array-like, optional | Quadrature nodes $\theta_q$ (default: target group's fitted grid, and source group's grid for the reverse direction when `symmetric=True`) |
| `weights` | 1-D array-like, optional | Non-negative weights normalized to sum to 1 (default: fitted latent distribution weights, or uniform when `nodes` is given without `weights`) |
| `source_group`, `target_group` | str, optional | Group name when `source` or `target` is a multiple-group `FitResult` |
| `tol` | float | Convergence tolerance on $\|\nabla F(A, B)\|_\infty$ (default `1e-10`) |
| `max_iter` | int | Maximum Newton iterations (default `200`) |

**Returns**: [`LinkResult`](#linkresult)

**Raises**: `PrecisionError` (JAX 64-bit disabled), `SpecificationError` (non-`FitResult` or
multidimensional input, multiple-group input without selecting a group, no common items,
mismatched item kernels/categories on common items, invalid options/nodes/weights).

**Warns**: `ConvergenceWarning` if `max_iter` is reached before `grad_norm <= tol`.

### `LinkResult`

Returned by `link()`. Immutable.

| Attribute / method | Type | Description |
|---|---|---|
| `A`, `B` | float | Linear linking constants ($\theta^* = A\theta + B$, $A > 0$) |
| `A_se`, `B_se` | float | Delta-method standard errors from the implicit function theorem (Ogasawara, 2001), or NaN when unavailable |
| `vcov` | ndarray (2, 2) / None | Asymptotic covariance matrix of `(A, B)`, or `None` when `se=None` on a side with free common-item parameters, all common parameters are fixed, or $H_{cc}$ is not positive definite |
| `criterion` | float | Minimized criterion value at `(A, B)` |
| `method` | `"stocking-lord"` / `"haebara"` | Criterion used |
| `symmetric` | bool | Whether the symmetric criterion was minimized |
| `converged` | bool | Whether `grad_norm <= tol` was met |
| `n_iter` | int | Number of Newton iterations taken |
| `grad_norm` | float | $\|\nabla F(A, B)\|_\infty$ at `(A, B)` |
| `common_items` | tuple[str, ...] | Source item names used as common items |
| `items` | DataFrame | All source items transformed onto the target scale (`item`, `model`, `common`, internal slope–intercept parameters and display IRT thresholds `b` / `bk`); directly usable with `Model.fix_from()` |
| `transform_theta(theta)` | scalar / ndarray / Series / DataFrame | Transform source-scale ability scores via $\theta^* = A\theta + B$ (`theta` and all `pv1..` columns of a scoring DataFrame; `se` $\mapsto A\,\mathrm{se}$, excluding linking error) |
| `summary(*, digits=4)` | str | Readable plain-text summary (Jupyter displays HTML via `_repr_html_()`) |

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
Quadrature.default(n_factors=1, *, lim=(-6.0, 6.0))
```

An equally spaced quadrature grid (endpoints included), extended to a Cartesian product grid in
2–3 dimensions. `Quadrature.default(n_factors)` matches mirt's defaults: 61 points for 1 factor,
31 per dimension ($31^2 = 961$) for 2 factors, and 15 per dimension ($15^3 = 3{,}375$) for 3
factors on $[-6, 6]$. `nodes()` returns the 1-D axis of shape `(n_points,)`; `grid(n_factors)`
returns the product grid of shape `(n_points**n_factors, n_factors)`. At least 3 points per
dimension; limits must be finite with `lower < upper`.

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
