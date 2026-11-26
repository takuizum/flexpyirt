# API reference

English | [日本語](reference.ja.md)

The public API of flexpyirt (everything importable with `from flexpyirt import ...`) and of its
companion package flexpyctt. For the workflow see the [user guide](guide.md); for the formulas and
algorithms see [Theory](theory.md). The docstrings carry the same information plus the formulas
and sources (`help(flexpyirt.fit)` and so on).

**flexpyctt** holds the tools that need no fitted model: [correlations](#correlations) and the
model-free part of [number of dimensions](#number-of-dimensions) (each entry below says which
package it belongs to). It depends on NumPy, SciPy and pandas only, so it can be installed without
JAX (`pip install flexpyctt`) and needs no 64-bit setting; installing flexpyirt installs it too.
Import its functions with `from flexpyctt import ...`. flexpyirt re-exports them under the same
names, except `polychoric`: `flexpyirt.polychoric` is a JAX implementation of the same estimator.
flexpyctt's functions take `rng` (an integer seed or a `numpy.random.Generator`) where they draw
random numbers; flexpyirt's take a JAX PRNG `key`.

Apart from the first one or two, arguments of public functions are keyword-only (after `*`).

- Fitting: [`fit`](#fit), [`FitResult`](#fitresult)
- Scale linking: [`link`](#link), [`LinkResult`](#linkresult)
- Correlations (flexpyctt): [`polychoric`](#polychoric), [`PolychoricResult`](#polychoricresult)
- Number of dimensions (flexpyctt, except `model_parallel_analysis` and `dimensionality`): [`parallel_analysis`](#parallel_analysis), [`velicer_map`](#velicer_map), [`empirical_kaiser`](#empirical_kaiser), [`efa`](#efa), [`hull`](#hull), [`comparison_data`](#comparison_data), [`model_parallel_analysis`](#model_parallel_analysis), [`detect`](#detect), [`exploratory_detect`](#exploratory_detect), [`dimtest`](#dimtest), [`dimensionality`](#dimensionality) and their results
- Information and reliability: [`item_information`](#item_information), [`test_information`](#test_information), [`expected_score`](#expected_score), [`reliability`](#reliability)
- Summed scores: [`summed_score_table`](#summed_score_table), [`summed_score_distribution`](#summed_score_distribution)
- Classification: [`classification_accuracy`](#classification_accuracy), [`ClassificationResult`](#classificationresult)
- Model: [`Model`](#model)
- Item models: [`TwoPL`](#twopl), [`Rasch`](#rasch), [`ThreePL`](#threepl), [`FourPL`](#fourpl), [`GRM`](#grm), [`GPCM`](#gpcm), [`PCM`](#pcm), [`Nominal`](#nominal), [user-defined item models](#user-defined-item-models)
- Priors: [`NormalPrior`](#normalprior), [`LogNormalPrior`](#lognormalprior), [`BetaPrior`](#betaprior)
- Latent distribution: [`Normal`](#normal)
- Settings: [`BAEM`](#baem), [`Quadrature`](#quadrature)
- Exceptions and warnings: [`flexpyirt.errors`](#flexpyirterrors)

---

## Fitting

### `fit`

```text
fit(spec, data, *, group=None, table=None, item_names=None, method=None,
    quadrature=None, se="oakes", weights=None) -> FitResult
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
| `se` | `"oakes"` / `"hessian"` / `"sandwich"` / `"crossprod"` / `None` | Standard-error method: the inverse observed information (`"oakes"`, `"hessian"`), the inverse cross-product of the per-pattern scores (`"crossprod"`, mirt `SE.type = "crossprod"`) or the robust sandwich covariance (`"sandwich"`, mirt `SE.type = "sandwich"`). All include the prior Hessian under MAP. `"sandwich"` and `"crossprod"` raise `SpecificationError` with `DavidianCurve` or `NormalMixture`, for bifactor and two-tier models, and together with `weights`. `None` skips it |
| `weights` | array (one number per row), optional | Sampling (survey) weights, finite and non-negative (mirt's `survey.weights`). The estimates maximize $\sum_i w_i\log L_i$. Weights are used as given (not rescaled); rows with weight 0 are ignored; integer weights are the same as repeating rows. `se="oakes"` / `"hessian"` then invert the weighted information, which is valid for frequency weights only, not for sampling weights (a `FlexpyirtWarning` says so) |

**Returns**: [`FitResult`](#fitresult)

**Raises**: `PrecisionError` (JAX 64-bit disabled), `SpecificationError` (invalid specification
or data), `IdentificationError` (not identified). All are raised before estimation starts.

**Warns**: `ConvergenceWarning` (`max_iter` reached, or SQUAREM fell back to plain EM frequently),
`HeywoodWarning` (free asymptote within `1e-4` of its bound or free slope $|a| > 20$, or a
degenerate `NormalMixture` component),
`FlexpyirtWarning` (information matrix not positive definite, an anchor that is a single
parameter, or `weights` together with `se="oakes"` / `"hessian"`).

### `FitResult`

Returned by `fit()`. Immutable.

| Attribute | Type | Description |
|---|---|---|
| `items` | DataFrame | One row per item (per group × item, with a `group` column, for multiple groups): `model`, internal parameters (`a` or `a1`…`aD`, `d` / `d1`…, `g`, `u`), `<param>_se`, and for unidimensional models display `b` / `bk` ($-d/a$, GRM $-d_k/a$, GPCM/PCM $-(d_k-d_{k-1})/a$ with $d_0=0$) and delta-method `b_se` / `bk_se` (from `vcov` via JAX autodiff; NaN when `se=None` or all parameters entering a threshold are fixed) |
| `latent` | DataFrame | One row per group: `group`, `mean`, `var`, `mean_se`, `var_se` for 1 factor, or `group`, `mean_1`…`mean_D`, `var_1`…`var_D`, `cor_12`… and their `_se` columns for 2–3 factors (SEs of fixed values are NaN); for `EmpiricalHistogram`, `group`, `mean`, `var`, `sd`, `skewness`, `kurtosis` (excess) of each histogram; for `NormalMixture`, `mean`, `var`, the raw shape `rmean_c`, `rvar_c`, `logit_c` and their SEs, then the standardized components `pi_c`, `mean_c`, `var_c` ($c = 1..k$) with delta-method `_se` columns |
| `latent_weights` | DataFrame / None | `EmpiricalHistogram` only: `group`, `node`, `weight` (one row per group × quadrature node); None for `Normal` |
| `parameters` | DataFrame | The validated ParameterTable plus `estimate` (the estimate, or the fixed value) and `se` |
| `loglik` | float | Marginal log-likelihood at the estimates (likelihood only, excluding priors) |
| `log_posterior` | float | Log posterior ($\ell(\hat\eta) + \log\pi(\hat\eta)$) at the estimates; equals `loglik` without priors |
| `aic`, `bic` | float | $-2\ell + 2k$ and $-2\ell + k\log N$ ($k$ = `n_params`, $N$ = `n_obs`, or the sum of the weights for a fit with `weights`); emits `FlexpyirtWarning` when priors are active |
| `n_params` | int | Number of free parameters (an equality constraint counts once) |
| `n_obs` | int | Number of respondents (rows with a positive weight for a fit with `weights`) |
| `converged` | bool | Whether the convergence criterion was met |
| `n_iter` | int | Number of accepted EM (or SQUAREM) iterations |
| `n_em_evaluations` | int | Total number of EM map evaluations (E-step + M-step); equals `n_iter` for plain EM, and 2 or 3 per accepted iterate for SQUAREM |
| `n_safeguards` | int | Number of SQUAREM iterations on which the monotonicity/feasibility safeguard intervened (0 for plain EM) |
| `history` | DataFrame | `iter`, `loglik`, `log_posterior` (at the parameters entering the iteration; non-decreasing across iterations), `max_change`, `em_evaluations` (cumulative EM map evaluations), `safeguard` (whether the SQUAREM safeguard fired) |
| `vcov` | ndarray / None | Covariance matrix of the free parameters (in `free_id` order); None if `se=None` |
| `se_method` | str / None | `"oakes"` / `"hessian"` / `"sandwich"` / `"crossprod"` / None |

A result fitted with `weights` shows the sum of the weights in `summary()`. Its `item_fit()`,
`m2()`, `local_dependence()`, `jsi()` and `dif()` raise `SpecificationError` (these statistics are
defined for unweighted data), as does `model_parallel_analysis()`; `score()` works as usual.

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
are dropped from the likelihood (and from $I(\theta)$ in WLE) for every method except `"EAPsum"`.

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / polars DataFrame / 2-D array | Responses on the fitted items (need not be the fitting data) |
| `method` | `"EAP"` / `"MAP"` / `"ML"` / `"WLE"` / `"PV"` / `"EAPsum"` | Scoring method (default `"EAP"`): EAP (posterior mean & SD on the fitted quadrature grid; Bock & Mislevy, 1982), MAP (posterior mode with continuous normal prior $N(\mu_g,\sigma_g^2)$ & observed curvature SE), ML (maximum likelihood & observed-information SE; $-\infty$/$+\infty$ with SE NaN for patterns with no finite maximum, NaN for all-missing), WLE (Warm, 1989; Firth, 1993: solves $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta))=0$, finite on extreme patterns, NaN for all-missing, SE from observed WLE curvature), or PV (`n_draws` independent posterior draws via piecewise-linear inverse-CDF interpolation on a 501-node grid; Mislevy, 1991), or EAPsum (posterior mean & SD given the summed score, looked up in [`summed_score_table`](#summed_score_table) of the respondent's group; NaN for a row with any missing response; unidimensional models with `Normal` latent distributions only) |
| `group` | array, optional | Required for a multiple-group model: each row's group label |
| `item_names` | list[str], optional | Column names when `data` is an array |
| `key` | `jax.Array`, optional | Explicit JAX PRNG key (from `jax.random.key()` or `jax.random.PRNGKey()`). Required for `method="PV"`; must be `None` for other methods |
| `n_draws` | int | Number of plausible values per respondent when `method="PV"` (default `5`, must be `>= 1`) |

**Returns**: `DataFrame` with one row per respondent (index follows `data` when it is a
DataFrame): for unidimensional models, columns `theta`, `se` for `"EAP"`, `"MAP"`, `"ML"`, `"WLE"`, `"EAPsum"`
and `pv1`…`pv{n_draws}` for `"PV"`; for multidimensional models, columns `theta_<factor>` and `se_<factor>` in declared factor
order (`"PV"`: `pv<m>_<factor>`).

**Raises**: `SpecificationError` if `method` is unknown, `method="MAP"` with an `EmpiricalHistogram`,
`DavidianCurve` or `NormalMixture` latent distribution, `key` is missing or invalid for `"PV"`, `key` is supplied for
a deterministic method, `n_draws` is not a positive integer, or `method="EAPsum"` with a
multidimensional model or a latent distribution other than `Normal`.

#### `FitResult.item_fit`

```text
item_fit(data, *, min_cell=1.0, group=None, item_names=None) -> DataFrame
```

Orlando–Thissen S-X2 item fit ([guide section 5](guide.md#item-fit-s-x2)). For each item,
observed and model-implied counts of each category at summed scores $1,\dots,S-1$ are compared;
the expected counts use the Lord–Wingersky recursion with the item left out and the fitted latent
weights on the fitted quadrature grid. Sparse cells are collapsed exactly as mirt's `itemfit()`
(`min_cell` is mirt's `mincell`), and $df_j$ = kept cells − rows − free parameters of the item −
free latent parameters (mirt's convention). For a confirmatory multidimensional model
(`with_factors()`, single group) the recursion runs at every node of the fitted product grid with
the multivariate-normal weights, and the free factor correlations count as latent parameters.

| Argument | Description |
|---|---|
| `data` | Complete responses on the fitted items (typically the fitting data) |
| `min_cell` | Minimum expected count per cell before collapsing (default `1.0`) |
| `group` | Group labels (required for a multiple-group model; S-X2 is computed within each group) |
| `item_names` | Column names when `data` is an array |

**Returns**: `DataFrame` with columns `item`, `s_x2`, `df`, `rmsea`
($\sqrt{\max(0, X^2/df-1)/(N-1)}$) and `p`; items whose collapsed table leaves `df <= 0` get
`df = 0` and NaN statistics. A multiple-group model adds a leading `group` column (one row per
group and item).

**Raises**: `SpecificationError` for bifactor, two-tier, multiple-group multidimensional,
`EmpiricalHistogram`, `DavidianCurve`, or `NormalMixture` models, missing responses, an item with an unobserved category (within a group),
or a non-positive `min_cell`.

#### `FitResult.m2`

```text
m2(data, *, item_names=None, ci=0.90) -> DataFrame
```

Limited-information goodness-of-fit statistic $M_2$ / $M_2^*$ of Maydeu-Olivares & Joe (2005, 2006)
([guide section 5](guide.md#limited-information-fit-m2)), computed on the $J$ item-score means and
$\binom{J}{2}$ pairwise cross-products (matching mirt's default `M2(type = "M2*")`; identically
$M_2$ for dichotomous items). The Jacobian $\Delta_2 = \partial\pi_2/\partial\eta^\top$ includes
every free item and latent parameter (shared parameters once) via JAX autodiff. Confirmatory
multidimensional models (`with_factors()`) are integrated over the fitted product grid, and
$\Delta_2$ has a column for every free factor correlation (mirt's `M2()` omits the latent columns,
so its value is slightly larger).

| Argument | Description |
|---|---|
| `data` | Complete responses on the fitted items (typically the fitting data) |
| `item_names` | Column names when `data` is an array |
| `ci` | Confidence level strictly in `(0, 1)` for the RMSEA confidence interval (default `0.90`) |

**Returns**: One-row `DataFrame` with columns `m2`, `df` ($J + \binom{J}{2} - k$), `p`, `rmsea`,
`rmsea_<lo>`, `rmsea_<hi>` (named from `ci`, e.g. `rmsea_5`, `rmsea_95` for `0.90`), `srmsr`,
`tli`, `cfi`. When `df <= 0` or $\Delta_2$ is rank-deficient (the model is not locally identified
from first- and second-order moments), a `FlexpyirtWarning` is emitted and `m2`, `p`, RMSEA, its
CI, `tli` and `cfi` are `NaN` (`df` and `srmsr` remain finite). Polytomous models return `NaN` for
`tli` and `cfi` (the independence null model is dichotomous only); if the dichotomous null $M_2$ is not greater than the fitted $M_2$,
a `FlexpyirtWarning` is emitted and `tli`/`cfi` are `NaN`.

**Raises**: `PrecisionError` if JAX 64-bit is disabled; `SpecificationError` for bifactor, two-tier,
multiple-group, `EmpiricalHistogram`, `NormalMixture`, or MAP (with priors) models, missing responses, invalid
`ci`, or fewer than 2 items or respondents.

#### `FitResult.local_dependence`

```text
local_dependence(data, *, group=None, item_names=None) -> LocalDependenceResult
```

Local-dependence statistics for every item pair
([guide section 18](guide.md#18-local-dependence-and-the-dimensionality-report)):
Yen's (1984) Q3 (correlation of residuals $x_{ij} - E(X_j \mid \hat\theta_i)$ with EAP
$\hat\theta$), Q3\* $=$ Q3 $-$ mean Q3 (Christensen, Makransky & Horton, 2017), and Chen &
Thissen's (1997) LD $X^2$ / $G^2$ against the model-implied table on the fitted grid, with
$df = (K_i-1)(K_j-1)$, $p$-values and a signed Cramér's $V$, as mirt's
`residuals(type = "Q3" / "LD" / "LDG2")`. Missing responses are allowed (pairwise).

The expected tables and the EAP use the fitted latent distribution on the fitted grid: `Normal`,
`EmpiricalHistogram`, `DavidianCurve` or `NormalMixture`, and the multivariate Normal on the
product grid (with the vector EAP) for models with 2 or 3 factors. In a multiple-group model
`group` is required and every statistic is computed within a group, from that group's
respondents, item parameters and latent distribution.

For bifactor and two-tier models (`with_bifactor()`, `with_two_tier()`) the expected tables are
summed on the reduced grids instead of the full product grid: over the primary factors and the
shared specific factor for a pair inside one specific factor, and over the primary factors only,
with each item's specific factor integrated out first, for any other pair. Q3 uses the EAP of all
factors. The results equal mirt's `residuals()` on a `bfactor()` fit with the same grid (mirt
builds the product grid). A pair inside a specific factor is tested for dependence beyond that
factor.

Two differences from mirt's defaults: for `dentype = "EH"` / `"Davidian-k"` fits mirt takes a
standard Normal prior unless called as `residuals(m, type = "LD", use_dentype_estimate = TRUE)`
and `residuals(m, type = "Q3", Theta = fscores(m, use_dentype_estimate = TRUE))`; and mirt's Q3
scores on `fscores()`'s own grid (121 / 61 / 31 points per dimension for 1 / 2 / 3 factors),
flexpyirt on the fitted grid.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | DataFrame or array-like | | Responses on the fitted items; missing as NaN |
| `group` | array-like | `None` | Group labels (required for a multiple-group model) |
| `item_names` | list of str | `None` | Column names when `data` is an array |

**Returns**: [`LocalDependenceResult`](#localdependenceresult)

**Raises**: `PrecisionError`; `SpecificationError` for invalid data or group labels.

#### `FitResult.jsi`

```text
jsi(data, *, item_names=None, method="refit", newton_steps=1, tol=1e-8, max_iter=100) -> JSIResult
```

Edwards, Houts & Cai's (2018) jackknife slope index for a single-group unidimensional model with
a `Normal` latent distribution and at least three items
([guide section 18](guide.md#18-local-dependence-and-the-dimensionality-report)). For each item
$j$ the model is re-estimated without item $j$ (its responses treated as missing, so the latent
distribution, constraints and priors stay as specified), and for every other item $i$

$$\mathrm{JSI}_{ji} = \frac{\hat a_i - \hat a_i^{(-j)}}{\mathrm{SE}(\hat a_i^{(-j)})}.$$

Items without a free slope give NaN. `data` is the fitting data.

| Argument | Meaning |
|---|---|
| `method` | `"refit"` (default): Newton–Raphson from the full estimates until the largest step is below `tol`, SE from the refit, as mirt's `residuals(type = "JSI")`. `"newton"`: `newton_steps` Newton steps (a Taylor approximation). `"one_step"`: one step with the full model's covariance matrix and the full model's SE in the denominator, about $J$ E-steps in all |
| `newton_steps` | Steps for `"newton"` |
| `tol`, `max_iter` | Convergence of `"refit"` |

The approximations agree with the refit to about 0.02 from 15 items on; with fewer than 10 items
they warn (`FlexpyirtWarning`), because removing one item changes the estimates too much.

**Returns**: [`JSIResult`](#jsiresult)

JSI is defined only for single-group unidimensional models; its multidimensional /
multiple-group extension has not been studied and there is no reference implementation, so it is
not available for other models.

**Raises**: `PrecisionError`; `SpecificationError` for multiple-group, multidimensional
(including bifactor and two-tier) or non-`Normal` latent models, fewer than three items, invalid options, or `"one_step"` on a fit without `se="oakes"` or `se="hessian"`.

**Warns**: `ConvergenceWarning` if a refit does not converge.

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
top-level `"format": "flexpyirt.FitResult"`, `"schema_version": 7` (files with versions 1–6 are
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
model/prior type (register a user-defined item model with
[`register_item_model`](#flexpyirtitemsregister_item_model) first).

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
version, or a registered type name is unknown (including a user-defined item model that has not
been registered in this session), or internal consistency checks fail;
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
model.fix_from(bank, *, items=None, parameterization="slope_intercept") -> Model
```

Fix item parameters at item-bank values (FIPC). Combine with `.with_latent(Normal.free())` to
estimate the latent distribution.

| Argument | Description |
|---|---|
| `bank` | A DataFrame with an `item` column and parameter columns: `a`, `d` (GRM/GPCM/PCM: `d1`, `d2`, …) for `"slope_intercept"`; `a`, `b` (GRM/GPCM/PCM: `b1`, …) for `"irt"`. Rasch and PCM items may omit `a`. Asymptotes `g` (3PL/4PL) and `u` (4PL) are passed unchanged in both parameterizations; `u` may be omitted for 3PL items |
| `items` | Items to fix (a list of names or a `str -> bool` function). Default: every bank item that is also in the model |
| `parameterization` | `"slope_intercept"` or `"irt"` (converted with $d = -ab$, or $d_k = -a\sum_{c=1}^{k} b_c$ for GPCM/PCM) |

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
`COV = F1*F2`) or fixed at `0` when `correlated=False`. With `with_groups()` (either order), each
group's `Normal` applies to every factor ([guide section 14](guide.md#14-confirmatory-multidimensional-models)).
`with_latent()` and `EmpiricalHistogram` are not supported for multidimensional models
(`SpecificationError`).

#### `Model.with_bifactor`

```text
model.with_bifactor(*, specific, general="general") -> Model
```

Declare a bifactor model ([guide section 14](guide.md#bifactor-models)): a general factor named
`general` on which every item loads, and the specific factors of `specific` (a mapping from factor
name to items, in order). Each item loads on at most one specific factor, items under none load on
the general factor only, and each specific factor needs at least two items. Item parameters are
`a1` (general), `a2`…`aD` (specific; slopes on other specific factors fixed at `0`) and `d` (or
`d1`…), matching mirt's `bfactor()`. The factors are orthogonal: the latent rows are `mean_1`…`mean_D`
(fixed `0`) and `var_1`…`var_D` (fixed `1`). The E-step uses Gibbons–Hedeker dimension reduction
(one two-dimensional grid per specific factor; default 31 points on $[-6, 6]$). Raises
`SpecificationError` for groups, `with_latent()`, combining with `with_factors()`, Rasch / PCM
items, or free latent parameters in an edited table. Scoring supports `"EAP"` only.

#### `Model.with_two_tier`

```text
model.with_two_tier(*, primary, specific, correlated=True) -> Model
```

Declare a two-tier model ([guide section 14](guide.md#two-tier-models)): the primary factors of
`primary` and the specific factors of `specific` (both mappings from factor name to items, in
order; one or two primary factors). Every item loads on at least one primary factor and on at most
one specific factor, and each specific factor needs at least two items. Item parameters are
`a1`…`aP` (primary), `a(P+1)`…`aD` (specific; slopes an item does not load on are fixed at `0`)
and `d` (or `d1`…), matching mirt's `bfactor(data, model, model2 = )`. The latent rows are
`mean_1`…`mean_D` (fixed `0`), `var_1`…`var_D` (fixed `1`) and, for two primary factors, `cor_12`
(free when `correlated=True`); specific factors are orthogonal to every other factor. The E-step
integrates over the primary factors' product grid and one specific factor at a time (default 15
points per dimension on $[-6, 6]$ for two primary factors, 31 for one). Raises
`SpecificationError` for more than two primary factors, an item on no primary factor, groups,
`with_latent()`, combining with `with_factors()` or `with_bifactor()`, Rasch / PCM items, or free
latent means or variances in an edited table. Scoring supports `"EAP"` only.

#### `Model.with_latent`

```text
model.with_latent(dist) -> Model
```

Set the latent distribution of a single-group model to a `Normal`, `EmpiricalHistogram`,
`DavidianCurve`, or `NormalMixture` (unidimensional models only). Not available with groups
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

Attach a prior distribution `dist` (`NormalPrior`, `LogNormalPrior`, or `BetaPrior`) to internal parameter
`param` of the selected items across all groups (MAP estimation; [guide section 11](guide.md#11-priors-and-map-estimation)).
Priors are evaluated on the natural parameter scale without a Jacobian term. Fixed parameters
ignore their prior, and a parameter shared across groups (`free_id`) is counted once. Priors on
latent-distribution parameters (`"mean"`, `"var"`, `"phi_*"`, `"rmean_*"`, `"rvar_*"`,
`"logit_*"`, ...) raise `SpecificationError`.

| Argument | Description |
|---|---|
| `param` | Internal item parameter name (`"a"`, `"d"`, `"g"`, `"u"`, `"d1"`, …) |
| `dist` | `NormalPrior`, `LogNormalPrior`, or `BetaPrior` |
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
| `is_multidimensional` | Whether factors were declared with `with_factors()`, `with_bifactor()` or `with_two_tier()` |
| `n_factors` | Number of latent factors (`1`, `2`, or `3`; `1 + S` for a bifactor model, `P + S` for a two-tier model) |
| `bifactor` | Whether the model was declared with `with_bifactor()` |
| `n_primary_factors` | Number of primary factors of a model declared with `with_two_tier()` (`0` otherwise) |
| `quadrature_dimension` | Dimension of the grid the E-step integrates over (`2` for bifactor models, `P + 1` for two-tier models) |
| `latent_parameter_names()` | Names of the Normal latent parameters of each group |
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
GRM(n_categories)
```

Graded response model (Samejima, 1969) with categories `0..n_categories-1`:
$P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$. Internal parameters `a`,
`d1`…`d{K-1}`; display $b_k=-d_k/a$. Same as mirt's `itemtype = "graded"`. GRM items with the same
`n_categories` are estimated together.

### `GPCM`

```text
GPCM(n_categories)
```

Generalized partial credit model (Muraki, 1992) with categories `0..n_categories-1` (`n_categories >= 2`):
$P(y=k\mid\theta)=\exp(k\,a\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)$ with $d_0=0$ fixed.
Internal parameters `a`, `d1`…`d{K-1}` (no ordering constraint); display step difficulties
$b_k=-(d_k-d_{k-1})/a$ with $d_0=0$. Same as mirt's `itemtype = "gpcm"`. With `n_categories = 2` this
equals `TwoPL`. Shares `GPCMKernel` with `PCM` of the same `n_categories`.

### `PCM`

```text
PCM(n_categories)
```

Partial credit model (Masters, 1982) with categories `0..n_categories-1` (`n_categories >= 2`): GPCM with `a = 1`
fixed, $P(y=k\mid\theta)=\exp(k\theta+d_k)/\sum_{c=0}^{K-1}\exp(c\theta+d_c)$ with $d_0=0$ fixed.
Display step difficulties $b_k=-(d_k-d_{k-1})$. In a single-group model whose items are all `Rasch`
or `PCM`, the latent variance is estimated by default (same as mirt's `itemtype = "Rasch"` on
polytomous data). With `n_categories = 2` this equals `Rasch`. Shares `GPCMKernel` with `GPCM` of the same
`n_categories`.

### `Nominal`

```text
Nominal(n_categories)
```

Nominal response model (Bock, 1972) in Thissen, Cai & Bock's (2010) overall-slope ×
scoring-function parameterization with categories `0..n_categories-1` (`n_categories >= 2`):
$P(y=k\mid\theta)=\exp(a\,\mathrm{ak}_k\,\theta+d_k)/\sum_{c=0}^{K-1}\exp(a\,\mathrm{ak}_c\,\theta+d_c)$
with $\mathrm{ak}_0=0$, $\mathrm{ak}_{K-1}=K-1$, and $d_0=0$ fixed. Internal parameters `a` (or
`a1`…`aD` for multidimensional models), `ak1`…`ak{K-2}`, and `d1`…`d{K-1}` (all free, no ordering
constraint; no display `b` / `bk` columns). Same as mirt's `itemtype = "nominal"`. Fixing
$\mathrm{ak}_k=k$ reproduces `GPCM(n_categories)` identically (and with `n_categories = 2` there are no free `ak`
parameters, so `Nominal(2)` equals `GPCM(2)` / `TwoPL`). Uses `NominalKernel(n_categories)`.

### User-defined item models

An item model that is not built in can be written against two protocols in `flexpyirt.items`
and used in a `Model` like the classes above (see the [user guide](guide.md#20-user-defined-item-models)
for a worked example). Items whose kernels compare equal are estimated together.

**`flexpyirt.items.ItemKernel`** (the response function; must be hashable and compare by value:
use a frozen dataclass)

| Member | Required | Description |
|---|---|---|
| `n_factors` | yes | Number of latent factors $D$ |
| `with_factors(n_factors)` | yes | A copy of the kernel for `n_factors` factors |
| `param_names` | yes | Parameter names in order; the first $D$ are the slopes |
| `n_categories` | yes | Number of categories $K \ge 2$ (responses coded `0..K-1`) |
| `slope_params` | yes | The slopes, `param_names[:n_factors]` |
| `intercept_params` | yes | Parameters that carry the location (used by the identification checks; may be empty) |
| `log_prob(theta, params)` | yes | `(Q, K)` log-probabilities at nodes `theta` of shape `(Q, D)` for the parameter vector `params`; written with `jax.numpy`, differentiable twice; NaN outside the parameter domain |
| `irt_names`, `irt_dependency_indices`, `irt_thresholds(params)` | no | Display thresholds added to `FitResult.items` with delta-method SEs: their names, the indices of the parameters each depends on, and their values |

**`flexpyirt.items.ItemModel`** (the specification; use a frozen dataclass)

| Member | Description |
|---|---|
| `name` | Class attribute: the name in the `model` column and in JSON (not a built-in name) |
| `kernel` | The kernel with `n_factors == 1` |
| `default_free()` | Whether each parameter is estimated by default |
| `start_values(item, rest)` | Starting values from one item's observed responses and their rest scores |
| `bounds()` | `(lower, upper)` tuples of parameter bounds |

`default_free()`, `start_values()` and `bounds()` follow the unidimensional `param_names`; in a
model with $D$ factors their slope entry applies to every slope. `Model` checks these rules when
it is built (`SpecificationError`). Everything that evaluates `log_prob` works with a user-defined
model: estimation, standard errors, scoring, item fit, M2, local dependence, DIF, information,
summed scores, multiple groups, priors and multidimensional models. ML and WLE scoring assume
that a higher category means a higher $\theta$ for a positive slope. Not available:
`link()` and `Model.fix_from(..., parameterization="irt")` (`SpecificationError`), and the free
latent variance that a model of only `Rasch` / `PCM` items gets by default (use
`Model.with_latent(Normal(free_var=True))`).

#### `flexpyirt.items.register_item_model`

```text
register_item_model(cls, *, to_dict=None, from_dict=None) -> cls
```

Register a user-defined item-model class so that `FitResult.to_json()` can save fits that use it
and `FitResult.from_json()` can load them. The model is stored as
`{"type": cls.name, **to_dict(model)}` and rebuilt with `from_dict(fields)`. By default the class
must be a dataclass whose fields are `bool`, `int`, finite `float` or `str`. Registration lasts
for the Python session: register the class again before loading in a new one. Can be used as a
class decorator.

| Argument | Type | Description |
|---|---|---|
| `cls` | type | The item-model class; `cls.name` is the JSON type name |
| `to_dict` | callable, optional | `model -> dict` of JSON values (without `"type"`) |
| `from_dict` | callable, optional | `dict -> model`, the inverse of `to_dict` (pass both or neither) |

**Returns**: `cls`.

**Raises**: `SpecificationError` if `cls.name` is missing or built in, another class is registered
under the same name, only one of `to_dict` / `from_dict` is given, or the default encoding is used
for a class that is not a dataclass. Without registration, `to_json()` raises
`SpecificationError`.

---

## Priors

Frozen dataclasses representing item-parameter priors on the natural parameter scale without a
Jacobian term (as in mirt). Each validates its hyperparameters (`SpecificationError`), exposes
`support`, and provides `log_pdf(x) -> Array`.

### `NormalPrior`

```text
NormalPrior(mean, sd)
```

Normal prior on $(-\infty, \infty)$:
$\log\pi(x) = -\tfrac{1}{2}\log(2\pi) - \log\sigma - \frac{(x-\mu)^2}{2\sigma^2}$
(Mislevy, 1986; mirt `norm`). Requires finite `mean` and positive finite `sd`. Named `NormalPrior`
to avoid clashing with the latent distribution `Normal`.

### `LogNormalPrior`

```text
LogNormalPrior(mean_log, sd_log)
```

Log-normal prior on $(0, \infty)$ ($\log X \sim N(\mu, s^2)$):
$\log\pi(x) = -\log x - \log s - \tfrac{1}{2}\log(2\pi) - \frac{(\log x - \mu)^2}{2s^2}$
($-\infty$ for $x \le 0$; Mislevy, 1986; mirt `lnorm`). Requires finite `mean_log` and positive
finite `sd_log`. Commonly used for item slopes `a`.

### `BetaPrior`

```text
BetaPrior(a, b)
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

### `DavidianCurve`

```text
DavidianCurve(degree, *, mean=0.0, var=1.0, free_mean=False, free_var=False)
DavidianCurve.free(degree, *, mean=0.0, var=1.0)
```

Seminonparametric Davidian-curve latent density of polynomial degree `degree` in `1..10`
(Zhang & Davidian, 2001; Woods & Lin, 2009; [guide section 8](guide.md#davidian-curves)).
Evaluates $h_k(z;\boldsymbol\phi) = P_k(z;\boldsymbol\phi)^2\varphi(z)$ in normalized probabilists'
Hermite polynomials with unit-sphere coefficients parameterized by $k$ angles
`phi_1`…`phi_k` ($\boldsymbol\phi = \mathbf 0$ reproduces $N(\mu,\sigma^2)$ non-singularly),
analytically standardized by the exact continuous moments of $h_k$ so the continuous mean and
variance equal `mean` and `var` for every $\boldsymbol\phi$. Quadrature weights are normalized on
the grid, keeping EM monotone (non-decreasing) and Oakes / Hessian standard errors exact for all
free item and latent parameters. Single-group unidimensional models only. The Fisher information
is singular at $\boldsymbol\phi = \mathbf 0$ in `phi_1`, `phi_2` (in `phi_1` for `degree=1`), so
near-normal traits give unreliable angle standard errors, a non-standard likelihood-ratio test
against `Normal` (use AIC / BIC), and possibly several local maxima
([guide section 8](guide.md#davidian-curves)).

| Argument | Description |
|---|---|
| `degree` | Integer polynomial degree $k \in \{1,\dots,10\}$ (number of angle parameters `phi_1`…`phi_k`) |
| `mean`, `var` | Value (or starting value when free) of the continuous mean and positive variance |
| `free_mean`, `free_var` | Whether `mean` / `var` is estimated (same identification rules as `Normal`) |

**Raises**: `SpecificationError` for an invalid `degree`, `mean`, `var`, or flags, or when used
with `Model.with_groups()` or `Model.with_factors()`; `IdentificationError` when `free_mean` or
`free_var` lacks fixed intercepts or slopes.

### `NormalMixture`

```text
NormalMixture(n_components, *, mean=0.0, var=1.0, free_mean=False, free_var=False,
              means=None, variances=None, weights=None)
NormalMixture.free(n_components, *, mean=0.0, var=1.0, means=None, variances=None, weights=None)
```

Finite normal-mixture latent density with $k$ = `n_components` in `2..10` components and one
common set of item parameters (Mislevy, 1984; [guide section 8](guide.md#normal-mixtures)). The
shape uses mirt's `dentype = "mixture-k"` class parameterization: raw component 1 is $N(0,1)$ and
components $c \ge 2$ have raw means `rmean_c`, raw variances `rvar_c` and logits
`logit_c` $=\log(\pi_c/\pi_1)$. The mixture is analytically standardized by its exact moments so
its continuous mean and variance equal `mean` and `var` for every shape; each component is
normalized on the grid and then mixed. `FitResult.latent` reports the standardized components
`pi_c`, `mean_c`, `var_c` with delta-method SEs. Oakes / Hessian standard errors cover item and
latent parameters. Single-group unidimensional models only; equispaced or Gauss–Hermite grids.
Components are exchangeable (label switching) and the likelihood can have several local maxima:
refit from a few starting shapes and keep the highest `loglik`. `score(method="MAP")`,
`item_fit()` and `m2()` raise `SpecificationError`.

| Argument | Description |
|---|---|
| `n_components` | Integer number of components $k \in \{2,\dots,10\}$ |
| `mean`, `var` | Value (or starting value when free) of the continuous mixture mean and positive variance |
| `free_mean`, `free_var` | Whether `mean` / `var` is estimated (same identification rules as `Normal`) |
| `means`, `variances`, `weights` | Optional length-$k$ starting shape (component means, positive variances, positive weights summing to 1) in any affine scale; defaults `(0, 1.5, -1.5, 3, -3, ...)`, all `1`, and `1/k` |

**Raises**: `SpecificationError` for an invalid `n_components`, `mean`, `var`, flags or starting
shape, or when used with `Model.with_groups()` or `Model.with_factors()`; `IdentificationError`
when `free_mean` or `free_var` lacks fixed intercepts or slopes.

---

## Scale linking

### `link`

```text
link(source, target, *, items=None, method="stocking_lord", symmetric=False,
     transformation="diagonal", nodes=None, weights=None, source_group=None,
     target_group=None, tol=1e-10, max_iter=200) -> LinkResult
```

Link a separately calibrated `source` (new) scale onto `target` (base) via common
items by finding linear constants $(A, B)$ ($A > 0$) for $\theta^* = A\theta + B$
([guide section 15](guide.md#15-scale-linking), [Theory](theory.md)). Validated against R for
2PL, 3PL, 4PL, GRM, GPCM, nominal, mixed 2PL + GPCM and mixed 2PL + nominal common items; Rasch
and PCM share these kernels and are covered by unit tests only. Two confirmatory models with the
same number of factors $D > 1$ (`Model.with_factors`; factors matched by position) are linked by
$\boldsymbol\theta^* = \mathbf A\boldsymbol\theta + \boldsymbol\beta$ with the same criteria on
the product grid (validated against R for two- and three-factor 2PL and two-factor GRM models).
Derivatives for Newton optimization and delta-method standard
errors (Ogasawara, 2001) come from JAX automatic differentiation. SEs assume independent
calibrations (the joint covariance is used when one multiple-group fit is both `source` and
`target`), hold the quadrature weights fixed, and are not reported without convergence.

| Argument | Type | Description |
|---|---|---|
| `source` | `FitResult` | Fitted model on the source (new) scale |
| `target` | `FitResult` | Fitted model on the target (base) scale, with the same number of factors |
| `items` | iterable[str] / Mapping[str, str], optional | Common items (default `None`: all items sharing the same name). Pass `{source_item: target_item}` when names differ. Common pairs must share the same kernel and category count |
| `method` | `"stocking_lord"` / `"haebara"` | Criterion: Stocking & Lord (1983) squared TCC difference (default) or Haebara (1980) squared category-curve differences (polytomous forms: Kim & Lee, 2006) |
| `symmetric` | bool | If `True`, add the reverse-direction squared difference (Kim & Lee, 2006); default `False` |
| `transformation` | `"diagonal"` / `"orthogonal"` / `"oblique"` | Family of $\mathbf A$ for $D > 1$ factors (no effect for one factor): `"diagonal"` (default) $\operatorname{diag}(A_m)$, $A_m > 0$, the transformation between two calibrations of the same confirmatory structure; `"orthogonal"` a dilation times a rotation, $k\,\mathbf R$ (Li & Lissitz, 2000); `"oblique"` any matrix with $\det\mathbf A > 0$ (Oshima, Davey & Lee, 2000). Reflections are excluded |
| `nodes` | array-like, optional | Quadrature nodes $\theta_q$, 1-D, or of shape `(Q, D)` for $D > 1$ factors (default: target group's fitted grid, and source group's grid for the reverse direction when `symmetric=True`) |
| `weights` | 1-D array-like, optional | Non-negative weights normalized to sum to 1 (default: fitted latent distribution weights, or uniform when `nodes` is given without `weights`) |
| `source_group`, `target_group` | str, optional | Group name when `source` or `target` is a multiple-group `FitResult` |
| `tol` | float | Convergence tolerance on $\|\nabla F(A, B)\|_\infty$ (default `1e-10`) |
| `max_iter` | int | Maximum Newton iterations (default `200`) |

**Returns**: [`LinkResult`](#linkresult)

**Raises**: `PrecisionError` (JAX 64-bit disabled), `SpecificationError` (non-`FitResult` input,
different numbers of factors, bifactor or two-tier input, multiple-group input without selecting a group, no common items,
mismatched item kernels/categories on common items, invalid options/nodes/weights).

**Warns**: `ConvergenceWarning` if `max_iter` is reached before `grad_norm <= tol`.

### `LinkResult`

Returned by `link()`. Immutable.

| Attribute / method | Type | Description |
|---|---|---|
| `A`, `B` | float / ndarray | Linear linking constants ($\theta^* = A\theta + B$, $A > 0$). For $D > 1$ factors, the `(D, D)` matrix $\mathbf A$ and the `(D,)` vector $\boldsymbol\beta$ |
| `A_se`, `B_se` | float / ndarray | Delta-method standard errors from the implicit function theorem (Ogasawara, 2001), or NaN when unavailable; same shapes as `A`, `B` (0 for the structural zeros of a diagonal $\mathbf A$) |
| `vcov` | ndarray / None | Asymptotic covariance matrix of `(A, B)`, `(2, 2)`; for $D > 1$ factors the `(D*D + D, D*D + D)` covariance of `(A.ravel(), B)`, of rank at most the number of free transformation parameters. `None` when `se=None` on a side with free common-item parameters, all common parameters are fixed, or $H_{cc}$ is not positive definite |
| `criterion` | float | Minimized criterion value at `(A, B)` |
| `method` | `"stocking_lord"` / `"haebara"` | Criterion used |
| `symmetric` | bool | Whether the symmetric criterion was minimized |
| `transformation` | `"diagonal"` / `"orthogonal"` / `"oblique"` | Family of $\mathbf A$ (always `"diagonal"` for one factor) |
| `converged` | bool | Whether `grad_norm <= tol` was met |
| `n_iter` | int | Number of Newton iterations taken |
| `grad_norm` | float | $\|\nabla F(A, B)\|_\infty$ at `(A, B)` |
| `common_items` | tuple[str, ...] | Source item names used as common items |
| `items` | DataFrame | All source items transformed onto the target scale (`item`, `model`, `common`, internal slope–intercept parameters and display IRT thresholds `b` / `bk`); directly usable with `Model.fix_from()` |
| `transform_theta(theta)` | scalar / ndarray / Series / DataFrame | Transform source-scale ability scores via $\theta^* = A\theta + B$ (`theta` and all `pv1..` columns of a scoring DataFrame; `se` $\mapsto A\,\mathrm{se}$, excluding linking error). For $D > 1$ factors: the `theta_<factor>` / `pv<m>_<factor>` columns or an array whose last axis has length $D$; `se_<factor>` is rescaled under a diagonal $\mathbf A$ and NaN otherwise |
| `summary(*, digits=4)` | str | Readable plain-text summary (Jupyter displays HTML via `_repr_html_()`) |

## Correlations

Package: flexpyctt (`from flexpyctt import polychoric`).

### `polychoric`

```text
polychoric(data, *, smooth=True, zero_cell_correction=0.5, tol=1e-10,
           max_iter=100) -> PolychoricResult
```

Polychoric correlation matrix (tetrachoric for binary items) of ordinal responses, by two-step
maximum likelihood ([guide section 16](guide.md#16-polychoric-correlations)): each item's
thresholds come from its marginal proportions, then each pair's correlation maximizes the
likelihood of the pair's table under a discretized bivariate normal (Olsson, 1979). The
bivariate normal probabilities use Owen's (1956) $T$ function. Agrees with lavaan's `lavCor` on
ordered variables.

`flexpyirt.polychoric` has the same arguments and result and computes the same estimator with JAX
(bivariate normal probabilities by Genz's, 2004, algorithm); it also raises `PrecisionError` when
JAX's 64-bit mode is disabled. The two agree to about $10^{-9}$.

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / 2-D array | One row per respondent, one column per item. Ordered category codes (any numbers; ascending order is the category order); missing as NaN. Each item needs at least two observed categories; unobserved codes are not categories |
| `smooth` | bool | Smooth a matrix that is not positive definite (psych's `cor.smooth` rule) with a `NotPositiveDefiniteWarning` (default `True`) |
| `zero_cell_correction` | float | For a 2 x 2 table with exactly one empty cell, add this to the diagonal containing the empty cell and subtract it from the other diagonal, keeping the margins (lavaan's default rule). Without it the estimate is $\pm 1$. `0` disables it (default `0.5`) |
| `tol` | float | Convergence criterion on the gradient per observation (default `1e-10`) |
| `max_iter` | int | Maximum Newton iterations per pair (default `100`) |

Missing responses use available cases: thresholds use every observed response to the item, and a
pair's table uses respondents who answered both items.

**Returns**: [`PolychoricResult`](#polychoricresult)

**Raises**: `SpecificationError` (non-numeric data, fewer
than two items, an item with one observed category, a pair with no respondent answering both, or
one item constant among those who did).

**Warns**: `HeywoodWarning` (a pair's likelihood is largest at $|\rho| \to 1$),
`ConvergenceWarning` (a pair did not converge), `NotPositiveDefiniteWarning` (the matrix was
smoothed).

### `PolychoricResult`

Returned by `polychoric()`. Immutable.

| Attribute | Type | Description |
|---|---|---|
| `correlation` | DataFrame | Item x item correlation matrix (smoothed when `smooth=True` and needed) |
| `unsmoothed_correlation` | DataFrame | The pairwise estimates before smoothing |
| `thresholds` | dict[str, ndarray] | Per item, the $K - 1$ thresholds $\tau_k = \Phi^{-1}(P(Y \le k))$ of its $K$ observed categories |
| `categories` | dict[str, ndarray] | Per item, the observed response values in ascending order |
| `n_obs` | DataFrame | Respondents with both responses observed (diagonal: responses to the item) |
| `converged`, `at_boundary`, `zero_cell_corrected` | DataFrame (bool) | Per pair: convergence, maximum at $\lvert\rho\rvert \to 1$ (the estimate is then $\pm\tanh 8$), whether `zero_cell_correction` was applied |
| `smoothed` | bool | Whether smoothing changed the matrix |
| `min_eigenvalue` | float | Smallest eigenvalue of `unsmoothed_correlation` |

## Number of dimensions

These answer "how many dimensions?" as diagnostics; combine several rather than trusting one
([guide section 17](guide.md#17-number-of-dimensions)).

Package: flexpyctt, except [`model_parallel_analysis`](#model_parallel_analysis),
[`dimensionality`](#dimensionality), `LocalDependenceResult` and `JSIResult`, which use a fitted
model and belong to flexpyirt.

### `parallel_analysis`

```text
parallel_analysis(data, *, rng, correlation="polychoric", eigen_type="pca",
                  reference="permutation", n_replications=100, quantile=0.95,
                  zero_cell_correction=0.5) -> ParallelAnalysisResult
```

Horn's (1965) parallel analysis: the leading eigenvalues of the observed correlation matrix that
exceed the `quantile` (Glorfeld, 1995) of eigenvalues from reference data without common factors
are counted.

| Argument | Type | Description |
|---|---|---|
| `data` | DataFrame / 2-D array | Respondents x items (ordered codes, or continuous scores for `correlation="pearson"`); NaN is missing |
| `rng` | int / `numpy.random.Generator` | Seed or generator for the reference data, e.g. `rng=0` |
| `correlation` | `"polychoric"` / `"pearson"` | Correlation of the data and of permutation replications (both smoothed when not positive definite) |
| `eigen_type` | `"pca"` / `"smc"` | Eigenvalues of the correlation matrix, or of the reduced matrix with squared multiple correlations on the diagonal |
| `reference` | `"permutation"` / `"normal"` | Permute each item independently (keeps the marginals; Garrido, Abad & Ponsoda, 2013), or i.i.d. normal data of the same size (Pearson) |
| `n_replications` | int | Number of reference data sets (default `100`) |
| `quantile` | float / None | Quantile in (0, 1) of the reference eigenvalues (default `0.95`); `None` uses the mean |
| `zero_cell_correction` | float | Passed to `polychoric()` |

**Returns**: [`ParallelAnalysisResult`](#parallelanalysisresult)

**Raises**: `SpecificationError` (invalid options or data, fewer than 3 items).

**Warns**: `FlexpyirtWarning` when permutation replications had to be skipped (missing data
leaving a permuted pair without usable responses).

### `ParallelAnalysisResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Number of leading observed eigenvalues above `reference` |
| `observed` | ndarray | Observed eigenvalues, descending |
| `reference`, `reference_mean` | ndarray | Reference quantile (or mean) and mean per position |
| `replications` | ndarray | `(n_replications - n_skipped, p)` reference eigenvalues |
| `correlation` | DataFrame | The observed correlation matrix |
| `eigen_type`, `reference_kind`, `correlation_kind`, `quantile`, `n_skipped` | | Settings used and skipped replications |

### `velicer_map`

```text
velicer_map(correlation, *, power=4) -> MAPResult
```

Velicer's minimum average partial test: after partialling out the first $m$ principal
components, the average of the off-diagonal partial correlations raised to `power` is minimized
over $m = 0, \dots, p - 1$. `power=2` is the original test (Velicer, 1976), `power=4` the revised
one (Velicer, Eaton & Fava, 2000). `correlation` is a matrix (DataFrame / array) or a
`PolychoricResult`.

### `MAPResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | The minimizing $m$ |
| `statistic` | ndarray | Average partial correlation statistic for $m = 0, \dots, p - 1$ (NaN once a partial variance is 0) |
| `power` | int | 2 or 4 |

### `empirical_kaiser`

```text
empirical_kaiser(correlation, *, n_obs=None) -> EmpiricalKaiserResult
```

Empirical Kaiser criterion (Braeken & van Assen, 2017): the $j$-th eigenvalue is compared with
$\max\{(1+\sqrt{p/n})^2 (p-\sum_{i<j}\lambda_i)/(p-j+1),\ 1\}$. `n_obs` is required unless
`correlation` is a `PolychoricResult` (whose smallest pairwise $n$ is used).

### `EmpiricalKaiserResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Number of leading eigenvalues above the reference |
| `eigenvalues`, `reference` | ndarray | Eigenvalues and reference values |
| `n_obs` | int | Sample size used |

### `efa`

```text
efa(correlation, *, n_factors, method="uls", n_obs=None) -> EFAResult
```

Unrotated exploratory factor analysis of a correlation matrix (DataFrame, array or
`PolychoricResult`) by ULS (minres; Harman & Jones, 1966) or ML (Jöreskog, 1967), minimizing over
the uniquenesses from $1 - \mathrm{SMC}$ within $[0.005, \max(1, \mathrm{SMC})]$, as psych's
`fa(rotate = "none")` and EFAtools' `EFA(rotation = "none")`. With `n_obs` (default for a
`PolychoricResult`: its smallest pairwise $n$) the fit includes the Bartlett-corrected $\chi^2$, its
$p$-value, CFI and RMSEA; CAF and $df$ are always reported. Warns `HeywoodWarning` when a
uniqueness reaches its bound and `ConvergenceWarning` when the optimizer stops early.

### `EFAResult`

| Attribute | Type | Description |
|---|---|---|
| `loadings` | DataFrame | Unrotated loadings, columns signed to non-negative sums |
| `uniquenesses`, `communalities` | Series | $\psi_i$ and $h_i^2$ |
| `fit` | dict | `caf`, `df` and, with `n_obs`, `chi2`, `p`, `cfi`, `rmsea` |
| `method`, `converged`, `heywood`, `objective` | | Estimator and diagnostics |

### `hull`

```text
hull(correlation, *, n_obs=None, max_factors=None, rng=None, method="uls",
     fit_index="caf", n_replications=1000) -> HullResult
```

Hull method (Lorenzo-Seva, Timmerman & Kiers, 2011), as EFAtools' `HULL()`: solutions with
$k = 0, \dots, J$ factors are placed by fit (CAF, CFI or $1-$RMSEA) and degrees of freedom; on the
upper convex boundary the selected $k$ maximizes the scree ratio
$st_k = \frac{(f_k - f_{k-1})/(df_k - df_{k-1})}{(f_{k+1} - f_k)/(df_{k+1} - df_k)}$. By default
$J$ is one more than parallel analysis (needs `rng` and `n_obs`), capped for positive degrees of
freedom. At least 6 variables.

### `HullResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Selected number of factors |
| `table` | DataFrame | `n_factors`, `fit`, `df`, `st`, `on_hull`, `heywood`, `converged` per $k$ |
| `fit_index`, `method`, `max_factors` | | Settings used |

### `comparison_data`

```text
comparison_data(data, *, rng, max_factors=None, n_population=10000, n_samples=500,
                alpha=0.30, max_iter=50) -> ComparisonDataResult
```

Comparison-data method (Ruscio & Roche, 2012), as EFAtools' `CD()`. For $k = 1, 2, \dots$ a
population with $k$ common factors that reproduces the observed Pearson correlations and every
variable's distribution is generated (Ruscio & Kaczetow, 2008); samples of the data's size are
drawn and the RMSE between their eigenvalues and the observed ones is recorded. A factor is added
while a one-sided Mann–Whitney test at `alpha` says it lowers the RMSE. Complete numeric data.

### `ComparisonDataResult`

| Attribute | Type | Description |
|---|---|---|
| `n_factors` | int | Suggested number of factors |
| `rmse` | DataFrame | Eigenvalue RMSE per sample (rows) and tested $k$ (columns) |
| `p_values` | Series | Rank-sum $p$ for $k$ vs $k - 1$ ($k \ge 2$) |
| `population_rmse` | Series | How well each population reproduces the observed correlations |
| `max_factors` | int | Largest $k$ that could be tested |

### `model_parallel_analysis`

```text
model_parallel_analysis(fit, data, *, key, n_replications=100, eigen_type="max",
                        zero_cell_correction=0.5, item_names=None) -> ModelParallelAnalysisResult
```

Drasgow & Lissak's (1983) modified parallel analysis: the second eigenvalue of the observed
polychoric (tetrachoric) matrix is compared with its distribution in data simulated from `fit`
(same size and missingness, $\theta \sim N(\hat\mu, \hat\sigma^2)$);
$p = (1 + \#\{T_b \ge T\})/(B + 1)$. `eigen_type="max"` puts the largest off-diagonal correlation
on the diagonal, as ltm's `unidimTest`; `"smc"` and `"pca"` are also available. `fit` must be a
single-group unidimensional model with a `Normal` latent distribution.

### `ModelParallelAnalysisResult`

| Attribute | Type | Description |
|---|---|---|
| `statistic` | float | Observed second eigenvalue |
| `p_value` | float | Monte Carlo $p$-value |
| `observed` | ndarray | Observed eigenvalues |
| `replications` | ndarray | `(B, p)` eigenvalues of the simulated data |
| `eigen_type`, `n_skipped` | | Setting used and skipped simulations |

### `LocalDependenceResult`

Returned by `FitResult.local_dependence()`.

| Attribute | Type | Description |
|---|---|---|
| `pairs` | DataFrame | One row per pair: `item_i`, `item_j`, `n`, `q3`, `q3_star`, `x2`, `g2`, `df`, `p_x2`, `p_g2`, `cramer_v`; with a leading `group` column and one row per group and pair in a multiple-group model |
| `q3` | DataFrame | Item x item Q3 matrix; one matrix per group on a `(group, item)` row index in a multiple-group model |
| `q3_mean` | float or Series | Mean off-diagonal Q3; a Series indexed by group in a multiple-group model (each group's Q3\* uses its own mean) |
| `max_q3_star` | float | Largest Q3\* (over all groups) |

### `JSIResult`

Returned by `FitResult.jsi()`.

| Attribute | Type | Description |
|---|---|---|
| `matrix` | DataFrame | Removed item (rows) x item (columns): $\mathrm{JSI}_{ji}$, NaN on the diagonal |
| `pairs` | DataFrame | One row per pair $i<j$: `item_i`, `item_j`, `jsi_ij` (item $i$'s slope with $j$ removed), `jsi_ji`, `folded` (their sum, mirt's `fold = TRUE`) |
| `method` | str | Method used |
| `converged` | Series | Per removed item, whether the refit converged |

### `detect`

```text
detect(data, clusters) -> DetectResult
```

Confirmatory DETECT (Zhang & Stout, 1999) for a given partition of the items. The conditional
covariance of each pair is the average of its covariances given the total score and given the
rest score; DETECT $= 100 \times$ the mean of $\delta_{ij}\,\mathrm{ccov}_{ij}$ ($\delta = 1$
within a cluster, $-1$ across), with ASSI and RATIO (Zhang, 2007). `data` must be complete
numeric item scores; `clusters` is a mapping item → label or labels in column order. A common
reading (Jang & Roussos, 2007; Zhang, 2007): DETECT $< 0.2$ essentially unidimensional,
$0.2$–$0.4$ weak, $0.4$–$1$ moderate, $> 1$ strong; ASSI $> 0.25$ or RATIO $> 0.36$ an essential
deviation for the partition.

### `DetectResult`

| Attribute | Type | Description |
|---|---|---|
| `indices` | DataFrame | Rows `DETECT`, `ASSI`, `RATIO`, `MADCOV100`, `MCOV100`; columns `unweighted`, `weighted` |
| `detect`, `assi`, `ratio` | float | Unweighted indices |
| `conditional_covariance` | DataFrame | Item x item conditional covariances |
| `clusters` | Series | Partition used |
| `n_obs` | int | Respondents |

### `exploratory_detect`

```text
exploratory_detect(data, *, max_clusters=4, rng=None,
                   estimation_rows=None) -> ExploratoryDetectResult
```

Splits the rows into an estimation half and a validation half (`rng` for a random split, or
`estimation_rows`), clusters the items on the estimation half by Ward's method on
$\max(\mathrm{ccov}) - \mathrm{ccov}$, and reports DETECT, ASSI and RATIO of every partition
into $2, \dots,$ `max_clusters` clusters on both halves.

### `ExploratoryDetectResult`

| Attribute | Type | Description |
|---|---|---|
| `table` | DataFrame | Per number of clusters: `cluster_sizes` and `DETECT` / `ASSI` / `RATIO` on the estimation and validation halves |
| `clusters` | dict[int, Series] | Partition for each number of clusters |
| `n_clusters` | int | Number of clusters with the largest estimation-half DETECT |
| `estimation_rows`, `validation_rows` | ndarray | Row positions of the halves |

### `dimtest`

```text
dimtest(data, *, rng, assessment_items=None, training_fraction=1/3, n_bootstrap=100,
        min_group=20, max_assessment_size=None) -> DimtestResult
```

DIMTEST (Stout, 1987; Nandakumar & Stout, 1993; Stout, Froelich & Gao, 2001) for complete binary
data. The items are split into an assessment subtest (AT) and a partitioning subtest (PT); given
the PT score, the AT items' covariances are near zero under essential unidimensionality. Their
standardized sum $T_L$ is corrected by its mean in unidimensional data generated from
kernel-smoothed item response functions, $T = (T_L - \bar T_G)/\sqrt{1 + 1/B}$, and
$p = 1 - \Phi(T)$. Without `assessment_items` the AT is chosen by conditional covariances
(DETECT) on a random `training_fraction` of the respondents and the test uses the rest. The test
is conservative (type I error below the nominal level in simulations): a small $p$ is strong
evidence of a second dimension, a large $p$ only weak evidence for unidimensionality.

### `DimtestResult`

| Attribute | Type | Description |
|---|---|---|
| `statistic`, `p_value` | float | $T$ and $1 - \Phi(T)$ |
| `t_l`, `t_g` | float, ndarray | $T_L$ and the bootstrap $T_L$ values |
| `assessment_items`, `partitioning_items` | list[str] | AT and PT |
| `n_groups` | int | PT-score groups used |
| `training_rows`, `test_rows` | ndarray | Rows used to choose the AT and to test |

### `dimensionality`

```text
dimensionality(data, *, key, fit=None, n_replications=100, zero_cell_correction=0.5,
               alpha=0.05, q3_star_threshold=0.2, rmsea_threshold=0.05,
               detect_threshold=0.2, max_clusters=4, item_names=None) -> DimensionalityResult
```

Runs several diagnostics and reports them side by side, with two separate consensus lines
(no method is reliable alone; Auerswald & Moshagen, 2019).

- **How many dimensions?** `parallel_analysis()` (polychoric, PCA, permutation, 95%),
  `velicer_map()` (revised), `empirical_kaiser()`, with at least 6 items `hull()` (CAF, ULS), and with complete data
  `comparison_data()`; consensus = mode (all values on a tie) and
  median, with the number of methods agreeing.
- **Is essential unidimensionality tenable?** `exploratory_detect()` on complete data (the
  validation-half DETECT of the selected partition $<$ `detect_threshold`), `dimtest()` on
  complete binary data ($p \ge$ `alpha`), and with `fit`:
  `model_parallel_analysis()`
  ($p \ge$ `alpha`), the largest Q3\* $<$ `q3_star_threshold` (0.2 is a common rule; critical
  values depend on the test and sample) and the RMSEA of `FitResult.m2()` $\le$ `rmsea_threshold`
  (close fit; unavailable with missing data, then reported as such). Consensus = number of
  available methods supporting it. Methods are not weighted.

### `DimensionalityResult`

| Attribute | Type | Description |
|---|---|---|
| `number_of_dimensions` | DataFrame | `method`, `n_factors`, `detail`, `source` |
| `unidimensionality` | DataFrame | `method`, `statistic`, `threshold`, `supports_unidimensionality` (`None` when unavailable), `detail`, `source` |
| `n_factors_mode`, `n_factors_median`, `n_agreeing` | tuple[int, ...], float, int | Consensus on the number of dimensions |
| `n_supporting`, `n_unidimensionality_methods` | int | Consensus on essential unidimensionality |
| `results` | dict | The individual results |
| `summary()` | str | Plain-text report (also `str(result)`) |

---

## Information and reliability

Functions of a fitted unidimensional model ([guide section 5](guide.md#information-expected-score-and-reliability)).
All raise `SpecificationError` for a multidimensional model.

### `item_information`

```text
item_information(res, theta, *, group=None) -> DataFrame
```

Fisher information of every item at the latent trait values `theta`,
$I_j(\theta)=\sum_kP_{jk}(\theta)\,(\partial\log P_{jk}(\theta)/\partial\theta)^2$ (Birnbaum, 1968;
Samejima, 1969), computed for every item model by automatic differentiation. It is what mirt's
`iteminfo()` returns.

| Argument | Type | Description |
|---|---|---|
| `res` | `FitResult` | A fitted unidimensional model |
| `theta` | float / 1-D array | Latent trait values (finite) |
| `group` | str, optional | Group name; required for a multiple-group model |

**Returns**: `DataFrame` with one row per value of `theta` (the index, named `theta`) and one
column per item.

### `test_information`

```text
test_information(res, theta, *, group=None) -> DataFrame
```

Test information $I(\theta)=\sum_jI_j(\theta)$ and the conditional standard error
$1/\sqrt{I(\theta)}$ (mirt's `testinfo()`). Arguments as for
[`item_information`](#item_information).

**Returns**: `DataFrame` indexed by `theta` with the columns `information` and `se`.

### `expected_score`

```text
expected_score(res, theta, *, group=None) -> DataFrame
```

Expected item scores $E_j(\theta)=\sum_kk\,P_{jk}(\theta)$ and the test characteristic curve
$T(\theta)=\sum_jE_j(\theta)$, with categories scored $0,\dots,K_j-1$ as in the data (mirt's
`expected.item()` and `expected.test()`). Arguments as for
[`item_information`](#item_information).

**Returns**: `DataFrame` indexed by `theta` with one column per item and the column `total`. An
item named `total` raises `SpecificationError`.

### `reliability`

```text
reliability(res, data=None, *, group=None, item_names=None) -> DataFrame
```

Marginal reliability and, with `data`, the empirical reliability of EAP scores.

- **Marginal**: $\bar\rho=\int\frac{\sigma^2I(\theta)}{\sigma^2I(\theta)+1}\,\phi(\theta;\mu,\sigma^2)\,d\theta$
  for a latent distribution $N(\mu,\sigma^2)$ and test information $I(\theta)$ (Green et al., 1984;
  Thissen & Wainer, 2001). It equals mirt's `marginal_rxx()` for $N(0,1)$ and does not change
  under a linear transformation of the latent scale. The integral is evaluated on 801 equally
  spaced points on $\mu\pm8\sigma$, whatever quadrature the model was fitted with.
- **Empirical**: $\hat\rho=s^2_{\hat\theta}/(s^2_{\hat\theta}+\overline{\mathrm{SE}^2})$ from the
  EAP scores and standard errors of the respondents in `data` (`score(data, method="EAP")`), with
  the sample variance $s^2_{\hat\theta}$ (denominator $N-1$). It is mirt's `empirical_rxx()`.
  Rows with missing responses are scored and included.

| Argument | Type | Description |
|---|---|---|
| `res` | `FitResult` | A fitted unidimensional model with `Normal` latent distributions |
| `data` | DataFrame / 2-D array, optional | Responses on the fitted items; adds the empirical reliability |
| `group` | array, optional | Group label of each row of `data` (required with `data` for a multiple-group model) |
| `item_names` | list[str], optional | Column names when `data` is an array |

**Returns**: `DataFrame` with one row per group (and a leading `group` column for a
multiple-group model): `marginal` and, with `data`, `empirical` (NaN for a group with fewer than
two respondents in `data`).

**Raises**: `SpecificationError` for a multidimensional model, an `EmpiricalHistogram`,
`DavidianCurve` or `NormalMixture` latent distribution, or `group` / `item_names` without `data`.

---

## Summed scores

### `summed_score_table`

```text
summed_score_table(res) -> DataFrame
```

Summed-score EAP table of a fitted model (Thissen, Pommerich, Billeaud & Williams, 1995;
[guide section 7](guide.md#summed-score-eap)). For each summed score $s$, with the fitted
quadrature nodes $\theta_q$ and latent weights $w_q$ and the probability $L(s\mid\theta_q)$ of the
score from [`summed_score_distribution`](#summed_score_distribution):
$P(s)=\sum_qL(s\mid\theta_q)w_q$, the posterior mean
$\sum_q\theta_qL(s\mid\theta_q)w_q/P(s)$ and the posterior standard deviation. It is the table of
mirt's `fscores(method = "EAPsum", full.scores = FALSE)`, whose `expected` column is $N\,P(s)$.

**Returns**: `DataFrame` with one row per summed score $0,\dots,\sum_j(K_j-1)$ (per group and
score, with a leading `group` column, for a multiple-group model): `score`, `theta`, `se`,
`probability`.

**Raises**: `SpecificationError` for a multidimensional model or an `EmpiricalHistogram`,
`DavidianCurve` or `NormalMixture` latent distribution.

### `summed_score_distribution`

```text
summed_score_distribution(probs, *, method="recursion", log=False) -> Array
```

Distribution of the summed score of independent item scores
([guide section 5](guide.md#summed-score-distribution)). `probs` has shape `(..., J, K)`: the
category probabilities of `J` items (each row sums to 1; items with fewer than `K` categories are
padded with zeros), for example an IRT model's category probabilities at one $\theta$; leading axes
are batch axes. The result has shape `(..., J * (K - 1) + 1)` with the probabilities of the summed
scores $0, \dots, J(K-1)$. For dichotomous items this is the Poisson-binomial distribution. The
function works inside `jax.jit` and `jax.vmap` and can be differentiated; values are not checked.

| `method` | What it is | Accuracy |
|---|---|---|
| `"recursion"` (default) | Lord–Wingersky recursion | Exact; small relative error in every probability, tails included |
| `"fft"` | Discrete Fourier transform of the characteristic function | Exact up to an absolute error of about `1e-16`; probabilities below about `1e-15` are not reliable |
| `"normal"` | Normal approximation with continuity correction | Dichotomous items only; need not sum to exactly 1 |
| `"refined_normal"` | Normal approximation with a skewness correction | Dichotomous items only; need not sum to exactly 1 |
| `"poisson"` | Poisson approximation | Dichotomous items only; useful when every probability is small |

`log=True` returns log probabilities. With `"recursion"` the recursion runs in log space and keeps
tail probabilities that would underflow to 0 (it is slower); for the other methods the logarithm
of the result is taken. Raises `SpecificationError` for a wrong shape, an unknown `method`, or an
approximation with `K != 2`.

---

## Classification

### `classification_accuracy`

```text
classification_accuracy(res, *, cut_scores, method="lee", scale=None, data=None, group=None,
                        item_names=None, score_method="EAP") -> ClassificationResult
```

Classification accuracy and consistency of cut scores
([guide section 7](guide.md#classification-accuracy-and-consistency); DESIGN.md §18.3). The cuts
$c_1<\dots<c_C$ define the categories $h(v)=\#\{k: v\ge c_k\}=0,\dots,C$. At a true value
$\theta$ each method gives the probabilities $P_h(\theta)$ of the observed categories and a true
category $t(\theta)$:

- `"lee"` (Lee, 2010): $P_h(\theta)=\sum_{s:\,h(s)=h}L(s\mid\theta)$ with the Lord–Wingersky
  distribution of the summed score, $t(\theta)=h(\tau(\theta))$ with the expected summed score
  $\tau$. A non-integer cut $c$ puts the scores $s\ge\lceil c\rceil$ above it.
- `"rudner"` (Rudner, 2001, 2005): $P_h(\theta)=\Phi((c_{h+1}-\theta)/\mathrm{SE})-\Phi((c_h-\theta)/\mathrm{SE})$,
  $\mathrm{SE}=I(\theta)^{-1/2}$, $t(\theta)=h(\theta)$.

Over points $\theta_n$ with weights $w_n$ (the fitted quadrature and latent weights without `data`,
the "D" approach; the respondents' estimates with weights $1/N$ with `data`, the "P" approach;
Lathrop & Cheng, 2013), $A_{tk}=\sum_nw_n[t(\theta_n)=t]P_k(\theta_n)$ and
$B_{hk}=\sum_nw_nP_h(\theta_n)P_k(\theta_n)$ give accuracy $\sum_hA_{hh}$, consistency
$\sum_hB_{hh}$ and Cohen's kappa of consistency. These are cacIRT's `class.Lee()` / `class.Rud()`.

| Argument | Type | Description |
|---|---|---|
| `res` | `FitResult` | A fitted unidimensional model |
| `cut_scores` | float / array | Strictly increasing cuts. On the summed score in $(0, S_{\max}]$, selecting different scores |
| `method` | `"lee"` / `"rudner"` | Method (default `"lee"`) |
| `scale` | `"summed_score"` / `"theta"`, optional | Scale of the cuts; default the method's own. Otherwise converted through the test characteristic curve ($\tau(c)$ for Lee, $\tau^{-1}(c)$ for Rudner) |
| `data` | DataFrame / 2-D array, optional | Responses; averages over the respondents' estimates (P approach) |
| `group` | array, optional | Group label of each row of `data` (required with `data` for a multiple-group model) |
| `item_names` | list[str], optional | Column names when `data` is an array |
| `score_method` | `"EAP"` / `"MAP"` / `"ML"` / `"WLE"` | Estimates of the P approach (default `"EAP"`); Rudner's SE is always $I(\hat\theta)^{-1/2}$ |

**Returns**: [`ClassificationResult`](#classificationresult).

**Raises**: `SpecificationError` for a multidimensional model, an unknown `method` / `scale` /
`score_method`, invalid cuts, a summed-score cut outside the range of the test characteristic
curve (Rudner), `group` / `item_names` / `score_method` without `data`, or a non-finite estimate
in the P approach (ML on an extreme pattern).

### `ClassificationResult`

Frozen dataclass returned by [`classification_accuracy`](#classification_accuracy). In a
multiple-group result every DataFrame has a leading `group` column (the two tables a
`(group, category)` row index).

| Attribute | Content |
|---|---|
| `method`, `scale`, `cut_scores`, `approach` | The settings (`approach` is `"D"` or `"P"`) |
| `summary` | One row per group: `accuracy`, `consistency`, `kappa` (and `n` for the P approach) |
| `by_category` | `category`, `true_proportion`, `proportion` (observed), `accuracy` (observed $=h$ given true $=h$), `consistency` (second $=h$ given first $=h$) |
| `by_cut` | `cut` (and `summed_score_cut` / `theta_cut` when converted), `accuracy`, `consistency`, `kappa`, `false_positive` (true below, observed at or above), `false_negative` |
| `conditional` | Per node (D) or respondent (P): `theta`, `weight`, `true_category`, `accuracy`, `consistency`, and `true_score` (Lee) or `se` (Rudner) |
| `accuracy_table` | $A$: joint probabilities of the true (rows) and observed (columns) categories |
| `consistency_table` | $B$: joint probabilities of the categories of two independent administrations |
| `accuracy`, `consistency`, `kappa` | Floats of a single-group result (`SpecificationError` for a multiple-group result) |

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
Quadrature.equispaced(n_points=61, *, bounds=(-6.0, 6.0))
Quadrature.default(n_factors=1, *, bounds=(-6.0, 6.0))
Quadrature.gauss_hermite(n_points=21)
```

An equally spaced quadrature grid (endpoints included), extended to a Cartesian product grid in
2–3 dimensions. `Quadrature.default(n_factors)` matches mirt's defaults: 61 points for 1 factor,
31 per dimension ($31^2 = 961$) for 2 factors, and 15 per dimension ($15^3 = 3{,}375$) for 3
factors on $[-6, 6]$. `nodes()` returns the 1-D axis of shape `(n_points,)`; `grid(n_factors)`
returns the product grid of shape `(n_points**n_factors, n_factors)`. At least 3 points per
dimension; limits must be finite with `lower < upper`.

`Quadrature.gauss_hermite(n_points)` uses the Gauss–Hermite nodes $\sqrt2\,x_q$ (3–200 per
dimension, product rule in 2–3 dimensions) for `Normal` latent distributions, reweighted for the
estimated mean and covariance; `base_log_weights(n_factors)` returns the fixed
per-node log weights (0 for equispaced grids). Also used by `NormalMixture` (each component gets
the weights of a `Normal` with its mean and variance). Not available with `EmpiricalHistogram` or
`DavidianCurve`.

---

## Exceptions and warnings

### `flexpyirt.errors`

Import with `from flexpyirt.errors import ...`.

#### `FlexpyirtError`

Base class of every exception flexpyirt raises on purpose.

`SpecificationError` and the warning classes are shared with flexpyctt (defined in
`flexpyctt.errors` and re-exported here), and `FlexpyirtError` / `FlexpyirtWarning` are the same
classes as flexpyctt's bases `FlexpycttError` / `FlexpycttWarning`. Catching flexpyirt's classes
therefore also catches what a flexpyctt function raises.

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
near 0 or `u` near 1 in 3PL/4PL) or a free slope has $|a| > 20$, naming every affected item; also
for a degenerate `NormalMixture` solution (a component with weight below `1e-3`, standard
deviation below half the grid spacing, or mean outside the grid, or two coinciding components).

#### `NotPositiveDefiniteWarning`

Emitted by `polychoric()` when the pairwise matrix is not positive definite and `smooth=True` smoothed it.
