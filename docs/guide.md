# User guide

English | [日本語](guide.ja.md)

How to specify, fit and interpret IRT models with flexpyirt. The code on this page runs top to
bottom, and `tests/test_docs.py` runs all of it on every test run. For the arguments of each
function, see the [API reference](reference.md).

1. [Setup](#1-setup)
2. [Data format](#2-data-format)
3. [Specifying a model](#3-specifying-a-model)
4. [Fitting](#4-fitting)
5. [Reading the results](#5-reading-the-results)
6. [Standard errors](#6-standard-errors)
7. [Scoring](#7-scoring)
8. [Latent distributions and identification](#8-latent-distributions-and-identification)
9. [Multiple groups](#9-multiple-groups)
10. [Fixed item parameter calibration (FIPC)](#10-fixed-item-parameter-calibration-fipc)
11. [Editing the ParameterTable](#11-editing-the-parametertable)
12. [Choosing the quadrature grid](#12-choosing-the-quadrature-grid)
13. [Errors and warnings](#13-errors-and-warnings)

## 1. Setup

flexpyirt computes in 64-bit floating point. The JAX setting belongs to you, so the library does
not change it at import. Enable it before fitting (calling `fit()` without it raises
`PrecisionError`).

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd

from flexpyirt import BAEM, GRM, FourPL, Model, Normal, Quadrature, Rasch, ThreePL, TwoPL, fit
```

Simulate the data used in the examples below.

```python
rng = np.random.default_rng(2026)


def simulate_2pl(a, d, theta, rng):
    """Generate 0/1 responses from a 2PL model."""
    p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    return (rng.random(p.shape) < p).astype(float)


n_items = 12
items = [f"q{j:02d}" for j in range(1, n_items + 1)]
a_true = rng.lognormal(0.0, 0.3, n_items)
d_true = rng.normal(0.0, 1.0, n_items)
df = pd.DataFrame(simulate_2pl(a_true, d_true, rng.normal(0, 1, 1500), rng), columns=items)
```

## 2. Data format

- **One row per respondent, one column per item.** Column names are the item names.
- A pandas DataFrame, a polars DataFrame or a 2-D numpy array. For an array, pass the column names
  with `item_names=` (without it, columns are taken in the model's item order).
- Values are category numbers `0, 1, ..., K-1`; dichotomous items are 0/1. **Missing is NaN**
  (missing at random; it is left out of the likelihood).
- Columns that are not in the model, or model items missing from the data, are errors. Drop extra
  columns first.
- Every category of an item with estimated parameters must appear at least once in the data
  (otherwise its threshold diverges; collapse categories or fix the item).

```python
arr = df.to_numpy()
spec = Model.uniform(TwoPL(), items=items)
res_np = fit(spec, arr, item_names=items, se=None)
```

## 3. Specifying a model

A model is an immutable `Model` object. Methods such as `.fix()` and `.with_groups()` return a new
`Model` and leave the original unchanged.

```python
spec = Model.uniform(TwoPL(), items=items)  # same item model for every item
mixed = Model.from_items({"x1": TwoPL(), "x2": Rasch(), "s1": GRM(n_cat=4)})  # per item
print(spec.item_names[:3], mixed.item_names)
```

### Item models

| Class | Response probability | Internal parameters | Display (added to `items`) |
|---|---|---|---|
| `TwoPL()` | $P(y=1)=\sigma(a\theta+d)$ | `a`, `d` | `b = -d/a` |
| `Rasch()` | $P(y=1)=\sigma(\theta+d)$ ($a=1$ fixed) | `a` (fixed), `d` | `b = -d` |
| `ThreePL()` | $P(y=1)=g+(1-g)\sigma(a\theta+d)$ ($u=1$ fixed) | `a`, `d`, `g`, `u` (fixed) | `b = -d/a` |
| `FourPL()` | $P(y=1)=g+(u-g)\sigma(a\theta+d)$ | `a`, `d`, `g`, `u` | `b = -d/a` |
| `GRM(n_cat=K)` | $P(y\ge k)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$ | `a`, `d1`…`d{K-1}` | `bk = -dk/a` |

The parameterization is mirt's slope–intercept form (`itemtype = "2PL"`, `"Rasch"`, `"3PL"`,
`"4PL"`, `"graded"`). For 3PL and 4PL, `g` and `u` are estimated on the natural scale with
$0 < g < u \le 1$ (mirt optimizes them on the logit scale; the interior ML estimates agree). GRM
categories start at 0 (subtract 1 from data coded from 1, such as mirt's `Science`).

```python
theta3 = rng.normal(0, 1, 2000)
a3 = np.array([1.6, 1.8, 1.7, 1.6, 1.9, 1.7])
d3 = np.array([-1.0, -0.7, -1.2, -0.8, -0.9, -0.6])
g3 = np.full(6, 0.2)
p3 = g3 + (1.0 - g3) / (1.0 + np.exp(-(np.outer(theta3, a3) + d3)))
df_3pl = pd.DataFrame(
    (rng.random(p3.shape) < p3).astype(float), columns=[f"c{j}" for j in range(1, 7)]
)
res_3pl = fit(Model.uniform(ThreePL(), items=df_3pl.columns), df_3pl)
print(res_3pl.items[["item", "a", "d", "g", "b"]].round(3))
```

```python
n_cat = 4
cuts = np.array([-1.0, 0.2, 1.3])
theta = rng.normal(0, 1, 800)
latent_cont = np.outer(theta, np.ones(5)) + rng.logistic(0, 1, (800, 5))
grm_df = pd.DataFrame(
    np.searchsorted(cuts, latent_cont).astype(float), columns=[f"s{j}" for j in range(1, 6)]
)
grm = fit(Model.uniform(GRM(n_cat=n_cat), items=grm_df.columns), grm_df)
print(grm.items[["item", "a", "d1", "d2", "d3", "b1", "b2", "b3"]].round(3))
```

### Fixing parameters

`.fix(item=..., <param>=value)` fixes internal (slope–intercept) parameters. Fixed parameters are
not estimated and their standard errors are NaN.

```python
spec_fixed = Model.uniform(TwoPL(), items=items).fix(item="q01", a=1.0, d=0.0)
```

## 4. Fitting

```python
res = fit(
    spec,
    df,
    method=BAEM(tol=1e-4, max_iter=2000, m_step_newton_steps=3),  # defaults
    quadrature=Quadrature.equispaced(61, lim=(-6, 6)),  # default
    se="oakes",  # default
)
```

- **`method`**: EM settings. EM stops when the largest absolute change of the free parameters is
  below `tol` (the same definition as mirt's `TOL`). Reaching `max_iter` gives a
  `ConvergenceWarning`. Each M-step takes `m_step_newton_steps` Newton steps; none of them
  decreases the expected log-likelihood, so EM stays monotone.
- **`quadrature`**: the latent-variable grid. The default is 61 equally spaced points on
  $[-6, 6]$ (mirt's default). See [section 12](#12-choosing-the-quadrature-grid) for when to
  widen it.
- **`se`**: `"oakes"` (default), `"hessian"`, or `None` (skip).

The first fit includes JAX compilation time. Later fits of data with the same shape are faster.

## 5. Reading the results

`fit()` returns a `FitResult`.

```python
print(res)  # FitResult(loglik=..., n_params=..., n_obs=..., converged in ... iterations, se=oakes)
print(res.items.head(3))  # estimates and SEs per item
print(res.latent)  # latent distribution per group (mean, var and their SEs)
print(res.loglik, res.aic, res.bic, res.n_params, res.n_obs)
print(res.converged, res.n_iter)
print(res.history.tail(3))  # iter, loglik, max_change
```

| Attribute | Contents |
|---|---|
| `items` | One row per item (per group × item with groups): `model`, internal parameters, `<param>_se`, display `b` / `bk` |
| `latent` | One row per group: `group`, `mean`, `var`, `mean_se`, `var_se` (SEs of fixed values are NaN) |
| `parameters` | The ParameterTable ([section 11](#11-editing-the-parametertable)) plus `estimate` and `se` |
| `loglik`, `aic`, `bic` | Marginal log-likelihood and information criteria ($-2\ell+2k$, $-2\ell+k\log N$) |
| `n_params`, `n_obs` | Number of free parameters and respondents |
| `converged`, `n_iter`, `history` | Whether EM converged, the number of iterations, log-likelihood and largest change per iteration |
| `vcov`, `se_method` | Covariance matrix of the free parameters (in `free_id` order) and how it was computed |

`history.loglik` never decreases between iterations (EM is monotone). If it does, that is a bug;
please report it.

```python
assert (res.history["loglik"].diff().dropna() >= -1e-8).all()
```

## 6. Standard errors

- **`se="oakes"`** (default): the observed information from Oakes' (1999) identity. The second
  derivatives through the E-step expected counts come from JAX autodiff.
- **`se="hessian"`**: the second derivative of the marginal log-likelihood, taken directly.

The two are the same quantity mathematically and agree numerically up to rounding. Rows and
columns of `vcov` follow the free-parameter numbers (`free_id` in the ParameterTable).

```python
res_h = fit(spec, df, se="hessian")
print(np.max(np.abs(res.vcov - res_h.vcov)))  # practically 0

params = res.parameters
labels = params[params["free"]].drop_duplicates("free_id").sort_values("free_id")
vcov = pd.DataFrame(
    res.vcov,
    index=labels["item"] + "." + labels["param"],
    columns=labels["item"] + "." + labels["param"],
)
print(vcov.iloc[:2, :2])
```

If the information matrix is not positive definite (estimates on a boundary, weak
identification), flexpyirt warns and the SEs are NaN. SEs of the display `b` (delta method) are
planned for M4.

## 7. Scoring

`FitResult.score()` returns EAP estimates and posterior standard deviations. The prior is the
estimated latent distribution (the respondent's group's distribution with multiple groups).
Missing responses are left out of the likelihood; a respondent with every item missing gets the
prior mean and SD.

```python
scores = res.score(df)
print(scores.head(3))  # theta, se (index follows df)

new = df.iloc[:2].copy()
new.iloc[0, :3] = np.nan  # partly missing rows are scored too
print(res.score(new))
```

The scored data need not be the data used for fitting (the columns must be the same items).
MAP / ML / WLE / plausible values are planned for M4.

## 8. Latent distributions and identification

The latent distribution is a normal `Normal`, with weights from the density normalized on the
grid.

```python
Normal.standard()  # N(0, 1), nothing estimated
Normal(free_var=True)  # N(0, σ²), variance estimated (mean fixed at 0)
Normal.free()  # mean and variance estimated
Normal(mean=0.5, var=1.2)  # fixed values
```

**Default**: in a single-group model without an explicit latent distribution, it is
$N(0,\sigma^2)$ with the variance estimated when every item is Rasch, and $N(0,1)$ otherwise.
Rasch slopes are fixed at 1, so the variance carries the scale (as in mirt).

```python
rasch = fit(Model.uniform(Rasch(), items=items), df)
print(rasch.latent[["mean", "var", "var_se"]])
```

Estimating the latent mean or variance needs something that fixes the scale. flexpyirt checks
this before fitting and raises `IdentificationError` if it is missing.

- A free variance needs at least one **fixed slope** with a nonzero value (Rasch items,
  `.fix(a=...)`, or, with multiple groups, anchor items whose slopes are equal across groups).
- A free mean needs at least one **fixed intercept** (FIPC, or anchor items with multiple groups).

```python
from flexpyirt.errors import IdentificationError

try:
    fit(Model.uniform(TwoPL(), items=items).with_latent(Normal(free_var=True)), df)
except IdentificationError as e:
    print("IdentificationError:", e)
```

## 9. Multiple groups

Calibrate responses from several populations (grades, test years, language versions) together.
Anchor items with parameters equal across groups link the scales, and the focal groups' latent
means and variances are estimated.

```python
n_per_group = 1200
theta_ref = rng.normal(0.0, 1.0, n_per_group)
theta_foc = rng.normal(0.5, 1.1, n_per_group)
d_foc = d_true.copy()
d_foc[6:] += rng.normal(0, 0.4, n_items - 6)  # non-anchor items may differ between groups
mg_df = pd.DataFrame(
    np.vstack(
        [simulate_2pl(a_true, d_true, theta_ref, rng), simulate_2pl(a_true, d_foc, theta_foc, rng)]
    ),
    columns=items,
)
group = np.repeat(["ref", "focal"], n_per_group)

anchors = items[:6]
# Groups are defined in keyword order; the first is the reference group.
mg_spec = (
    Model.uniform(TwoPL(), items=items)
    .with_groups(ref=Normal.standard(), focal=Normal.free())
    .equal_across_groups(items=anchors)  # all parameters equal across groups
)
mg = fit(mg_spec, mg_df, group=group, quadrature=Quadrature.equispaced(81, lim=(-8, 8)))
print(mg.latent)
print(mg.items[mg.items["item"] == "q01"])  # anchors have the same values in both groups
```

- `group=` gives each row's group label. Labels must match the groups declared in
  `with_groups()`.
- By default, parameters of items that are not made equal are **estimated separately per group**
  (mirt `multipleGroup`'s default).
- `equal_across_groups(items=..., params=["a"])` makes only some parameters equal. `items=`
  accepts a list of names or a function such as `lambda name: name.startswith("q0")`.
- Estimating a focal group's variance with no anchors, or with only intercepts equal, raises
  `IdentificationError`. If a focal group's mean is linked by a single parameter, flexpyirt warns
  because the estimate tends to be unstable.
- With multiple groups, `items` has a `group` column. Pass groups to `score(data, group=labels)`.

```python
mg_scores = mg.score(mg_df.iloc[[0, -1]], group=["ref", "focal"])
print(mg_scores)
```

## 10. Fixed item parameter calibration (FIPC)

Fix some items at the values of an already scaled item bank and estimate the new items and the
latent distribution. The fixed items set the scale, so the latent mean and variance are estimated
(equivalent to Kim's (2006) MWU-MEM).

```python
theta_new = rng.normal(0.4, 1.15, 2000)
fipc_df = pd.DataFrame(simulate_2pl(a_true, d_true, theta_new, rng), columns=items)

bank = pd.DataFrame({"item": items[:6], "a": a_true[:6], "d": d_true[:6]})
fipc_spec = (
    Model.uniform(TwoPL(), items=items)
    .fix_from(bank)  # fix the bank items in every group
    .with_latent(Normal.free())  # estimate the mean and variance
)
fipc = fit(fipc_spec, fipc_df, quadrature=Quadrature.equispaced(81, lim=(-8, 8)))
print(fipc.latent[["mean", "var"]])  # generating values: mean 0.4, variance 1.3225
```

For a bank in IRT form ($a, b$), pass `parameterization="irt"` (converted with $d=-ab$; GRM uses
columns `b1`, `b2`, …). A bank of Rasch items may omit the `a` column.

```python
bank_irt = pd.DataFrame({"item": items[:6], "a": a_true[:6], "b": -d_true[:6] / a_true[:6]})
fipc_spec_irt = (
    Model.uniform(TwoPL(), items=items)
    .fix_from(bank_irt, parameterization="irt")
    .with_latent(Normal.free())
)
```

## 11. Editing the ParameterTable

Every constraint is a column of the ParameterTable (like mirt's `pars = "values"`). Take the
table out, edit it with pandas and pass it to `fit(..., table=...)`; the table overrides the
`Model`'s settings.

```python
table = spec.parameter_table(df)
print(table.head(4))
```

| Column | Meaning |
|---|---|
| `group`, `item`, `param` | Row identifiers (the single group is `"all"`; latent rows have `item == "__latent__"` and `param` `mean` / `var`) |
| `model` | Item model name (edits are ignored) |
| `value` | Starting value (the fixed value if not free) |
| `free` | Whether the parameter is estimated |
| `lower`, `upper` | Bounds; starting values must lie strictly inside |
| `free_id` | Free-parameter number. **Rows with the same number are constrained to be equal.** Newly freed rows may keep `-1` (they get a number) |

```python
edited = table.copy()
row = (edited["item"] == "q05") & (edited["param"] == "a")
edited.loc[row, "value"] = 1.0
edited.loc[row, "free"] = False  # fix q05's slope at 1
row_d = (edited["item"] == "q05") & (edited["param"] == "d")
edited.loc[row_d, "value"] = 0.3  # change only the intercept's starting value
res_edit = fit(spec, df, table=edited, se=None)
print(res_edit.items.set_index("item").loc["q05", ["a", "d"]])
```

- Rows cannot be added, removed or renamed (`SpecificationError`).
- Sharing a `free_id` is limited to making **the same parameter of the same item equal across
  groups**. Equality across different items (for example one common slope) is not supported yet
  (planned for M5).

## 12. Choosing the quadrature grid

The latent weights are normalized on the grid $[\text{lower}, \text{upper}]$, so a distribution cut
off at the grid's ends affects the estimates. The default 61 points on $[-6, 6]$ are enough near
$N(0,1)$. When **an estimated group's variance is large (above 2) or its mean is far from 0**,
widen the grid while keeping the spacing.

```python
wide = Quadrature.equispaced(81, lim=(-8, 8))  # same 0.2 spacing, wider range
```

The reported `var` is the parameter of the normal distribution before the grid cuts it off; the
variance of the distribution on the grid is slightly smaller. mirt updates the variance with
closed-form moments, so where the grid ends matter its values differ from flexpyirt's. Widening
the grid makes them agree ([VALIDATION.md](VALIDATION.md)).

## 13. Errors and warnings

| Type | When |
|---|---|
| `PrecisionError` | `fit()` was called with JAX's 64-bit floating point disabled |
| `SpecificationError` | The specification and data disagree (missing or extra columns, out-of-range values, unobserved categories, fixing an unknown item, invalid table edits, unknown group labels) |
| `IdentificationError` | The scale is not identified ([section 8](#8-latent-distributions-and-identification)) |
| `ConvergenceWarning` | EM did not converge within `max_iter` |
| `HeywoodWarning` | A free item parameter ended within `1e-4` of a finite bound (e.g. `g` near 0 or `u` near 1 in 3PL/4PL) or a free slope has $|a| > 20$ |
| `FlexpyirtWarning` | The information matrix is not positive definite (SEs are NaN), or an anchor is a single parameter |

Every exception derives from `flexpyirt.errors.FlexpyirtError` and every warning from
`flexpyirt.errors.FlexpyirtWarning`. `SpecificationError` and `IdentificationError` are also
`ValueError`s.

```python
from flexpyirt.errors import SpecificationError

try:
    fit(spec, df.assign(extra=0))
except SpecificationError as e:
    print("SpecificationError:", e)
```

## References

- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
