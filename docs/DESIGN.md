# flexpyirt design (DESIGN)

English | [日本語](DESIGN.ja.md)

> Status: draft v0.2 (2026-11-20, reflects the M1 implementation). Decisions are marked
> "Decision"; open questions are in §12 "Open questions".

## 1. Goals and non-goals

### Goals
- A Python IRT estimation library built around **marginal maximum likelihood estimation (MMLE) with
  the Bock–Aitkin EM algorithm (BA-EM; Bock & Aitkin, 1981)**.
- Provide, with a Pythonic API, what R `mirt` does in practice:
  - multiple-group calibration (equality constraints across groups, partial anchors)
  - fixed item parameter calibration (FIPC)
  - flexible latent distributions (Normal / Empirical Histogram / later Davidian and normal
    mixtures)
  - priors on item parameters (Bayes modal = MAP estimation; EM finds the posterior mode)
  - standard errors (Oakes' 1999 identity, via automatic differentiation)
- jit, vmap and GPU support through JAX, and easy extension through autodiff (new item models and
  linking criteria without hand-written derivatives).

### Non-goals (at least for v1)
- Covering every feature of R `mirt`. MHRM, MCEM, exploratory high-dimensional MIRT, CDM and CAT
  are out of scope.
- Reproducing R's `mirt.model` string syntax.
- Beating mirt (C++) on speed for its own sake. Small problems may lose to jit compilation time.
  The selling points are "verified correctness" and "differentiable, extensible".

## 2. Layers

```
┌──────────────────────────────────────────────────────────────┐
│ ① Spec layer (written by the user, immutable)                │
│   Model, ItemSpec(TwoPL, GRM, ...), Group, LatentSpec(Normal, │
│   EmpiricalHistogram), Prior, constraints (fix / equal)       │
├──────────────────────────────────────────────────────────────┤
│ ② Compile layer (Python, validation)                         │
│   Spec + Data → ParameterTable (like mirt's pars='values')    │
│   → FreeMap (free-parameter vector ⇄ all parameters)          │
│   → response-pattern compression, padding, identification     │
├──────────────────────────────────────────────────────────────┤
│ ③ Core layer (pure JAX, jit)                                  │
│   e_step, m_step_items, m_step_latent, em_loop(lax.while_loop) │
│   squarem, oakes_information, scoring                         │
├──────────────────────────────────────────────────────────────┤
│ ④ Result layer (Python)                                       │
│   FitResult: parameter tables (DataFrame), SE, logLik,        │
│   AIC/BIC, convergence history, score(), summary(), repr_html │
└──────────────────────────────────────────────────────────────┘
```

**Decision**: ③ knows nothing about names, strings or DataFrames. Its inputs and outputs are
pytrees only (nested `jax.Array`s and our own frozen dataclasses registered with
`@jax.tree_util.register_dataclass`, not `NamedTuple` / `flax.struct`).

## 3. Notation

| Symbol | Meaning |
|---|---|
| $g=1..G$ | Groups (populations); $g=1$ is the reference group |
| $j=1..J$ | Items |
| $p$ | Unique response patterns within a group; $c_p$ is its frequency |
| $q=1..Q$ | Quadrature nodes $\theta_q$ |
| $w_{gq}$ | Quadrature weights of group $g$'s latent distribution ($\sum_q w_{gq}=1$) |
| $\xi_j^{(g)}$ | Parameters of item $j$ in group $g$ (identical across groups under equality constraints) |
| $\eta$ | Free-parameter vector (expanded into the parameters of $\xi$ and $w$ by the FreeMap) |

## 4. Algorithms

### 4.1 Preprocessing: response-pattern compression
- Make each group's response matrix (missing as `-1`) unique to get frequencies $c_p$.
- To limit jit recompilation, **pad the number of patterns to a power-of-two bucket** (rows with
  frequency 0).
- Groups have different numbers of patterns, so all groups are concatenated into one array plus
  a `group_index` (no per-group Python loop inside jit).

### 4.2 E-step
For each pattern and node,
$$\ell_{pq}=\sum_j m_{pj}\log P_j(y_{pj}\mid\theta_q;\xi_j^{(g)}),\qquad m_{pj}=\mathbf 1[y_{pj}\ \text{observed}]$$
$$h_{pq}=\operatorname{softmax}_q\big(\ell_{pq}+\log w_{g(p)q}\big)$$
Expected counts ("artificial data"):
$$\bar n_{gq}=\sum_{p\in g}c_p h_{pq},\qquad \bar r_{gjkq}=\sum_{p\in g}c_p h_{pq}\mathbf 1[y_{pj}=k]$$
Marginal log-likelihood: $\ell(\eta)=\sum_p c_p\operatorname{logsumexp}_q(\ell_{pq}+\log w_{g(p)q})$.

Implementation note: $\bar r$ is an einsum of `one_hot(y)` and $c_p h_{pq}$. To avoid
$P\times J\times K\times Q$ memory, contract the `(P,Q)` weights with the `(P,J,K)` one-hot to get
`(G,J,K,Q)` directly.

### 4.3 M-step (items)
The expected complete-data log-likelihood of item $j$ over groups, plus the log prior:
$$Q_j(\xi_j)=\sum_g\sum_q\sum_k \bar r_{gjkq}\log P_{jk}(\theta_q;\xi_j^{(g)})\;+\;\log\pi(\xi_j)$$
- Equality constraints across groups stay **within one item**, so the item-wise decomposition holds
  (multiple groups can still be solved item by item).
- **Decision (v1)**: item-wise Newton–Raphson (gradient and Hessian by autodiff,
  $\dim\xi_j\le 5$ or so). Items of the same item model ("item family") are stacked and `vmap`ed.
  A backtracking line search guarantees that $Q_j$ increases (monotone, as a generalized EM). The
  M-step need not converge fully; a few Newton steps are enough.
- Fixed parameters are not part of $\eta$ (they come from the ParameterTable as constants).
- **Equality constraints across items** (e.g. one common $a$ for all items) break the
  decomposition. Not supported in v1; M5 will add "joint Newton per component of items connected
  by constraints".

### 4.4 M-step (latent distribution)
- **Normal** (when group $g$'s mean or variance is free):
  **Decision (M1)**: the quadrature weights are the density normalized on the grid,
  $w_{gq}(\mu_g,\sigma_g^2)=\phi(\theta_q;\mu_g,\sigma_g^2)/\sum_{q'}\phi(\theta_{q'};\mu_g,\sigma_g^2)$, so
  the M-step maximizes the discretized expected log-likelihood
  $Q_g^{\text{lat}}=\sum_q\bar n_{gq}\log w_{gq}(\mu_g,\sigma_g^2)$ with the same masked Newton as the
  items (§4.3). The closed-form moments
  $\mu_g=\frac{1}{N_g}\sum_q\bar n_{gq}\theta_q$, $\sigma_g^2=\frac{1}{N_g}\sum_q\bar n_{gq}(\theta_q-\mu_g)^2$
  maximize over the continuous density and are not an exact EM step for the discretized marginal
  likelihood (monotonicity is no longer guaranteed, and they are inconsistent with Oakes'
  identity). The difference is negligible on 61 equally spaced points, but exactness comes first.
  Parameters are kept on their natural scale (mean and **variance**); $\sigma^2>0$ is enforced by
  the line search's feasibility check.
  (Confirmed in M1: mirt uses the closed form, which gives Rasch variances up to ~8e-5 apart; see
  VALIDATION.md.)
  **Decision (2026-11-20, reviewed)**: keep only the exact M-step; the mirt-style closed-form
  update is not offered even as an option. The difference comes from the grid cut-off
  ($V_w(\sigma^2)<\sigma^2$); the stationary conditions are
  exact: $V_w(\hat\sigma^2)=\hat m$, closed form: $\hat\sigma^2=\hat m$ ($\hat m$ is the second moment of
  the expected counts). The reported `var` is the parameter of the normal distribution before the
  grid cuts it off; the variance of the distribution on the grid is $V_w(\hat\sigma^2)$. For groups
  with a large mean or variance (e.g. M2 focal groups) the cut-off also affects the mean, so the
  grid width matters.
- **Empirical Histogram (EH)** (`EmpiricalHistogram(*, min_weight=1e-10)`; Woods, 2007; §15.34–§15.38):
  Each group's latent distribution is represented by discrete weights $w_{g1},\dots,w_{gQ}$ on the
  fixed quadrature grid $\theta_1<\dots<\theta_Q$ (default 61 equispaced points on $[-6,6]$),
  enforcing a configurable positive floor $w_{gq}\ge\tau$ ($\tau=\texttt{min\_weight}>0$,
  $\tau Q<1$) so $\log w_{gq}$ is always finite. The EH M-step solves the exact constrained
  complete-data problem $\max_{w\in\Delta^{Q-1},\,w_q\ge\tau}\sum_q\bar n_{gq}\log w_q$ by exact
  KKT water-filling $w_{gq}=\max(\tau,\,c_g^*\bar n_{gq})$ (reducing identically to
  $\bar n_{gq}/N_g$ whenever no node hits the floor).
  **Decision (M3, resolves §12-1)**: the reference group ($g=0$) is standardized to mean 0 and
  variance 1 at **every** EM iteration (mean iff no intercept is fixed in the reference group's
  connected component; variance iff no nonzero slope is fixed in that component; **exception**:
  under FIPC with fixed slopes and intercepts nothing is standardized and all groups' EH weights
  are free; for Rasch/PCM with fixed slopes and free intercepts only the mean is standardized;
  fixed intercepts without a fixed nonzero slope raise `IdentificationError`, because a
  variance-only constraint is not linear in the weights). The standardization is part of the
  reference group's M-step, which solves
  $\max\sum_q\bar n_{0q}\log w_q$ subject to $w_q\ge\tau$, $\sum_q w_q=1$,
  $\sum_q w_q\theta_q=0$ and $\sum_q w_q\theta_q^2=1$ (§15.34). EM therefore stays exactly
  monotone and no item parameters are transformed. (The first implementation rescaled $\theta$
  affinely, transformed the item parameters back and re-gridded every group's weights by CDF
  interpolation, as mirt's `EHW` and Woods (2007) re-grid the histogram; the interpolation error
  made the log-likelihood decrease between iterations and stalled SQUAREM, so it was replaced
  before release.)
  When EH is used in a multiple-group model, **every group uses
  EH** (mixing `EmpiricalHistogram` and `Normal` raises `SpecificationError`).
- Later: Davidian curves (Woods & Lin, 2009), normal mixtures. Both can be added without touching
  the Core by following the `LatentDist` protocol (§6).

### 4.5 Identification rules (checked in the Compile layer)
- Default: the reference group is fixed at $N(0,1)$; focal groups' means and variances are free.
  Freeing a focal group needs anchor items with all parameters equal across groups (for 2PL, one
  item identifies in theory; several are recommended in practice, with a warning when there are
  few).
- **FIPC**: the fixed items set the scale, so every group's mean and variance may be free. With no
  fixed items, the reference group is fixed. Estimating the non-fixed items with the latent
  distribution held at $N(0,1)$ shifts the scale (Kim, 2006), so FIPC updates the latent
  distribution (Normal mean and variance, or EH weights) at every EM iteration by default
  (equivalent to Kim's MWU-MEM).
- **Empirical Histogram (EH)**: when every group uses `EmpiricalHistogram`, the reference group's
  location (mean) is identified by fixed intercepts if present in its connected component, and
  otherwise by the mean-0 constraint of §4.4; its scale (variance) is identified by fixed
  nonzero slopes if present in its connected component, and otherwise by the variance-1
  constraint (so FIPC leaves all EH weights free, Rasch/PCM standardizes mean only, and
  free-slope/intercept models standardize both mean and variance). Every focal group's EH weights
  carry free location and scale and must be linked to an identified group (or have fixed items) via
  anchor slopes and anchor intercepts, else `IdentificationError` (same graph check and single
  location-anchor warning as `Normal`).
- **Rasch and PCM (Decision, M1/M3)**: Rasch and PCM fix $a_j=1$ and free $d_j$ ($d_{jk}$); the
  slopes set the scale, so the **latent variance is estimated** and the mean is fixed at 0 (as
  mirt's `itemtype="Rasch"`, which fits the Rasch model on dichotomous data and the PCM on
  polytomous data: `a1` fixed, `MEAN_1` fixed, `COV_11` free). In a single-group model without an
  explicit latent distribution, **the default is $N(0,\sigma^2)$ (variance free) when every item is
  Rasch or PCM**, and $N(0,1)$ fixed otherwise.
- General rule (checked in M1): a free variance needs at least one item with a fixed slope
  (nonzero value); a free mean needs at least one item with a fixed intercept.
- Unidentified specifications raise `IdentificationError` before fitting.

### 4.6 Convergence and acceleration
- Convergence: largest absolute change of the free parameters (natural-scale $\eta$) between
  successive accepted iterates $<$ `tol` (default 1e-4, the same definition as mirt's `TOL`). The
  histories of the log-likelihood, log posterior, largest change, cumulative EM map evaluations and
  SQUAREM safeguard triggers are kept; monotonicity in the log posterior (log-likelihood without
  priors) is checked by tests.
  **Decision (M1)**: the stopping rule is the largest absolute change only (adding a
  log-likelihood change condition would break the correspondence with mirt).
- **SQUAREM** (Varadhan & Roland, 2008, scheme S3) via `BAEM(squarem=True)` (default `False` in M3
  until wall-time benchmarks against mirt are run; §15.13). If an extrapolated candidate leaves the
  parameter bounds (intersected with any active prior support) or yields a non-finite or lower log
  posterior than $\theta_2$, $\alpha$ is stepped back towards $-1$ up to a fixed number of halvings;
  if still unacceptable, $\theta_2$ is used directly, guaranteeing monotone log posterior
  (§15.11–§15.12, §15.17).
- Both plain EM and SQUAREM run entirely on device in `lax.while_loop` (no Python iteration loop or
  host callbacks). Optional progress output with `jax.debug.callback` is left for later.

### 4.7 MAP (Bayes modal)
- **Decision (M3, §15.14–§15.17)**: public frozen prior distributions `NormalPrior(mean, sd)`,
  `LogNormal(meanlog, sdlog)`, and `Beta(a, b)` (each with `log_pdf` and parameter validation;
  `NormalPrior` avoids clashing with the latent `Normal`).
- Priors are defined on the **natural parameter scale** ($a, d, g, u, d_k$ and so on) without
  a Jacobian term (the posterior mode on the natural scale; verified against R mirt 1.46's
  `LL.Priors`, where `expbeta` evaluates `dbeta(plogis(logit_g), a, b, log = TRUE)` with no logit
  Jacobian; see VALIDATION.md).
- Even if optimization ran on a transformed scale, the objective is evaluated with the
  natural-scale prior.
- Fixed parameters ignore their priors; a parameter shared across groups (same `free_id`) is
  counted **once**. Priors on latent-distribution parameters are out of scope (`SpecificationError`).
- Results report `loglik` (likelihood only) and `log_posterior` (equal to `loglik` without priors)
  separately; `history` records both `loglik` and `log_posterior`. `aic`/`bic` emit a
  `FlexpyirtWarning` when priors are active.

### 4.8 Standard errors
- **Decision**: the default is Oakes' (1999) identity
  $$I(\hat\eta)=-\Big[\tfrac{\partial^2 Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}+\tfrac{\partial^2 Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\Big]_{\eta=\eta'=\hat\eta}$$
  where $Q_{\mathrm{post}}(\eta\mid\eta')=Q(\eta\mid\eta')+\log\pi(\eta)$ ("E-step weights computed
  at $\eta'$, complete-data likelihood plus log prior evaluated at $\eta$"). All derivatives come
  from JAX autodiff, exploiting the block structure of the first term and pattern-chunked category
  co-occurrences for the cross term while reusing a single E-step (§15.20).
- Alternatives: the observed (posterior) information $-\nabla^2_\eta[\ell(\eta)+\log\pi(\eta)]$ by
  chunked forward-over-reverse autodiff (`jax.linearize(jax.grad(...))` over basis batches; also
  used to check Oakes), cross-product ($\sum_p c_p s_p s_p^\top$), sandwich.
- With MAP, both `oakes` and `hessian` add $-\nabla^2_\eta\log\pi(\hat\eta)$ to the information
  matrix and remain mathematically identical. Fixed parameters are not in $\eta$ and are excluded
  automatically.

### 4.9 Quadrature
- **Decision**: the unidimensional default is 61 equally spaced points on $[-6,6]$ (the same grid
  can be shared with EH; Normal weights come from the density).
- Gauss–Hermite is available for Normal only (`Quadrature.gauss_hermite(n)` — not yet implemented).
- **Multidimensional product grids (M4, §15.30)**: 2- and 3-dimensional models use the Cartesian
  product of the 1-D equispaced grid (`Quadrature.grid(n_factors)` → `(Q, D)`). Default points per
  dimension match mirt's `select_quadpts` via `Quadrature.default(n_factors)`: 61 for 1D, 31 per
  dimension ($31^2 = 961$) for 2D, and 15 per dimension ($15^3 = 3{,}375$) for 3D on $[-6, 6]$.
  More than 3 dimensions is out of scope (non-goal).

### 4.10 Starting values
- $d_j$: logit of the proportion correct; $a_j$: converted from the item–total point-biserial
  correlation (classical starting values as in mirt). GRM thresholds: logits of the cumulative
  category proportions.
- **Decision (M1, 2PL/Rasch)**: from the proportion correct $p_j$ (clipped to $[0.01,0.99]$) and
  the item–rest point-biserial correlation $r_j$,
  $a_j^{(0)}=\operatorname{clip}(1.7\,r_j/\sqrt{1-r_j^2},\,0.25,\,3)$ (1 for Rasch),
  $d_j^{(0)}=\operatorname{logit}(p_j)\sqrt{1+(a_j^{(0)}/1.7)^2}$ (to roughly match the marginal
  probability). The variance starts at 1. Starting values are assumed not to affect the converged
  point, and the validation tests check this.
- **Decision (M3, 3PL/4PL)**: $g_j^{(0)}=\min(0.2,\,p_j/2)$ and $u_j^{(0)}=\max(0.95,\,(1+p_j)/2)$
  (4PL) or $1$ (3PL) with $p_j$ clipped to $[0.01,0.99]$, so $g_j^{(0)}<p_j<u_j^{(0)}$; $a_j^{(0)}$
  as for 2PL, and $d_j^{(0)}$ from the rescaled proportion $(p_j-g_j^{(0)})/(u_j^{(0)}-g_j^{(0)})$
  (§15.4).
- **Decision (M3, GPCM/PCM)**: $a_j^{(0)}$ from the item–rest correlation as for 2PL (1 for PCM),
  and $d_{jk}^{(0)}=\sum_{c=1}^{k}\operatorname{logit}(p_{jc})\sqrt{1+(a_j^{(0)}/1.7)^2}$ where
  $p_{jc}=\operatorname{clip}(\hat P(y_j\ge c),\,0.01,\,0.99)$ (§15.9). Reduces identically to the
  2PL/Rasch start when $K=2$.
- User starting values override these through the ParameterTable's `value` column.

## 5. Data structures

### 5.1 ParameterTable (Compile layer, also public)
A long table like mirt's `pars='values'`. **Every constraint is expressed in its columns.**

| Column | Type | Meaning |
|---|---|---|
| `group` | str | Group name |
| `item` | str | Item name (`"__latent__"` for latent-distribution parameters) |
| `param` | str | Internal parameter name such as `a`, `d`, `g`, `u`, `d1..dK` |
| `value` | float | Starting value (the fixed value if not free) |
| `free` | bool | Whether it is estimated |
| `lower`, `upper` | float | Bounds |
| `prior` | `Prior \| None` (object) | Prior distribution on the natural scale (`None` when absent; ignored if fixed) |
| `free_id` | int \| -1 | Free-parameter number. **Equality constraints are expressed by sharing a `free_id`** |

- Users take the table out with `spec.parameter_table(data)`, edit values, fixed flags, bounds,
  priors or `free_id` with pandas and pass it to `fit(..., table=edited)` (keeping mirt's
  pars-editing workflow). Rows sharing a `free_id` must agree on their prior.
- The `FreeMap` holds gather indices from `free_id` into "all parameters (group × family × item ×
  parameter, padded)". $\eta\mapsto\xi$ is one line:
  `xi = fixed_values.at[idx].set(eta[free_id])`.

### 5.2 The pytree passed to the Core
```python
@register_dataclass
@dataclass(frozen=True)
class CompiledProblem:
    y: Array  # (P, J) int, missing -1, padding rows have frequency 0
    counts: Array  # (P,)
    group: Array  # (P,) int
    nodes: Array  # (Q, D) float, D in {1, 2, 3}
    families: tuple[FamilyBlock, ...]  # one block per item kernel (static)
    latent: LatentBlock  # (G, L_D): L_1=2 (mean, var), L_2=5, L_3=9 (means, vars, cors)
    priors: PriorBlock  # per free_id prior parameters
    n_groups: int  # static
```
`families` (including each kernel's `n_factors` and `n_cat`) and `n_groups` are static (jit
static); parameter values and arrays are dynamic.

## 6. Extension protocols

```python
class ItemKernel(Protocol):
    n_factors: int  # 1, 2, or 3
    param_names: tuple[str, ...]  # ("a", "d") or ("a1", ..., "aD", "d")
    n_cat: int
    slope_params: tuple[str, ...]
    intercept_params: tuple[str, ...]
    irt_names: tuple[str, ...]  # display thresholds (empty for D > 1)
    irt_dep_indices: tuple[tuple[int, ...], ...]

    def with_factors(self, n_factors: int) -> ItemKernel: ...
    def irt_thresholds(self, params: Array) -> Array: ...
    def log_prob(self, theta: Array, params: Array) -> Array: ...  # theta (Q, D) -> (Q, K)
```
The design goal: writing `log_prob` on `(Q, D)` nodes is enough for a new item kernel to get the
E-step, M-step (autodiff Newton), Oakes/Hessian SEs, and EAP scoring for $D \in \{1, 2, 3\}$.

## 7. Item models (in stages)
| Model | Internal parameters | Milestone |
|---|---|---|
| Rasch / 2PL | $a, d$ ($P=\sigma(a\theta+d)$) | M1 |
| 3PL / 4PL | $a, d, g, u$ (natural scale, $0<g<u\le 1$; §15.2) | M3 |
| GRM | $a, d_1>\dots>d_{K-1}$ | M2 |
| GPCM / PCM | $a, d_1..d_{K-1}$ | M3 |
| Nominal | M5 or later (undecided) | — |

The display IRT parameterization ($b=-d/a$ for 2PL/Rasch/3PL/4PL, $b_k=-d_k/a$ for GRM, and
step difficulties $b_k=-(d_k-d_{k-1})/a$ with $d_0=0$ for GPCM/PCM) is computed in the Result layer
(`FitResult.items`) with delta-method SEs (`b_se` / `bk_se`) using JAX forward-mode autodiff for
the Jacobian (§15.18).

## 8. Scoring
`FitResult.score(data, *, method="EAP"|"MAP"|"ML"|"WLE"|"PV", group=None, item_names=None, key=None, n_draws=5)`:
- **EAP** (Bock & Mislevy, 1982): posterior mean and posterior SD on the fitted model's quadrature
  grid using the respondent's group's estimated latent distribution as the prior.
- **MAP**: Bayes modal estimate maximizing $\log L(\theta\mid y) + \log\phi(\theta;\mu_g,\sigma_g^2)$
  with the group's estimated normal latent distribution as a continuous prior; SE from the observed
  posterior information $(-\partial^2/\partial\theta^2)^{-1/2}$ at the mode (§15.22).
- **ML**: maximum likelihood estimate maximizing $\log L(\theta\mid y)$; SE from the observed
  information $(-\partial^2\log L/\partial\theta^2)^{-1/2}$. Patterns with no finite maximum return
  $-\infty$ / $+\infty$ (or NaN for all-missing) with SE NaN (§15.23).
- **WLE** (Warm, 1989; Firth, 1993): weighted likelihood estimate solving
  $\partial\log L/\partial\theta + J(\theta)/(2I(\theta)) = 0$ with Fisher test information
  $I(\theta)$ over observed items and $J(\theta) = \partial I/\partial\theta$ from JAX autodiff,
  equivalently maximizing $\log L(\theta\mid y) + \tfrac{1}{2}\log I(\theta)$; finite for extreme
  patterns; SE from the observed curvature of the WLE objective matching mirt's default (§15.24).
- **PV** (Mislevy, 1991): `n_draws` independent draws per respondent from the posterior on a fine
  equispaced grid with piecewise-linear inverse-CDF interpolation, returning columns `pv1..pvM`.
  Requires an explicit `key: jax.Array` (`SpecificationError` if missing); deterministic given `key`
  (§15.25).

Missing responses are dropped from the likelihood (and from $I(\theta)$ in WLE) for every method.
All methods compress unique patterns with power-of-two bucket padding and run inside `jit` with
data-independent shapes (§15.21).

## 9. Performance
- Pattern compression and bucket padding limit recompilation.
- `vmap` per item family. Mixed item types loop only over the (static) families.
- The default target is "fast enough on CPU". GPU should help for large problems
  (N ≥ 10^5, J ≥ 100).
- Benchmarks: wall time against R mirt (`TOL=1e-4`, default settings and validation settings) in
  [benchmarks/](../benchmarks/README.md); committed Apple M5 CPU results are in
  [benchmarks/results/2026-11-21-apple-m5.md](../benchmarks/results/2026-11-21-apple-m5.md).
  Summary of that single-machine measurement (61 quadrature points on $[-6, 6]$, CPU only; XLA CPU
  may use multiple threads while mirt is single-threaded by default):
  - **Warm fits**: faster than mirt default across all six tested cases (`0.16×–0.60×` with plain
    EM, `0.12×–0.46×` with `squarem=True`), with $|\Delta \log L| \le 1.5 \times 10^{-3}$ at
    `TOL = 1e-4`.
  - **Cold fits**: a first call including JAX/XLA compilation takes ~0.8–1.1 s for EM (more with
    Oakes SE on `20000 × 40`), so mirt is faster for a single one-off fit without SEs (all six
    cases); with Oakes SEs flexpyirt's cold fit is faster on four of the six.
  - **Oakes SE at large `(N, J)`**: on `2PL (20000 × 40)`, `se="oakes"` takes ~3.1–3.5 s warm vs.
    ~0.28–0.39 s for EM alone.
  - **SQUAREM trade-off**: `BAEM(squarem=True)` reduces EM evaluations by 33–69% and warm EM time
    (`0.38×–0.90×` of plain EM on every case, biggest gains on 2-group 2PL and 3PL), but adds
    ~0.05–0.4 s of cold compilation overhead (more with Oakes SE on `20000 × 40`). The default (`squarem=False`) is kept unchanged pending a maintainer
    decision on cold vs. warm/slow-model priority.

## 10. Errors and warnings
- `SpecificationError`: inconsistent specification (fixing an unknown item, wrong number of
  categories, ...)
- `IdentificationError`: not identified (focal group free without anchors, ...)
- `ConvergenceWarning`: maximum iterations reached, frequent SQUAREM safeguard
- `HeywoodWarning`: boundary solution — emitted after fitting when a free item parameter ends
  within `1e-4` of a finite bound (e.g. $g\to 0$ or $u\to 1$) or a free slope has $|a|>20$ (§15.6).

## 11. Testing (details in VALIDATION.md)
- Unit: probabilities of each `log_prob` sum to 1, finite-difference gradient checks, E-step count
  conservation ($\sum_q\bar n_{gq}=N_g$).
- Property: hypothesis generates random parameters and data; EM must increase monotonically.
- Recovery: parameter recovery in simulations (bias, RMSE).
- Agreement: estimates, SEs and log-likelihood match R mirt within tolerance.

## 12. Open questions
1. ~~**Identification for EH**~~ → **Decision (2026-11-21)**: standardize the reference group to
   mean 0 and variance 1 at every iteration (not under FIPC), implemented as a moment-constrained
   EH M-step (§4.4, §15.34).
2. Many groups (e.g. school × grade). Keep the per-group latent distributions `vmap`-able.
3. ~~Missing-data design~~ → exclusion under MAR only (§14.8, 2026-11-20).
4. ~~Package name~~ → `flexpyirt` (2026-11-20).
5. How to implement the M-step with equality constraints across items (§4.3).

## 13. Implementation decisions (M1)
1. **Newton in the M-step**: every gradient and Hessian comes from JAX autodiff (`jax.grad` /
   `jax.hessian`); there are no hand-written analytic derivatives. Only the loop (step update and
   line search) is our own: Newton masked to the free coordinates. Fixed coordinates get Hessian
   $-I$ and gradient 0 so they do not move. If the Newton direction is not an ascent direction
   (the Hessian is not negative definite), switch to the gradient. Step sizes
   $2^{-k}\ (k=0,\dots,29)$ are evaluated together with `vmap`, and the largest one that does not
   decrease the objective and stays within bounds is taken (none: do not move). This shape keeps
   Python branches out of jit.
2. **FreeMap**: per family, `free_index (J_f, P_f)` (-1 if fixed) and `fixed_values`.
   $\xi=\text{where}(\text{free\_index}\ge0,\ \eta[\text{free\_index}],\ \text{fixed})$.
   After the M-step, write back with `eta.at[free_index].set(xi, mode="drop")`. In M1, sharing a
   `free_id` between different items is a `SpecificationError` (§4.3; supported in M5).
3. **Explicitly reject what M1 does not cover**: multiple groups (`group=`), missing data (NaN) and
   non-equispaced quadrature come in M2 or later. The E-step itself is written with a missing-data
   mask, but the data layer rejects missing data until validation (V9) is done. Items where
   everyone gave the same response make the MLE diverge and raise `SpecificationError`.
   (Multiple groups and missing data were enabled in M2, §14.)
4. **Dependencies**: pandas is required (results are DataFrames); polars is optional.
5. **SE**: the information matrix for $\eta$ (natural scale). Oakes differentiates
   $Q(\eta\mid\eta')$ with `jax.jacfwd(jax.grad(Q, 0), argnums=(0, 1))`. The direct method is
   `jax.hessian` of the marginal log-likelihood. The two agree as an identity, so the internal
   consistency test compares them near machine precision (relative 1e-6).

## 14. Implementation decisions (M2)
1. **Per-group item parameters**: the ParameterTable has group × item × parameter rows, with a
   separate `free_id` per group by default (as in mirt `multipleGroup`).
   `equal_across_groups(items=..., params=None)` gives the selected items' rows the same `free_id`
   across groups. In the Core, each family has `free_index (G, J_f, P_f)`.
2. **Item-local parameters**: to solve item $j$ including its parameters shared across groups,
   each item gets a local vector $u_j$ (padded to length $G\cdot P_f$), expanded with
   `local_index (G, J_f, P_f)` as $\xi^{(g)}_j=u_j[\text{local}]$. The M-step maximizes
   $\sum_g Q_{jg}(\xi_j^{(g)}(u_j))$ over $u_j$ with the masked Newton (§13.1), `vmap`ed. `free_id`s
   do not cross items (until M5), so the local vector stays within the item.
3. **Groups in the E-step**: $\ell_{pq}$ is computed for every group with
   `einsum("pjk,gjqk->pgq")`, then each pattern's own group is selected (no
   $P\times J\times Q\times K$ intermediate). $\bar r$ is `(G, J_f, K, Q)`.
4. **Declaring groups**: `Model.with_groups(**latents)` defines the groups in keyword order; the
   first is the reference group. The values of `fit(..., group=labels)` must match these group
   names (missing or extra groups are a `SpecificationError`). Passing `group=` without
   `with_groups` is also an error.
5. **Identification (multiple groups)**: for the mean, build a graph with groups as nodes and
   shared intercept `free_id`s as edges, and require every connected component to contain "a group
   with a fixed mean" or "a fixed intercept" (likewise for the variance with slopes). This is a
   necessary condition, not a sufficient one. Warn when a group with a free mean is connected by a
   single anchor.
6. **FIPC**: `Model.fix_from(bank, items=..., parameterization="slope-intercept" | "irt")` fixes the
   item parameters, and the user frees the latent distribution with `with_latent(Normal.free())`
   (the default does not change; the §4.5 identification check allows it). The IRT form ($b$) is
   converted with $d=-ab$ (2PL; for GRM, $b_k$ → $d_k=-ab_k$).
7. **GRM** (Samejima, 1969): $P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$, categories
   $0..K-1$. Log-probabilities are computed stably as $\log\sigma(-z_1)$ and $\log\sigma(z_{K-1})$ for
   the end categories and
   $\log\sigma(z_k)+\operatorname{log1mexp}(\log\sigma(z_{k+1})-\log\sigma(z_k))$ in between (log1mexp
   clamps each branch's input to its safe domain with a double `where` so gradients have no NaN).
   The ordering constraint is kept by the line search rejecting non-finite candidates. One family
   per $K$ of `GRM(n_cat=K)`. Items with a category that never occurs in the data have a diverging
   threshold and raise `SpecificationError`.
8. **Missing data (MAR)**: NaN is coded as `-1` and the E-step's one-hot row is all zeros, which
   drops it from the likelihood ($m_{pj}$ in §4.2). An item missing for everyone is an error.
   Starting values use observed values only. Not-reached responses are not handled (this settles
   §12-3).
9. **Grid for validation**: validation cases with a free latent mean or variance (V4, V5) use
   81 points on $[-8,8]$ (the same 0.2 spacing as the default) in both mirt and flexpyirt, as set
   out in VALIDATION.md.

## 15. Implementation decisions (M3)
1. **3PL and 4PL item models**: `ThreePL()` ($P(y=1\mid\theta)=g+(1-g)\sigma(a\theta+d)$;
   Birnbaum, 1968) and `FourPL()` ($P(y=1\mid\theta)=g+(u-g)\sigma(a\theta+d)$; Barton & Lord, 1981)
   share one kernel, `FourPLKernel`, with internal parameters `(a, d, g, u)` (same names as mirt),
   so 3PL and 4PL items form one family (like Rasch and 2PL share `DichotomousKernel`). `ThreePL`
   fixes `u = 1` by default; `FourPL` frees `u`.
2. **Natural scale and bounded active-set Newton**: `g` and `u` stay on the natural scale in $\eta$
   with bounds `0 < g < 1`, `0 < u < 1` when free (fixed `g = 0` or `u = 1` is allowed, and
   $g < u$ is enforced). Standard errors are therefore on the natural scale directly (when
   comparing with mirt, whose `coef(..., printSE=TRUE)` reports `logit(g)` / `logit(u)` and
   logit-scale SEs, convert mirt's SEs with the delta method $SE_g = g(1-g)\,SE_{\mathrm{logit}(g)}$).
   In the masked Newton M-step (`_newton.py`), a coordinate whose step would leave `(lower, upper)`
   moves halfway to its bound instead of shrinking the step of every other coordinate, and a free
   coordinate within $10^{-8}$ of a bound whose gradient points outwards is held for that step
   ($\varepsilon$-active set; Bertsekas, 1982) so a boundary asymptote does not block `a` and `d`
   from converging. The objective-ascent check still guarantees EM monotonicity.
3. **Numerical stability at $g=0$ and $u=1$**: log-probabilities are computed as
   $$\log P_1 = \operatorname{logaddexp}\big(\log g,\ \log(u-g)+\log\sigma(z)\big),\qquad
   \log P_0 = \operatorname{logaddexp}\big(\log(1-u),\ \log(u-g)+\log\sigma(-z)\big)$$
   with $z=a\theta+d$. Using the double-`where` pattern (replacing the argument of $\log$ with $1$
   where $g=0$ or $u=1$ and selecting the core branch), holding `g = 0` and/or `u = 1` fixed
   returns exactly the 2PL log-probabilities and finite gradients and Hessians for every parameter,
   so masked Newton never sees NaN/inf in a fixed coordinate. Invalid asymptotes ($u\le g$, $g<0$,
   or $u>1$) yield NaN and are rejected by the line search (and rejected up front in
   `validate_table`).
4. **Starting values**: for proportion correct $p_j$ clipped to $[0.01,0.99]$,
   $g_j^{(0)}=\min(0.2,\,p_j/2)$ and $u_j^{(0)}=\max(0.95,\,(1+p_j)/2)$ (4PL) or $1$ (3PL), ensuring
   $g_j^{(0)}<p_j<u_j^{(0)}$; $a_j^{(0)}$ uses the same item–rest correlation formula as 2PL (§4.10),
   and $d_j^{(0)}$ uses the 2PL intercept formula applied to the rescaled proportion
   $(p_j-g_j^{(0)})/(u_j^{(0)}-g_j^{(0)})$.
5. **Display**: `FitResult.items` keeps adding `b = -d/a`; `g`, `u`, `g_se`, `u_se` appear
   automatically from the parameter table. `Model.fix_from()` passes `g` and `u` through unchanged
   in both `"slope-intercept"` and `"irt"` parameterizations, and allows omitting `u` for 3PL items.
6. **HeywoodWarning**: after fitting, `fit()` checks every free item parameter and emits one
   `flexpyirt.errors.HeywoodWarning` naming all affected items whenever a free parameter ends within
   `1e-4` of a finite bound (`g` or `u` near 0 or 1) or a free slope has $|a|>20$.
7. **Identification**: `g` and `u` carry neither latent scale nor location
   (`slope_params=("a",)`, `intercept_params=("d",)`), so fixing asymptotes alone does not identify
   a free latent mean or variance.
8. **GPCM and PCM item models**: `GPCM(n_cat)` (Muraki, 1992) and `PCM(n_cat)` (Masters, 1982)
   share one kernel per `n_cat`, `GPCMKernel(n_cat)`, with categories `0..K-1` (`n_cat >= 2`
   validated) and slope–intercept form matching mirt's `itemtype="gpcm"` (scoring coefficients
   $\mathrm{ak}_k=k$ and $d_0=0$ fixed, not parameters):
   $$P(y=k\mid\theta)=\frac{\exp(k\,a\theta+d_k)}{\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)},\qquad d_0=0.$$
   Internal parameters are `(a, d1, ..., d{K-1})`. Log-probabilities are computed with
   `jax.nn.log_softmax` (numerically stable, smooth derivatives, no ordering constraint on `d`).
   `PCM(n_cat)` fixes `a = 1` by default, and the single-group default latent rule (§4.5) is
   extended so that a model whose items are all `Rasch` or `PCM` gets $N(0,\sigma^2)$ with the
   variance free (matching mirt's `itemtype="Rasch"` on polytomous data). With `n_cat = 2`,
   `GPCM(2)` equals `TwoPL` and `PCM(2)` equals `Rasch` exactly.
9. **GPCM/PCM starting values, identification, and estimability**: $a_j^{(0)}$ uses the item–rest
   correlation as for 2PL (`1.0` for PCM); each step starts at
   $\delta_{jc}^{(0)}=\operatorname{logit}(p_{jc})\sqrt{1+(a_j^{(0)}/1.7)^2}$ from the cumulative
   proportion $p_{jc}=\operatorname{clip}(\hat P(y_j\ge c),\,0.01,\,0.99)$, summed to
   $d_{jk}^{(0)}=\sum_{c=1}^{k}\delta_{jc}^{(0)}$. Identification uses `slope_params=("a",)` and
   `intercept_params=("d1", ..., "d{K-1}")`. Every category `0..K-1` must be observed in the data
   for items with free parameters (same rule as GRM, enforced in `_check_estimable`).
10. **GPCM/PCM display and FIPC**: `FitResult.items` displays step difficulties
    $b_k=-(d_k-d_{k-1})/a$ (with $d_0=0$) as `b1`…`b{K-1}` for GPCM/PCM items while leaving
    2PL/Rasch/3PL/4PL (`b = -d/a`) and GRM (`bk = -dk/a`) unchanged, even when mixed in the same
    model. `Model.fix_from(..., parameterization="irt")` inverts this via
    $d_k=-a\sum_{c=1}^{k}b_c$, and allows omitting `a` for PCM items.
11. **SQUAREM scheme S3 (Varadhan & Roland, 2008)**: implemented as `_core.squarem` inside
    `lax.while_loop`. Starting from accepted iterate $\theta_0$, two EM map applications give
    $\theta_1=F(\theta_0)$ and $\theta_2=F(\theta_1)$, with $r=\theta_1-\theta_0$,
    $s=\theta_2-\theta_1$, $v=s-r$, and step length $\alpha=\min(-\|r\|_2/\|v\|_2,\,-1)$
    (falling back to $\alpha=-1$ when $\|v\|_2=0$ or $-\|r\|_2/\|v\|_2$ is non-finite). Writing
    $\delta=\alpha+1\le 0$ turns $\theta'(\alpha)=\theta_0-2\alpha r+\alpha^2 v$ into the
    algebraically identical $\theta'(\alpha)=\theta_2-2\delta s+\delta^2 v$, which equals
    $\theta_2$ exactly at $\delta=0$ (no cancellation) and makes halving towards $-1$
    ($\alpha\leftarrow(\alpha-1)/2$) simply $\delta\leftarrow\delta/2$.
12. **SQUAREM monotonicity and feasibility safeguard**: when $\alpha<-1$, $\theta'(\alpha)$ is
    checked against strict per-coordinate bounds `(lower, upper)` extracted once before the loop
    from `FamilyBlock.slot_lower/upper` and `LatentBlock.lower/upper` (`_core._eta_bounds`), so
    feasibility uses the exact same bounds as the M-step (including latent variance $>0$,
    $0<g,u<1$, and any user-edited table bounds). If strictly inside bounds, `marginal_loglik` is
    evaluated via `lax.cond` (skipped when out of bounds) and accepted iff finite and
    $\ge\max(\ell(\theta_2),\ell(\theta_0))$ — this also rejects GRM disordered thresholds and
    4PL $g\ge u$ because their kernels return NaN. On rejection, an inner `lax.while_loop` halves
    $\delta$ up to 10 times. An accepted $\theta'$ takes one stabilizing EM step
    $\theta_{\text{new}}=F(\theta')$ (3 EM map evaluations for the iteration); if all halvings
    fail, $\theta_{\text{new}}=\theta_2$ is used directly (2 EM map evaluations). Because $F$ is
    monotone, every accepted iterate satisfies $\ell(\theta_{\text{new}})\ge\ell(\theta_0)$.
13. **SQUAREM API default, history, and warnings**: `BAEM(squarem: bool = False)` keeps plain EM
    as the default in M3 because wall-time benchmarks against mirt (including JIT overhead and
    small- vs. large-problem trade-offs) are scheduled later in M3. The stopping rule stays
    $\max_i|\eta^{(t+1)}_i-\eta^{(t)}_i|<\text{tol}$. `FitResult` adds `n_em_evaluations` (total
    calls to $F$; equals `n_iter` for plain EM) and `n_safeguards` (iterations on which the
    safeguard stepped $\alpha$ back or fell back to $\theta_2$; 0 for plain EM). `history` keeps
    a uniform schema (`iter`, `loglik`, `max_change`, `em_evaluations`, `safeguard`) regardless of
    `squarem`. `ConvergenceWarning` reports safeguard/fallback counts when `max_iter` is reached,
    and also fires on a converged SQUAREM fit when full fallbacks to $\theta_2$
    (`3 * n_iter - n_em_evaluations`) are at least 5 and exceed 50% of iterations (partial
    step-backs that still accelerate do not trigger spurious warnings).
14. **Prior distributions and natural-scale MAP objective (M3, §4.7)**: `NormalPrior(mean, sd)`,
    `LogNormal(meanlog, sdlog)`, and `Beta(a, b)` are public frozen dataclasses in `priors.py`
    validating their parameters (`SpecificationError`) and providing `log_pdf(x)`. Priors are
    evaluated on the natural parameter scale without a Jacobian term — verified in R against
    `mirt` 1.46's `LL.Priors` (`expbeta` evaluates `dbeta(plogis(logit_g), a, b, log = TRUE)` with
    no logit Jacobian; note mirt 1.46's `DerivativePriors` / `BL.LL` defects documented in
    VALIDATION.md). `Beta.log_pdf` computes $\log B(a, b)$ via `gammaln(a) + gammaln(b) - gammaln(a + b)`
    rather than `jax.scipy.special.betaln`, which loses ~1e-7 accuracy in float64.
15. **Prior attachment, table editing, and scope**: `Model.prior(*, param, dist, items=None)`
    attaches `dist` to `param` of the selected items across all groups. The ParameterTable adds a
    `prior` column (`object` dtype, `None` when absent); `fit(..., table=...)` accepts and validates
    edited priors (rejecting non-`Prior` objects, priors on `__latent__` rows, starting values
    outside the prior's open support for free parameters, bounds disjoint from the prior's support,
    and conflicting priors on rows sharing a `free_id`). Fixed parameters ignore their prior, and
    a parameter shared across groups (`free_id`) is counted **once**. Priors on latent-distribution
    parameters are out of scope and raise `SpecificationError`.
16. **Branch-free Core prior representation**: `CompiledProblem` carries `priors: PriorBlock`
    (`kind: (n_free,) int32`, `param1/param2: (n_free,) float64`, one entry per unique `free_id`),
    evaluated elementwise by `eval_log_prior` using double-`where` guards on every branch so
    unselected branches and unused padding slots (`PRIOR_NONE = 0`) never produce NaN/inf
    derivatives. Inside `m_step`, each item gathers its local-slot priors from `prob.priors` via
    `slot_free_index` and adds them to `q_item`, keeping the item-wise decomposition (§14.2) and
    counting shared parameters once. `compile_problem` intersects `slot_lower`/`slot_upper` with
    each free slot's prior support (`(0, inf)` for `LogNormal`, `(0, 1)` for `Beta`) so both
    bounded Newton and SQUAREM respect prior support without altering the ParameterTable's model
    domain bounds.
17. **MAP results, history, SQUAREM safeguard, and posterior SEs**: `FitResult.loglik` stays
    marginal log-likelihood only; `FitResult.log_posterior` reports $\ell(\hat\eta)+\log\pi(\hat\eta)$
    (equal to `loglik` when no active priors exist). `history` keeps uniform columns
    `("iter", "loglik", "log_posterior", "max_change", "em_evaluations", "safeguard")`, where
    `log_posterior` is non-decreasing across EM and SQUAREM iterations and SQUAREM's safeguard
    compares `log_posterior`. `FitResult.aic` and `FitResult.bic` still use `loglik` and emit a
    `FlexpyirtWarning` when at least one free parameter has a prior. `information_oakes` adds
    `log_prior(prob, e)` to $Q(\eta\mid\eta')$ and `information_hessian` differentiates
    `log_posterior(prob, e)`, adding $-\nabla^2_\eta\log\pi(\hat\eta)$ identically to both.
18. **Delta-method SEs for display IRT thresholds (M4, §7)**: each `ItemKernel` defines
    `irt_names` (e.g. `("b",)` or `("b1", ..., "b{K-1}")`), `irt_dep_indices` (which internal
    parameters enter each threshold), and pure JAX `irt_thresholds(params)` ($b = -d/a$ for
    2PL/Rasch/3PL/4PL, $b_k = -d_k/a$ for GRM, and $b_k = -(d_k - d_{k-1})/a$ with $d_0 = 0$ via
    `jnp.diff` for GPCM/PCM). In `FitResult.items`, a per-kernel cached jitted
    `(irt_thresholds, jax.jacfwd(irt_thresholds))` evaluates both the display values and their
    Jacobian $J = \partial g/\partial\xi$ without hand-written derivatives. For each threshold $m$,
    $\operatorname{Var}(g_m(\hat\xi)) \approx J_m\Sigma_\xi J_m^\top$ (Casella & Berger, 2002;
    Oehlert, 1992) is computed over the free parameters entering $g_m$ by indexing `FitResult.vcov`
    with their `free_id`s. Fixed parameters contribute zero variance (so Rasch/PCM with $a=1$ fixed
    give `b_se = d_se` / `b1_se = d1_se`, and PCM step differences include $\operatorname{Cov}(d_{k-1},d_k)$);
    across-group equalities share `free_id` and automatically use the shared covariance entries
    (including partial equalities such as shared `a` with group-specific `d`). When `vcov` is
    `None`, non-PD (NaN), or no parameter entering a threshold is free (e.g. FIPC fixed items or
    3PL with `a, d` fixed and `g` free), `b_se` / `bk_se` is NaN.
19. **Result summaries (`FitResult.summary()` and `FitResult._repr_html_()`) (M4, §2 ④)**:
    `FitResult.summary(*, digits=4)` returns a plain-text report with model/estimation type
    (`MMLE` or `MAP`), data/parameter counts, groups, convergence/iterations/EM evaluations (and
    SQUAREM safeguards when nonzero), SE method, quadrature, log-likelihood, log-posterior, AIC/BIC,
    latent distribution table, and item parameter table (including `b_se` / `bk_se`). When priors
    are active, an inline note warns that AIC/BIC use the marginal log-likelihood at the MAP
    estimate rather than the MLE, without emitting duplicate runtime `FlexpyirtWarning`s.
    `FitResult._repr_html_()` renders the same information in plain, self-contained HTML/CSS with
    no external assets, light/dark-safe styling, and `html.escape()` on all names. `__repr__`
    remains a concise single-line summary.
20. **Structured Oakes and chunked Hessian performance (M3, §4.8, §9)**: the initial
    `information_oakes` ran `jax.jacfwd(jax.grad(q, argnums=0), argnums=(0, 1))(eta, eta)` over the
    full problem, pushing all $n_{\text{free}}$ tangents simultaneously through the E-step and
    materializing `(n_free, P, Q)` intermediates alongside dense $J$-item einsums per direction.
    On a 2PL with $N=20{,}000, J=40$ ($19{,}926$ unique patterns padded to $P=32{,}768$, $Q=61$,
    $n_{\text{free}}=80$), it took ~2.66 s warm (5.46 s cold) and ~6.11 GB peak RSS delta while EM
    itself took ~0.43 s. The structured implementation keeps Oakes' identity exact and derives every
    kernel, latent, and prior derivative strictly via JAX autodiff:
    - **Single E-step**: `_e_step_details` computes `counts` and posterior responsibilities $h_{pq}$
      once and is shared with `e_step`.
    - **Block-structured first term** (`_oakes_own_hessian`): $\partial^2 Q_{\mathrm{post}}/\partial\eta\,\partial\eta^\top$
      is evaluated block-by-block with `vmap(jax.hessian)` over each item's local parameter slots
      (matching `m_step`, never differentiating fixed boundary coordinates), `vmap(jax.hessian)`
      over each group's latent `(mean, var)`, and diagonal `vmap(jax.grad(jax.grad(...)))` over
      `prob.priors`, scatter-adding every block into `(n_free, n_free)`.
    - **Structured cross term** (`_oakes_cross_hessian`): writing
      $Q(\eta\mid\eta')=\sum_p c_p\sum_q h_{pq}(\eta')\,\psi_{pq}(\eta)$ with
      $h_{pq}(\eta')=\operatorname{softmax}_q(\psi_{pq}(\eta'))$ yields
      $\partial^2 Q/\partial\eta\,\partial\eta'^\top|_{\eta'=\eta}=\sum_p c_p(\sum_q h_{pq}s_{pq}s_{pq}^\top-\bar s_p\bar s_p^\top)$
      ($s_{pq}=\nabla_\eta\psi_{pq}$, $\bar s_p=\sum_q h_{pq}s_{pq}$; Oakes, 1999; Louis, 1982).
      Per-item and latent score tables come from `jax.jacfwd` and are centered across $q$ per
      `(g, j, k, s)` using $\bar r$ weights (algebraically invariant in $s_{pq}-\bar s_p$ while
      preventing cancellation). For each group, a pattern-chunked `lax.scan` skips zero-count
      padding chunks via `lax.cond` and accumulates node-level category co-occurrences
      $C_{g,q}=\sum_{p\in g}c_p h_{pq}x_p x_p^\top$ and $\sum_{p\in g}c_p\bar s_p\bar s_p^\top$
      (bounding the `(Q, B, D)` workspace to ~32 MB, without materializing `(P, Q, n_free)` or
      `(P, D)`), followed by a `lax.scan` over $q$ contracting $\sum_q L_{g,q}^\top C_{g,q} L_{g,q}$
      without materializing `(G, Q, D, n_free)`.
    - **Chunked direct Hessian**: `information_hessian` evaluates
      `jax.linearize(jax.grad(log_posterior), eta)` on batches of 8 basis vectors via `lax.scan`
      instead of all $n_{\text{free}}$ directions at once.
    - **Indicative measurements** (same 2PL $N=20{,}000, J=40$ case, CPU, median of 5 warm runs in
      isolated subprocesses, measured on a shared machine):
      `information_oakes`: warm **2.658 s → 0.0843 s** (~31.5× faster), cold **5.464 s → 0.365 s**
      (~15× faster), peak RSS delta **6,110 MB → 9.6 MB**;
      `information_hessian`: warm **0.768 s → 0.602 s** (~1.28× faster), cold **1.215 s → 0.898 s**,
      peak RSS delta **4,991 MB → 612 MB**.
      Across V1–V11 (2PL, Rasch, GRM, 2-group anchors, FIPC, missing data, 3PL, GPCM, PCM, MAP
      priors), agreement with the previous `jacfwd(grad(q))` implementation is
      $\max|\Delta I|/\max|I|\le 8.8\times 10^{-16}$ (elementwise $\le 3.1\times 10^{-12}$;
      `tests/test_oakes.py`).
    - **Idle-machine measurement** (maintainer, same 2PL $20{,}000\times40$ data, warm median of 5
      fits including EM ≈ 0.41 s): `fit(..., se="oakes")` 3.39 s → **0.50 s** (the Oakes part
      ≈ 2.98 s → ≈ 0.09 s); `se="hessian"` 1.16 s → 1.07 s; covariance matrices agree with the
      previous implementation to $8.5\times10^{-16}$ (relative to the largest entry).
21. **Scoring API and shared Core structure (M4, §8)**: `FitResult.score(data, *, method="EAP", group=None, item_names=None, key=None, n_draws=5)` supports `Literal["EAP", "MAP", "ML", "WLE", "PV"]`.
    EAP (`_core.eap`) is unchanged. The new scoring kernels live in `_scoring.py` (pure JAX Core
    layer), compress unique `(group, pattern)` pairs and pad to power-of-two buckets as for EAP,
    drop missing responses (`y == -1`) via masking, and run inside `jit` with shapes independent of
    the data values. Passing `key` to a non-PV method, omitting or passing a non-PRNG `key` for PV,
    or passing an invalid `method` or `n_draws` raises `SpecificationError`.
22. **MAP scoring (`method="MAP"`, M4, §8)**: maximizes
    $f_{\mathrm{MAP}}(\theta)=\log L(\theta\mid y) - (\theta-\mu_g)^2/(2\sigma_g^2)$ using the
    respondent's group's estimated normal latent distribution $(\mu_g,\sigma_g^2)$ as a
    **continuous** prior (not the quadrature grid), with
    $\mathrm{SE}=(-f''_{\mathrm{MAP}}(\hat\theta))^{-1/2}$ at the mode. Optimization runs a
    1-D safeguarded Newton ascent (`_newton_1d`, `vmap`-ed over unique patterns, starting from EAP):
    derivatives $f', f''$ come strictly from JAX autodiff; when $f''\ge -10^{-12}$ (e.g. non-concave
    3PL/4PL regions) the direction falls back to gradient ascent; candidate steps $2^{-k}\delta$
    ($k=0..29$, with step magnitude capped at `10.0`) are evaluated together with `vmap` so the
    objective never decreases. An all-missing row has $\log L\equiv 0$ and therefore returns the
    continuous prior mode $\mu_g$ and SD $\sigma_g$ (consistent with flexpyirt's EAP returning the
    prior mean and SD; note mirt strips completely empty rows to `NA` for all methods).
23. **ML scoring (`method="ML"`, M4, §8)**: maximizes $f_{\mathrm{ML}}(\theta)=\log L(\theta\mid y)$
    over $[-20, 20]$ (matching mirt's default `max_theta = 20`) via the same `vmap`-ed safeguarded
    1-D Newton starting from EAP, with observed-information SE
    $(-f''_{\mathrm{ML}}(\hat\theta))^{-1/2}$ (matching mirt's default `expected.info = FALSE`).
    Extreme-pattern conventions: all-missing returns `(NaN, NaN)`; all observed items in their
    lowest-$\theta$ category (category `0` for $a_j>0$, `K_j-1` for $a_j<0$) return `(-inf, NaN)`;
    all observed items in their highest-$\theta$ category return `(+inf, NaN)`. For non-concave
    models (3PL/4PL), patterns whose likelihood still has no finite interior maximum (e.g. low-score
    patterns where $\log L(\theta)$ increases monotonically toward $-\infty$ due to guessing) are
    also detected — when $|\hat\theta|$ hits the `±20` rail, $f''(\hat\theta)\ge 0$,
    $\mathrm{SE}>30$ (mirt's threshold in `mirt:::ML`), or $f(\pm 20) > f(\hat\theta)$ — and return
    `±inf` with SE NaN. The sign of `±inf` comes from whichever endpoint dominates when
    $f(\pm 20) > f(\hat\theta)$, and otherwise follows $\operatorname{sign}(\hat\theta)$ (matching
    `mirt:::ML`'s `Inf * sign(est)`). In particular, on bimodal 3PL likelihoods where the
    $-\infty$ lower-asymptote plateau strictly exceeds a strictly concave interior local mode,
    flexpyirt intentionally returns the true global supremum `(-inf, NaN)` (the actual maximum
    likelihood), whereas mirt 1.46.1's local `nlm` search starting from EAP stops at the interior
    local mode (25 of 3,790 unique patterns in `v11_3pl`; this check also cleanly handles the
    boundary cases where `nlm`'s finite-difference Hessian fails to trigger `SEest > 30`; see
    VALIDATION.md V13).
24. **WLE scoring (`method="WLE"`, M4, §8)**: Warm's (1989) weighted likelihood estimator is
    implemented by maximizing the Jeffreys/Firth (1993) penalized log-likelihood
    $f_{\mathrm{WLE}}(\theta)=\log L(\theta\mid y)+\tfrac{1}{2}\log I(\theta)$ on $[-20, 20]$ with
    the same `vmap`-ed safeguarded 1-D Newton starting from EAP, where
    $I(\theta)=\sum_{j:\,y_j\ge 0}\sum_k P_{jk}(\theta)\,(\partial\log P_{jk}/\partial\theta)^2$
    is the observed-item Fisher test information computed strictly via JAX autodiff (`jax.jacfwd` on
    each kernel's `log_prob`, working for every item family with no hand-written derivatives). Its
    stationary condition is identically $\partial\log L/\partial\theta + J(\theta)/(2I(\theta)) = 0$
    with $J(\theta)=\partial I/\partial\theta$, matching `mirt:::WLE.mirt` across 2PL, Rasch, GRM,
    GPCM, PCM, 3PL, and 4PL (for exponential-family models $J=\partial I/\partial\theta$ equals
    Warm's $\sum P'P''/P$ identically; for 3PL/4PL/GRM it matches Firth's Jeffreys-prior bias
    reduction and mirt). Extreme patterns (all-lowest / all-highest) have finite WLEs; all-missing
    patterns have $I(\theta)=0$ and return `(NaN, NaN)`. The SE is
    $(-f''_{\mathrm{WLE}}(\hat\theta))^{-1/2}$ from JAX autodiff, matching mirt's default
    `fscores(..., method="WLE", full.scores.SE=TRUE)` (whereas Warm's asymptotic $I(\hat\theta)^{-1/2}$
    corresponds to mirt's `expected.info=TRUE` and differs on short tests and extreme patterns; see
    VALIDATION.md V13).
25. **Plausible values (`method="PV"`, M4, §8)**: draws `n_draws` (default `5`) independent samples
    per respondent from the posterior $p(\theta\mid y_i;\mu_{g(i)},\sigma_{g(i)}^2)$, returned as a
    DataFrame with columns `pv1..pvM`. To produce continuous draws without quadrature quantization
    while keeping shapes static and fast inside `jit`, the posterior density is evaluated per unique
    (compressed, power-of-two-padded) pattern on a fine equispaced grid of `501` nodes spanning the
    fitted model's quadrature interval `[quad.lower, quad.upper]`, integrated via the trapezoidal
    rule to a piecewise-linear CDF on $[0, 1]$, and inverted via branch-free `searchsorted(..., side="right")`
    linear interpolation against `jax.random.uniform(key, (N_pad, n_draws))` (padded to a
    power-of-two respondent bucket `N_pad` and sliced to `N`). Respondents sharing the same response
    pattern reuse the same posterior CDF but receive independent uniform variates and thus
    independent draws; all-missing rows draw from the group's prior. Requires an explicit JAX PRNG
    `key` (`jax.random.key` or `jax.random.PRNGKey`, `SpecificationError` if absent or invalid) and
    is fully deterministic given `key`.
26. **`FitResult` JSON I/O API and versioned format (M4, API §7)**:
    `FitResult.to_json(path=None, *, indent=2) -> str` serializes a fitted result to a strict JSON
    string and also writes UTF-8 text to `path` (`str` or `os.PathLike`) when given;
    `FitResult.from_json(source) -> FitResult` loads from a file path or JSON string without
    refitting. Plain JSON is used (no `pickle` and never `eval`). The top-level object carries
    `"format": "flexpyirt.FitResult"`, `"schema_version"` (2 since multidimensional models were added in §15.33; version 1 files are still read), and
    `"flexpyirt_version": flexpyirt.__version__`. `FitResult` stores `_method: BAEM` alongside
    `_model` and `_quadrature` so estimation settings round-trip as well. `from_json` calls
    `check_x64()` at entry (raising `PrecisionError` when 64-bit floating point is disabled, since
    scoring requires float64).
27. **Strict JSON and exact float64 round-tripping (M4)**:
    Serialization uses `json.dumps(..., allow_nan=False)` so output is always strict RFC 8259 JSON,
    and deserialization passes a rejecting `parse_constant` to `json.loads` so bare `NaN`/`Infinity`
    literals are rejected with `SpecificationError`. Finite IEEE-754 float64 values are written as
    standard JSON numbers (Python's `repr`-exact shortest decimal representation, round-tripping
    every finite float64 bit-for-bit, including `-0.0`). Non-finite values (`NaN`, `+Inf`, `-Inf`)
    in float-typed fields (such as unbounded `lower`/`upper`, fixed or unavailable `se`, or non-PD
    `vcov`) are encoded explicitly as `"NaN"`, `"Infinity"`, and `"-Infinity"`; only float-typed
    schema slots decode these strings, so string fields such as item or group names are unaffected.
28. **Explicit registries and full state preservation (M4)**:
    Item models (`"2PL"`, `"Rasch"`, `"3PL"`, `"4PL"`, `"GRM"`, `"GPCM"`, `"PCM"`, recording
    `n_cat` for polytomous models), latent distributions (`"Normal"`), priors (`"NormalPrior"`,
    `"LogNormal"`, `"Beta"`), quadrature (`"equispaced"`), and estimation methods (`"BAEM"`) are
    (de)serialized through explicit name registries in `_json.py`. Unknown format tags, schema
    versions, registry names, invalid JSON, or inconsistent table/shape invariants raise
    `SpecificationError` with a clear message. Both the full `Model` specification (item names and
    order, item models, `fixed`, `latent`, `groups`, `equal`, `priors`) and the validated
    `ParameterTable` with `estimate` and `se` (preserving any user `table=` edits), `vcov`,
    `loglik`/`log_posterior`, convergence info and `history`, `quadrature`, `method`, `se_method`,
    and `n_obs` are stored; `_eta` is reconstructed from the validated free rows of `parameters`
    (checking contiguous `0..n_free-1` `free_id`s, identical estimates on shared `free_id`s, and
    matching `vcov` shape) so estimates have a single source of truth. Loaded results reproduce
    every public attribute, `items`, `latent`, `summary()`, `_repr_html_()`, `aic`/`bic`, and
    `score()` for every method identically.
29. **Confirmatory 2–3 dimensional MIRT specification and kernels (M4, §4.9, §6)**:
    `Model.with_factors(factors, *, correlated=True)` declares a confirmatory loading pattern over
    2 or 3 named factors (`Mapping[str, ItemSelector]`). An item may load on multiple factors
    (cross-loading); every factor requires at least one item, every item must load on at least one
    factor, and slopes on unloaded factors are fixed at 0. Calling `Model.uniform(...)` or
    `Model.from_items(...)` without `with_factors()` stays strictly unidimensional (`n_factors=1`)
    and bit-for-bit unchanged. All four kernels (`DichotomousKernel`, `FourPLKernel`,
    `GradedKernel`, `GPCMKernel`) take a static `n_factors: int = 1` and `with_factors(n_factors)`
    and evaluate the linear predictor $z = \mathbf a^\top\boldsymbol\theta + d$ (using `params[0] *
    theta[:, 0]` for $D=1$ and `theta @ params[:D]` for $D \in \{2, 3\}$), so all seven item models
    (`TwoPL`, `Rasch`, `ThreePL`, `FourPL`, `GRM`, `GPCM`, `PCM`) work in 2D and 3D automatically.
    For $D > 1$, internal item parameters are named `a1..aD` and `d` (`d1..d{K-1}` for GRM / GPCM /
    PCM), matching mirt, and unidimensional display thresholds (`b` / `bk`) are omitted. Multiple
    groups for $D > 1$, `with_latent()` for $D > 1$, and `fix_from(..., parameterization="irt")`
    for $D > 1$ raise `SpecificationError`.
30. **Product quadrature and exact MVN latent M-step (M4, §4.4, §4.9)**:
    `Quadrature.default(n_factors)` selects mirt's `select_quadpts` defaults (`61` for 1D, `31` per
    dimension for 2D → $961$ nodes, `15` per dimension for 3D → $3{,}375$ nodes on $[-6, 6]$), and
    `Quadrature.grid(n_factors)` returns the `(Q, D)` Cartesian product grid (while `nodes()` stays
    the 1-D `(n_points,)` axis). `CompiledProblem.nodes` has shape `(Q, D)`. For $D \in \{2, 3\}$,
    the multivariate normal latent distribution is parameterized on the natural scale in
    `ParameterTable` and `LatentBlock` as `(mean_1..D, var_1..D, cor_ij...)` ($L_2 = 5$, $L_3 = 9$,
    pairs $1 \le i < j \le D$): $\Sigma_{ii} = v_i > 0$ (`lower = 0`) and
    $\Sigma_{ij} = \rho_{ij}\sqrt{v_i v_j}$ with $\rho_{ij} \in (-1, 1)$ (`lower = -1, upper = 1`).
    When variances are fixed at 1 (default for 2PL / GRM / 3PL / 4PL / GPCM), `cor_ij` equals
    $\Sigma_{ij}$ identically (matching mirt's `COV_ji` and its SE); when every item is `Rasch` or
    `PCM`, `var_1..var_D` are free by default alongside `cor_ij`. `mvn_log_weights(nodes, params)`
    dispatches to the unchanged 1D `normal_log_weights` for $D = 1$, and for $D \in \{2, 3\}$
    computes normalized grid log-weights via `log_softmax` using the closed-form quadratic form
    ($D = 2$) or Cholesky + `solve_triangular` ($D = 3$), checking positive definiteness of the
    correlation matrix via Sylvester's criterion (leading principal minors $> 0$) with double-`where`
    guards so derivatives at valid PD points are smooth and NaN-free while non-PD candidates return
    NaN and are rejected by bounded Newton line search and the SQUAREM safeguard. As in 1D (§4.4),
    the latent M-step maximizes the exact discretized $Q_g^{\text{lat}} = \sum_q \bar n_{gq}\log w_{gq}$
    (keeping EM strictly monotone and Oakes' identity exact).
31. **Identification, sign indeterminacy, and starting values for $D > 1$ (M4, §4.5, §4.10)**:
    `_check_identification_mirt` checks necessary conditions before fitting: every factor must have
    at least one active loading (free or fixed nonzero); a free `var_m` requires at least one fixed
    nonzero slope `a{m}`; a free `mean_m` requires at least one item loading on factor $m$ with a
    fixed intercept; for confirmatory rotational identification, each factor with $c_m$ free
    correlations must have at least $c_m$ fixed slopes (typically 0s) and no two factors may share
    identical free loading sets without fixed nonzero slopes (Bollen, 1989; Reckase, 2009).
    **Sign indeterminacy**: negating $\theta_m$, all $a_{jm}$, $\mu_m$, and $\rho_{mm'}$ ($m' \neq m$)
    leaves the marginal likelihood invariant (as in 1D). Slopes remain unbounded (`(-inf, inf)`) so
    reverse-scored items and natural-scale SEs are unaffected by artificial bounds; orientation is
    anchored to the positive direction by computing strictly positive default starting slopes
    $a_{jm}^{(0)} \in [0.25, 3.0]$ from factor-specific item–rest correlations scaled by
    $1/\sqrt{L_j}$ ($L_j$ = number of factors item $j$ loads on; intercepts and asymptotes use
    $\|\mathbf a_j^{(0)}\|$), starting free correlations at `0.0` ($R^{(0)} = I_D$, always PD), and
    rejecting edited tables where all starting slopes on a factor are 0 (`SpecificationError`).
32. **Multidimensional results, Oakes SEs, and EAP scoring (M4, §4.8, §8)**:
    The Core E-step, M-step, SQUAREM, structured Oakes (`_oakes_own_hessian`,
    `_oakes_cross_hessian`), and chunked Hessian work identically for $D \in \{1, 2, 3\}$ with
    shape-static pytrees and autodiff only. `FitResult.items` displays `a1..aD`, `d` (`d1..`), and
    their SEs; `FitResult.latent` displays `mean_1..D`, `var_1..D`, `cor_ij`, and their SEs.
    `FitResult.score(data, method="EAP")` returns columns `theta_<factor>` and `se_<factor>` in
    declared factor order; `method="MAP"` is supported as well (§15.39), while `ML`, `WLE` and
    `PV` raise `SpecificationError` for $D > 1$.
33. **JSON I/O for multidimensional `FitResult` (schema version 2, M4)**: the model object in the
    JSON payload stores `"factors"` (`null`, or a list of `{"name", "items"}` in declared order)
    and `"correlated_factors"`, so `to_json()` / `from_json()` round-trip 2–3 dimensional results
    exactly (`items`, `latent`, `parameters`, `vcov`, `summary()`, and EAP scores). On load the
    factor structure is rebuilt through `Model.with_factors()`, so the same validation applies as
    when it was declared, and fixed values, equalities and priors are checked against the
    multidimensional parameter names (`a1`..`aD`); a model with factors must not carry `latent`
    or `groups`. `schema_version` is 2; version-1 files (written before factors were stored) are
    still read and must not contain the two new keys. `model_parameter_schema()` repeats the slope
    bounds for `a1`..`aD` exactly as `build_table()` does.
34. **Empirical Histogram (`EmpiricalHistogram`) specification, KKT floor, and moment-constrained reference M-step (M3, §4.4, §4.5, resolves §12-1)**:
    `EmpiricalHistogram(*, min_weight: float = 1e-10)` is a public frozen dataclass exported from
    `flexpyirt`, validating `0 < min_weight < 1` at construction and `min_weight * Q < 1` at
    compile time (`SpecificationError`). Mixing `EmpiricalHistogram` and `Normal` across groups in
    `Model.with_groups()` raises `SpecificationError`: when EH is used, every group uses EH.
    Weights initialize from the standard normal density on the quadrature grid projected onto the
    floored simplex. In the EH M-step, every unstandardized group (focal groups, and the reference
    group under FIPC) solves $\max_{w\in\Delta^{Q-1},\,w_q\ge\tau}\sum_q\bar n_{gq}\log w_q$
    ($\tau=\texttt{min\_weight}$) exactly via KKT water-filling
    $w_{gq}=\max(\tau,\,c_g^*\bar n_{gq})$ (computed in $O(Q\log Q)$ with `jnp.sort`; reduces
    identically to $\bar n_{gq}/N_g$ when no node hits the floor). The standardized reference group
    (`standardize_mean` iff no fixed intercept in its connected component; `standardize_var` iff no
    fixed nonzero slope in it; neither under FIPC; mean only for Rasch/PCM; variance without mean
    is rejected with `IdentificationError`) adds the linear moment constraints
    $\sum_q w_q\theta_q=0$ and (if `standardize_var`) $\sum_q w_q\theta_q^2=1$
    (`eh_constrained_weights`). With $\phi_q=(1,\theta_q,\theta_q^2)$ and multipliers $\lambda$,
    the KKT solution is $w_q=\max\{\tau,\,p_q/\lambda^\top\phi_q\}$ ($p_q=\bar n_{0q}/N_0$), where
    $\lambda$ minimizes the smooth convex dual
    $G(\lambda)=\sum_q[p_q\log w_q(\lambda)-(\lambda^\top\phi_q)w_q(\lambda)]+\lambda^\top c$ on
    $\{\lambda^\top\phi_q>0\ \forall q\}$ (Boyd & Vandenberghe, 2004, §5.2). It is solved by damped
    Newton ($K\le 3$ unknowns, `lax.while_loop`, backtracking that never leaves the domain) from
    $\lambda=(1,0,0)$, until the constraint residual $\nabla G$ is below $10^{-13}$. Because this
    is the exact maximizer of the latent Q-function over the identified set, EM is exactly
    monotone, the reference histogram has mean 0 and variance 1 after every iteration, and item
    parameters are never transformed. The minimizer is interior whenever every $p_q>0$ (E-step
    counts always are); if some $p_q=0$ and the optimum needs mass on those nodes, the dual
    infimum lies on the boundary, Newton stops early, and `fit()` emits `ConvergenceWarning` when
    the final reference moments miss their targets by more than $10^{-8}$.
35. **EH parameter counts and conditional item standard errors (M3, §4.8)**:
    EH weights are counted as parameters in `FitResult.n_params` (and hence `aic`/`bic`):
    $Q - 1 - \mathbf{1}[\text{standardize\_mean}] - \mathbf{1}[\text{standardize\_var}]$ for the
    reference group ($Q-3$ when both mean and variance are standardized, $Q-2$ for Rasch/PCM,
    $Q-1$ under FIPC) plus $Q-1$ for every focal group. For standard errors, the full joint
    observed information including all $G(Q-1)$ discrete histogram weights on the default 61-point
    grid is mathematically rank-deficient (analytic IRT item likelihoods span a much smaller
    subspace of $\mathbb R^Q$, giving numerical rank $\approx 20\ll 60$ and zero/negative
    curvature at boundary/floored nodes). Therefore, `information_oakes` and `information_hessian`
    compute the **conditional observed (posterior) information for the free item parameters holding
    the converged EH weights $\hat w$ fixed**, keeping `vcov` strictly positive definite,
    preserving the exact identity `oakes == hessian` (relative $<10^{-10}$), and matching R
    `mirt`'s `vcov(m)` shape (`n_free_items × n_free_items`) and `SE.type = "Richardson"`
    definition (which holds `EHPrior` fixed in `mirt:::BL.LL`). The `ParameterTable` contains item
    rows only for EH models (no `__latent__` rows, because EH weights depend on `Quadrature`,
    which is passed to `fit()`, not `Model.parameter_table()`).
36. **SQUAREM on the square-root weight scale for EH (M3, §4.6)**:
    When `BAEM(squarem=True)` is used with `EmpiricalHistogram`, SQUAREM extrapolates the joint
    vector $(\eta_{\text{items}},\,z)$ with $z_{gq}=\sqrt{w_{gq}}$, and candidates map back by
    $w_{gq}\propto z_{gq}^2$ (always on the simplex, whatever the sign of $z$) followed by the same
    M-step map used for expected counts (KKT floor; for the standardized reference group the
    moment-constrained I-projection of §15.34, which returns a feasible input unchanged). A
    candidate whose projection misses the moment constraints by more than $10^{-10}$ is set to NaN,
    so the monotonicity safeguard (`log_posterior(cand) >= max(log_posterior(theta2),
    log_posterior(theta0))`, halving $\delta$ up to 10 times and falling back to $\theta_2$ if
    needed) rejects it. The square-root scale makes the multinomial Fisher information isotropic
    ($4I$). The centered log-weight (softmax) scale decided first lets near-empty nodes, which
    barely affect the likelihood, dominate the step length ($\log w$ moves by whole units at
    $w\approx10^{-9}$): on V6a and two free 2PL examples (21 and 61 nodes) the safeguard fired on
    >99% of iterations, against <1% on the square-root scale, which also converged 3.5× faster
    on the 21-node example and reached a higher log-likelihood in the same iteration budget on the
    others. Both plain EM and SQUAREM measure convergence by
    $\max(\max_i|\Delta\eta_i|,\;\max_{g,q}|\Delta w_{gq}|)<\text{tol}$ (probability-scale weight
    changes alongside natural-scale item changes).
37. **EH scoring (M3, §8)**:
    `FitResult.score()` with `EmpiricalHistogram` supports `"EAP"` (posterior mean and SD on the
    fitted quadrature grid using each group's estimated EH weights), `"ML"` and `"WLE"`
    (likelihood-only, identical to Normal), and `"PV"` (piecewise-linear interpolation of each
    group's EH density from `prob.nodes` onto the 501-node fine grid, renormalized, followed by
    trapezoidal inverse-CDF sampling). `method="MAP"` raises `SpecificationError` for EH models
    because MAP mode-finding and curvature SE require a twice-differentiable continuous latent
    prior (note R `mirt` 1.46.1's `fscores(..., method="MAP")` silently ignores `EHW` weights and
    falls back to $N(0,1)$).
38. **EH reporting and JSON I/O (M3, §2 ④, §15.26–§15.28)**:
    For EH models, `FitResult.latent` reports per-group summary moments up to the 4th — columns
    `("group", "mean", "var", "sd", "skewness", "kurtosis")`, where `kurtosis` is Fisher's excess
    kurtosis $\mu_4/\sigma^4 - 3$ (`0` for Normal) — and the new attribute
    `FitResult.latent_weights` returns a DataFrame with columns `("group", "node", "weight")`
    (`None` for `Normal` models). Both the moments table and the weights table are rendered in
    `FitResult.summary()` and `FitResult._repr_html_()`. In `_json.py`, `"EmpiricalHistogram"` is
    registered in the latent-distribution registry (`{"type": "EmpiricalHistogram", "min_weight": ...}`)
    and EH results include a `"latent_weights"` field (shape `(G, Q)`, validated on load as finite,
    strictly positive, summing to 1, and matching `n_groups` and `quadrature.n_points`; omitted and
    forbidden for `Normal` models so existing schema-version-1 payloads remain unchanged). `_n_eh_params`
    is recomputed deterministically from `(model, parameters, quadrature)` on load.
39. **MAP scoring for multidimensional models (M4, §8)**: `score(method="MAP")` for
    $D \in \{2, 3\}$ maximizes
    $f(\boldsymbol\theta)=\log L(\boldsymbol\theta\mid y)-\tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu)$
    with the estimated multivariate normal ($\Sigma_{ij}=\rho_{ij}\sqrt{v_iv_j}$) as a continuous
    prior, starting from EAP, by safeguarded $D$-dimensional Newton (`_newton_nd`: Newton
    direction when the autodiff Hessian is negative definite, otherwise the gradient; step norm
    capped at 10; step sizes $2^{-k}$, $k=0..29$, evaluated together so $f$ never decreases).
    SEs are $\big[(-\nabla^2 f(\hat{\boldsymbol\theta}))^{-1}\big]_{mm}^{1/2}$, as in mirt's
    `fscores(method = "MAP")`; all-missing rows return the prior mean and $\sqrt{\Sigma_{mm}}$.
    `ML`, `WLE` and `PV` still raise `SpecificationError` for $D > 1$ (ML/WLE need per-factor
    handling of diverging directions).

## References
- Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series*, RR-81-20.
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM Journal on Control and Optimization*, 20(2), 221–246.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores* (pp. 397–479). Addison-Wesley.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431–444.
- Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56(2), 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Oehlert, G. W. (1992). A note on the delta method. *The American Statistician*, 46(1), 27–29.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory*. Springer.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Swaminathan, H., & Gifford, J. A. (1986). Bayesian estimation in the three-parameter logistic model. *Psychometrika*, 51, 589–601.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427–450.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
