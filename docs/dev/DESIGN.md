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
  - flexible latent distributions (Normal / Empirical Histogram / Davidian curves / normal
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
- Make each group's response matrix (missing as `-1`) unique to get frequencies $c_p$. With
  sampling weights, $c_p$ is the sum of the weights of the rows with pattern $p$ (§17.4).
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
  The Newton runs on the natural scale with a bounded active set (§15.2) by default;
  `BAEM(m_step_scale="unconstrained")` runs it on logit/log scales of the bounded coordinates
  instead, with the same objective (§15.47).
- Fixed parameters are not part of $\eta$ (they come from the ParameterTable as constants).
- **Equality constraints across items** (e.g. one common $a$ for all items) break the
  decomposition. **Decision (M5, §16.1)**: items connected by shared `free_id`s form a component
  that is solved by one joint masked Newton over the union of its free parameters; items outside
  such components keep the item-wise `vmap` path.

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
- **Davidian curves** (`DavidianCurve(degree, *, mean=0.0, var=1.0, free_mean=False, free_var=False)`;
  Zhang & Davidian, 2001; Woods & Lin, 2009; §16.4):
  Single-group, unidimensional seminonparametric density $h_k(z;\boldsymbol\phi)=P_k(z;\boldsymbol\phi)^2\varphi(z)$
  expanded in normalized probabilists' Hermite polynomials $He_j(z)/\sqrt{j!}$ with unit-sphere
  coefficients $\mathbf c(\boldsymbol\phi)\in S^k$ parameterized by $k\in\{1,\dots,10\}$ angles
  $\boldsymbol\phi=(\phi_1,\dots,\phi_k)$ ($\boldsymbol\phi=\mathbf 0$ gives $N(0,1)$ non-singularly).
  Analytically standardized by the closed-form continuous moments $(\mu_Z(\boldsymbol\phi),\sigma_Z^2(\boldsymbol\phi))$
  of $h_k$ so that $\theta=\mu+\sigma(Z-\mu_Z)/\sigma_Z$ has exact continuous mean $\mu$ and
  variance $\sigma^2$ for every $\boldsymbol\phi$. Quadrature weights are the density normalized on
  the grid, and the latent M-step maximizes $Q^{\text{lat}}=\sum_q \bar n_q\log w_q(\mu,\sigma^2,\boldsymbol\phi)$
  with the existing masked Newton (`_newton.py`), keeping EM monotone (non-decreasing), SQUAREM
  applicable, and Oakes' identity exact.
- **Normal mixtures** (`NormalMixture(n_components, *, mean=0.0, var=1.0, free_mean=False,
  free_var=False, means=None, variances=None, weights=None)`; Mislevy, 1984; §16.7):
  Single-group, unidimensional finite mixture of $k\in\{2,\dots,10\}$ normal components with one
  common set of item parameters. The shape is parameterized as in mirt's `dentype = "mixture-k"`
  (raw component 1 is $N(0,1)$; components $c\ge 2$ have raw means, raw variances and logits
  $\log(\pi_c/\pi_1)$) and the mixture is analytically standardized by its exact continuous
  moments so that $\theta$ has mean $\mu$ and variance $\sigma^2$, as for Davidian curves. Each
  component is normalized on the grid and then mixed; the latent M-step maximizes
  $Q^{\text{lat}}=\sum_q \bar n_q\log w_q$ with the same masked Newton.

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
  SQUAREM always works on the natural scale, whatever `m_step_scale` is (§15.47).
- Both plain EM and SQUAREM run entirely on device in `lax.while_loop` (no Python iteration loop or
  host callbacks). Optional progress output with `jax.debug.callback` is left for later.

### 4.7 MAP (Bayes modal)
- **Decision (M3, §15.14–§15.17)**: public frozen prior distributions `NormalPrior(mean, sd)`,
  `LogNormalPrior(mean_log, sd_log)`, and `BetaPrior(a, b)` (each with `log_pdf` and parameter validation;
  `NormalPrior` avoids clashing with the latent `Normal`).
- Priors are defined on the **natural parameter scale** ($a, d, g, u, d_k$ and so on) without
  a Jacobian term (the posterior mode on the natural scale; verified against R mirt 1.46's
  `LL.Priors`, where `expbeta` evaluates `dbeta(plogis(logit_g), a, b, log = TRUE)` with no logit
  Jacobian; see VALIDATION.md).
- Even if optimization runs on a transformed scale (`BAEM(m_step_scale="unconstrained")`,
  §15.47), the objective is evaluated with the natural-scale prior.
- **Decision (#15, §15.48)**: `LogitNormalPrior(mean_logit, sd_logit)` is a fourth prior, a
  distribution on $(0,1)$ with $\operatorname{logit}X\sim N(\mu,s^2)$. Its natural-scale density
  contains the logit Jacobian because that is part of the density of $X$ itself, not a change
  of optimization scale. `LogitNormalPrior(..., mode_scale="logit")` drops that Jacobian to give
  the logit-metric posterior mode of flexMIRT and mirt (§15.48).
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
  used to check Oakes).
- **Decision (M7, §17.5)**: `se="crossprod"` inverts the cross-product of the per-pattern scores,
  $B_{\mathrm{post}}=\sum_p c_p s_p s_p^\top-\nabla^2_\eta\log\pi(\hat\eta)$, and `se="sandwich"` is
  the robust covariance $H^{-1}B_{\mathrm{post}}H^{-1}$ with $H$ the observed (posterior)
  information above (Huber, 1967; White, 1982).
- With MAP, both `oakes` and `hessian` add $-\nabla^2_\eta\log\pi(\hat\eta)$ to the information
  matrix and remain mathematically identical. Fixed parameters are not in $\eta$ and are excluded
  automatically.

### 4.9 Quadrature
- **Decision**: the unidimensional default is 61 equally spaced points on $[-6,6]$ (the same grid
  can be shared with EH; Normal weights come from the density).
- Gauss–Hermite is available for Normal latent distributions (`Quadrature.gauss_hermite(n)`, §15.44).
- **Multidimensional product grids (M4, §15.30)**: 2- and 3-dimensional models use the Cartesian
  product of the 1-D equispaced grid (`Quadrature.grid(n_factors)` → `(Q, D)`). Default points per
  dimension match mirt's `select_quadpts` via `Quadrature.default(n_factors)`: 61 for 1D, 31 per
  dimension ($31^2 = 961$) for 2D, and 15 per dimension ($15^3 = 3{,}375$) for 3D on $[-6, 6]$.
  More than 3 dimensions is out of scope (non-goal) for product grids; bifactor models reduce
  any number of specific factors to $(1+1)$-dimensional grids (§17.1).

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
`families` (including each kernel's `n_factors` and `n_categories`) and `n_groups` are static (jit
static); parameter values and arrays are dynamic.

## 6. Extension protocols

```python
class ItemKernel(Protocol):
    n_factors: int  # 1, 2, or 3
    param_names: tuple[str, ...]  # ("a", "d") or ("a1", ..., "aD", "d")
    n_categories: int
    slope_params: tuple[str, ...]
    intercept_params: tuple[str, ...]

    def with_factors(self, n_factors: int) -> ItemKernel: ...
    def log_prob(self, theta: Array, params: Array) -> Array: ...  # theta (Q, D) -> (Q, K)

    # optional: display thresholds (empty for D > 1)
    irt_names: tuple[str, ...]
    irt_dependency_indices: tuple[tuple[int, ...], ...]

    def irt_thresholds(self, params: Array) -> Array: ...
```
The design goal: writing `log_prob` on `(Q, D)` nodes is enough for a new item kernel to get the
E-step, M-step (autodiff Newton), Oakes/Hessian SEs, and EAP scoring for $D \in \{1, 2, 3\}$.
This protocol, with `ItemModel`, is public: §20 fixes the contract for user-defined item models.

## 7. Item models (in stages)
| Model | Internal parameters | Milestone |
|---|---|---|
| Rasch / 2PL | $a, d$ ($P=\sigma(a\theta+d)$) | M1 |
| 3PL / 4PL | $a, d, g, u$ (natural scale, $0<g<u\le 1$; §15.2) | M3 |
| GRM | $a, d_1>\dots>d_{K-1}$ | M2 |
| GPCM / PCM | $a, d_1..d_{K-1}$ | M3 |
| Nominal | $a, \mathrm{ak}_1..\mathrm{ak}_{K-2}, d_1..d_{K-1}$ ($\mathrm{ak}_0=0, \mathrm{ak}_{K-1}=K-1, d_0=0$; §16.6) | M5 |

The display IRT parameterization ($b=-d/a$ for 2PL/Rasch/3PL/4PL, $b_k=-d_k/a$ for GRM, and
step difficulties $b_k=-(d_k-d_{k-1})/a$ with $d_0=0$ for GPCM/PCM) is computed in the Result layer
(`FitResult.items`) with delta-method SEs (`b_se` / `bk_se`) using JAX forward-mode autodiff for
the Jacobian (§15.18).

## 8. Scoring
`FitResult.score(data, *, method="EAP"|"MAP"|"ML"|"WLE"|"PV"|"EAPsum", group=None, item_names=None, key=None, n_draws=5)`:
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
- **EAPsum** (Thissen, Pommerich, Billeaud & Williams, 1995): posterior mean and SD given the
  summed score, looked up in the summed-score table of the respondent's group; NaN for a
  respondent with any missing response; unidimensional `Normal` models only (§18.2).

Missing responses are dropped from the likelihood (and from $I(\theta)$ in WLE) for every method
except EAPsum. All other methods compress unique patterns with power-of-two bucket padding and run inside `jit` with
data-independent shapes (§15.21).

## 9. Performance
- Pattern compression and bucket padding limit recompilation.
- `vmap` per item family. Mixed item types loop only over the (static) families.
- The default target is "fast enough on CPU". GPU should help for large problems
  (N ≥ 10^5, J ≥ 100).
- Benchmarks: wall time against R mirt (`TOL=1e-4`, default settings and validation settings) in
  [benchmarks/](../../benchmarks/README.md); committed Apple M5 CPU results are in
  [benchmarks/results/2026-11-21-apple-m5.md](../../benchmarks/results/2026-11-21-apple-m5.md).
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
2. ~~Many groups (e.g. school × grade). Keep the per-group latent distributions `vmap`-able.~~
   → **Decision (2026-11-26, #79)**: group-blocked pattern layout; the E-step is mapped over
   fixed-size single-group blocks and the latent M-step stays `vmap`ped over the stacked
   `(G, L)` latent parameters (§26).
3. ~~Missing-data design~~ → exclusion under MAR only (§14.8, 2026-11-20).
4. ~~Package name~~ → `flexpyirt` (2026-11-20).
5. ~~How to implement the M-step with equality constraints across items (§4.3)~~ → joint
   Newton per connected component of items (§16.1, 2026-11-21).

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
6. **FIPC**: `Model.fix_from(bank, items=..., parameterization="slope_intercept" | "irt")` fixes the
   item parameters, and the user frees the latent distribution with `with_latent(Normal.free())`
   (the default does not change; the §4.5 identification check allows it). The IRT form ($b$) is
   converted with $d=-ab$ (2PL; for GRM, $b_k$ → $d_k=-ab_k$).
7. **GRM** (Samejima, 1969): $P(y\ge k\mid\theta)=\sigma(a\theta+d_k)$, $d_1>\dots>d_{K-1}$, categories
   $0..K-1$. Log-probabilities are computed stably as $\log\sigma(-z_1)$ and $\log\sigma(z_{K-1})$ for
   the end categories and
   $\log\sigma(z_k)+\operatorname{log1mexp}(\log\sigma(z_{k+1})-\log\sigma(z_k))$ in between (log1mexp
   clamps each branch's input to its safe domain with a double `where` so gradients have no NaN).
   The ordering constraint is kept by the line search rejecting non-finite candidates. One family
   per $K$ of `GRM(n_categories=K)`. Items with a category that never occurs in the data have a diverging
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
   `BAEM(m_step_scale="unconstrained")` replaces this box with logit/log coordinates (§15.47);
   the default stays this active-set Newton.
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
   in both `"slope_intercept"` and `"irt"` parameterizations, and allows omitting `u` for 3PL items.
6. **HeywoodWarning**: after fitting, `fit()` checks every free item parameter and emits one
   `flexpyirt.errors.HeywoodWarning` naming all affected items whenever a free parameter ends within
   `1e-4` of a finite bound (`g` or `u` near 0 or 1) or a free slope has $|a|>20$.
7. **Identification**: `g` and `u` carry neither latent scale nor location
   (`slope_params=("a",)`, `intercept_params=("d",)`), so fixing asymptotes alone does not identify
   a free latent mean or variance.
8. **GPCM and PCM item models**: `GPCM(n_categories)` (Muraki, 1992) and `PCM(n_categories)` (Masters, 1982)
   share one kernel per `n_categories`, `GPCMKernel(n_categories)`, with categories `0..K-1` (`n_categories >= 2`
   validated) and slope–intercept form matching mirt's `itemtype="gpcm"` (scoring coefficients
   $\mathrm{ak}_k=k$ and $d_0=0$ fixed, not parameters):
   $$P(y=k\mid\theta)=\frac{\exp(k\,a\theta+d_k)}{\sum_{c=0}^{K-1}\exp(c\,a\theta+d_c)},\qquad d_0=0.$$
   Internal parameters are `(a, d1, ..., d{K-1})`. Log-probabilities are computed with
   `jax.nn.log_softmax` (numerically stable, smooth derivatives, no ordering constraint on `d`).
   `PCM(n_categories)` fixes `a = 1` by default, and the single-group default latent rule (§4.5) is
   extended so that a model whose items are all `Rasch` or `PCM` gets $N(0,\sigma^2)$ with the
   variance free (matching mirt's `itemtype="Rasch"` on polytomous data). With `n_categories = 2`,
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
    `LogNormalPrior(mean_log, sd_log)`, and `BetaPrior(a, b)` are public frozen dataclasses in `priors.py`
    validating their parameters (`SpecificationError`) and providing `log_pdf(x)`. Priors are
    evaluated on the natural parameter scale without a Jacobian term — verified in R against
    `mirt` 1.46's `LL.Priors` (`expbeta` evaluates `dbeta(plogis(logit_g), a, b, log = TRUE)` with
    no logit Jacobian; note mirt 1.46's `DerivativePriors` / `BL.LL` defects documented in
    VALIDATION.md). `BetaPrior.log_pdf` computes $\log B(a, b)$ via `gammaln(a) + gammaln(b) - gammaln(a + b)`
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
    each free slot's prior support (`(0, inf)` for `LogNormalPrior`, `(0, 1)` for `BetaPrior`) so both
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
    `irt_names` (e.g. `("b",)` or `("b1", ..., "b{K-1}")`), `irt_dependency_indices` (which internal
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
    `"format": "flexpyirt.FitResult"`, `"schema_version"` (4 since the naming refactor of §15.46, 5 since bifactor models, §17.1, 6 since two-tier models, §17.2, 7 since sampling weights, §17.4; older versions are still read), and
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
    `n_categories` for polytomous models), latent distributions (`"Normal"`), priors (`"NormalPrior"`,
    `"LogNormalPrior"`, `"BetaPrior"`), quadrature (`"equispaced"`), and estimation methods (`"BAEM"`) are
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
    PCM), matching mirt, and unidimensional display thresholds (`b` / `bk`) are omitted. 
    `with_latent()` for $D > 1$ and `fix_from(..., parameterization="irt")` for $D > 1$ raise
    `SpecificationError` (multiple groups for $D > 1$: §15.43).
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
    declared factor order; `"MAP"` (§15.39), `"ML"`, `"WLE"` (§15.41) and `"PV"` (§15.42) are
    supported as well.
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
    `ML` and `WLE` for $D > 1$ are in §15.41 and `PV` in §15.42.
40. **S-X2 item fit (M5, `FitResult.item_fit`)**: Orlando and Thissen's (2000, 2003) S-X2 for
    single-group unidimensional models with a `Normal` latent distribution and complete data.
    Expected proportions
    $E_j[s,k]=\sum_q P_{jk}(\theta_q)L_{-j}(s-k\mid\theta_q)w_q/\sum_q L(s\mid\theta_q)w_q$ come
    from the Lord–Wingersky recursion (Lord & Wingersky, 1984) on the fitted quadrature grid and
    latent weights, for summed scores $1,\dots,S-1$, and are scaled by the observed number of
    respondents at each score. Sparse cells are collapsed by a line-by-line port of
    `mirt:::collapseCells` (mirt 1.46–1.48; its heuristics, including the row flags computed
    before single-observation rows are merged, are reproduced exactly so the statistic and the
    degrees of freedom match), and $df_j$ = kept cells − rows − free parameters of item $j$ −
    free latent parameters, as in mirt (which subtracts the latent parameters from every item).
    RMSEA and the $\chi^2$ p-value follow mirt. Tables are post-fit NumPy bookkeeping (data
    dependent shapes, outside `jit`); category probabilities come from the JAX item kernels.
    EH and missing-data cases raise `SpecificationError` until they are
    validated (mirt imputes missing data and ignores EH weights for S-X2); multiple groups:
    §15.45; confirmatory multidimensional models (product grid): §16.19.
41. **ML and WLE scoring for multidimensional models (M4, §8)**: for $D \in \{2, 3\}$, `"ML"`
    maximizes $\log L(\boldsymbol\theta\mid y)$ and `"WLE"` maximizes the Jeffreys/Firth penalized
    $\log L(\boldsymbol\theta\mid y)+\tfrac12\log\det\mathbf I(\boldsymbol\theta)$ (`mirt:::WLE.mirt`;
    Warm, 1989; Wang, 2015), where $\mathbf I$ is the expected test information matrix of the
    observed items from autodiff, over the box $|\theta_m|<20$ (mirt's `max_theta`) by the bounded
    `_newton_nd`, starting from EAP. SEs are square roots of the diagonal of the inverse negative
    Hessian of the objective. ML: patterns whose observed responses are all in the lowest (highest)
    raw category return $-\infty$ ($+\infty$) on every dimension (as mirt does before optimizing);
    a dimension that reaches the bound or has SE $>30$ returns $\pm\infty$ with SE NaN, and the
    remaining dimensions take SEs from the inverse of their own block of the negative Hessian.
    mirt instead sometimes stops such a dimension at a finite value near $\pm20$ with SE 0 (its
    `diag(hessian) > 0` sub-selection); flexpyirt reports the signed infinity. WLE is finite on
    extreme patterns. All-missing rows return NaN.
42. **Plausible values for multidimensional models (M4, §8)**: for $D \in \{2, 3\}$, each unique
    pattern's posterior
    $\log L(\boldsymbol\theta\mid y)-\tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu)$
    (continuous MVN prior) is evaluated on its own equispaced lattice of 61 (2D) or 25 (3D) points
    per dimension spanning EAP $\pm 6$ posterior SD (clipped to the fitted grid's range). Each draw
    picks a cell from the discrete posterior (`jax.random.categorical`) and adds uniform jitter
    within the cell, so draws are continuous; the jitter inflates the variance by $\Delta_m^2/12$
    (about 0.2% in 2D and 0.6% in 3D of the posterior variance). Per-respondent keys come from
    `jax.random.split(key, N)`, so draws are independent across respondents sharing a pattern and
    deterministic given `key`. Patterns and respondents are processed with `lax.map` in batches to
    bound memory. Columns are `pv<m>_<factor>`. Validated statistically: with 4,000 draws the
    sample mean, SD and correlations match the grid posterior within Monte Carlo error (mirt's
    default plausible values use a normal approximation, so they are not a reference).
43. **Multiple-group multidimensional models (M4, §4.5, §4.9)**: `with_factors()` and
    `with_groups()` combine in either order (Normal latents only; `EmpiricalHistogram` stays
    unidimensional). Each group's `Normal` applies to every factor: its `mean` / `var` and
    `free_mean` / `free_var` set `mean_1..D` / `var_1..D` of that group, and `cor_ij` follow
    `with_factors(correlated=...)` in every group, as in mirt's `multipleGroup(...,
    invariance = c(anchors, "free_means", "free_var"))` with `COV = F1*F2`. Identification runs
    the rotational checks of §15.31 within each group, and scale/location linking per factor across
    groups: groups are connected by shared free `a{m}` (scale) or shared free intercepts of items
    loading on factor $m$ (location), and every free `var_m` / `mean_m` must be connected to a fixed
    nonzero `a{m}`, a fixed intercept, or a group with that parameter fixed (`IdentificationError`
    naming the group and factor otherwise). The JSON schema already stores groups and factors
    (§15.33); a model with factors may carry `Normal` groups.
44. **Gauss–Hermite quadrature (`Quadrature.gauss_hermite(n)`, §4.9)**: fixed nodes
    $\theta_q=\sqrt2x_q$ from the Gauss–Hermite rule $(x_q, w_q)$ for $e^{-x^2}$ (Golub & Welsch,
    1969; `numpy.polynomial.hermite.hermgauss`), and a fixed per-node log base weight
    $b_q=\log(w_q/\sqrt\pi)+\tfrac12\theta_q^2$ (summed over dimensions for product rules). The
    latent weights are $\operatorname{softmax}_q\big(b_q+\log\phi(\boldsymbol\theta_q;\boldsymbol\mu,
    \boldsymbol\Sigma)\big)$, i.e. the Gauss–Hermite rule for $N(0,1)$ reweighted by
    $\phi(\cdot;\boldsymbol\mu,\boldsymbol\Sigma)/\phi(\cdot;0,I)$, so nodes never move and free means,
    variances and correlations keep the exact EM structure (E-step, exact latent M-step, Oakes,
    SQUAREM) unchanged. `CompiledProblem.node_log_base` carries $b_q$ (0 for equispaced grids) and
    is added wherever latent log weights are formed (`_normal_logw`). Accuracy degrades for latent
    variances well above 1, where the reweighting factor grows in the tails. `EmpiricalHistogram`
    requires an equispaced grid (`SpecificationError`). PV for $D=1$ samples on its own fine
    equispaced grid (base weights 0). JSON stores `{"type": "gauss_hermite", "n_points": n}`.
    Validated against `scipy.integrate.quad` (marginal log-likelihood with $N(0.3, 1.4)$, 61 nodes:
    relative error $1.2\times10^{-10}$) and against fits on very fine equispaced grids (Rasch with
    free variance, 41 nodes vs 401 points on $[-12,12]$: estimates within $2\times10^{-10}$; 2D 2PL,
    $31^2$ vs $121^2$: within $5\times10^{-7}$); mirt does not use Gauss–Hermite in EM.
45. **S-X2 for multiple-group models (M5, §15.40)**: `item_fit(data, group=...)` computes S-X2
    within each group, as mirt's `itemfit()` does for a `multipleGroup` fit (`MGC2SC`): group
    $g$'s respondents give the observed table and $N_s$, the expected proportions use group $g$'s
    item parameters and latent weights $w_{gq}$, and
    $df_{gj}$ = kept cells − rows − free parameters of item $j$ in group $g$ − free latent
    parameters of group $g$ (an anchor shared across groups counts in every group; the reference
    group with mean 0 and variance 1 fixed subtracts none). RMSEA uses the group's $N_g$. Every
    category must be observed within each group. The result has a leading `group` column. Validated
    against mirt 1.48 (V15): V4's two-group 2PL at mirt's estimates, and Science (GRM) split into
    two groups with all items invariant, at mirt's estimates (mirt stops ~2e-4 short in the item
    parameters there): df identical, S-X2 within 1e-6 relative.

46. **Naming refactor (schema version 4)**: public names follow Python conventions rather than
    mirt/R argument names. Item models take `n_categories` (was `n_cat`); `Quadrature.equispaced`
    and `Quadrature.default` take `bounds=(lower, upper)` (was `lim`), and the default point
    counts are `quadrature.DEFAULT_N_POINTS`; all priors end in `Prior` (`NormalPrior`,
    `LogNormalPrior(mean_log, sd_log)`, `BetaPrior(a, b)`, were `LogNormal(meanlog, sdlog)` and
    `Beta`); `FitResult.item_fit(min_cell=...)` (was `mincell`, mirt's name); string options are
    snake_case (`parameterization="slope_intercept"`, `link(method="stocking_lord")`); and
    `Model.is_empirical_histogram` replaces `Model.is_eh`. Nothing had been released, so the old
    names are removed without deprecation aliases. JSON `schema_version` is 4: item models store
    `n_categories` and priors are registered as `"NormalPrior"`, `"LogNormalPrior"` (keys
    `mean_log`, `sd_log`) and `"BetaPrior"`. Files with versions 1–3 are still read: before
    validation, `_upgrade_legacy_names` renames `n_cat`, `LogNormal`/`meanlog`/`sdlog` and `Beta`
    to their version-4 names. Mathematical notation stays in internal kernels (`G`, `Q`, `K_f`, …)
    and in `LinkResult.A`/`B` (the linking constants of Kolen & Brennan, 2014), and the port of
    mirt's cell collapsing (`_itemfit.collapse_cells`) keeps mirt's local names so it can be
    compared line by line.
47. **Optional unconstrained M-step scale (`BAEM(m_step_scale="unconstrained")`, #15, V45)**:
    $\eta$, the ParameterTable, priors, standard errors and SQUAREM all stay on the natural scale
    (§4.7, §15.2). Only the masked Newton of the M-step (`_newton.newton_ascent`, used by the
    item-wise, coupled, latent and two-tier latent M-steps) can run on an unconstrained scale.
    `BAEM.m_step_scale: Literal["natural", "unconstrained"]` defaults to `"natural"`, the bounded
    active-set Newton of §15.2, so existing results do not change.
    - **Which coordinates.** With `"unconstrained"`, each free coordinate of the M-step box
      `(lower, upper)` (the table bounds intersected with the prior support, §15.16) is mapped by
      $$z=\begin{cases}\operatorname{logit}\frac{x-l}{u-l}, & l,u\text{ finite},\\
      \log(x-l), & \text{only } l \text{ finite},\\ \log(u-x), & \text{only } u \text{ finite},\\
      x, & \text{unbounded},\end{cases}$$
      with inverses $x=l+(u-l)\,\sigma(z)$, $x=l+e^{z}$, $x=u-e^{z}$. This gives logit($g$),
      logit($u$), log($\sigma^2$) for latent variances (also the mixture `rvar_c`), and
      $2\operatorname{atanh}(\rho)$ for MVN correlations in $(-1,1)$. The rule follows the bounds, not
      the parameter name: slopes keep $(-\infty,\infty)$ and are **not** transformed (an
      $\exp(a)$ scale would add the constraint $a>0$, while negative slopes are legitimate), but
      a slope whose box is $(0,\infty)$ because of a `LogNormalPrior` or user-edited bounds is
      optimized on $\log a$, which adds no constraint the box did not already impose.
    - **Objective and step.** The Newton step maximizes $f(T^{-1}(z))$, where $f$ is the same
      natural-scale objective (expected complete-data log-likelihood plus the natural-scale log
      prior). Gradient and Hessian in $z$ come from autodiff through $T^{-1}$ (the chain rule; no
      Jacobian term is added, because the objective is not a density in $z$). The step is the
      same as §15.2 without the box: Newton direction on $z$, gradient fallback if it is not an
      ascent direction, and halving $2^{-k}$ for $k<30$. A candidate is accepted iff its
      natural-scale value $x=T^{-1}(z)$ lies strictly inside `(lower, upper)` on the free
      coordinates (a saturated $\sigma(z)$ that rounds onto a bound is rejected), the objective
      is finite, and $f(x_{\text{cand}})\ge f(x)$. The iterate is carried on the natural scale
      and the comparison uses $f(x)$ itself, so a rejected step returns $x$ bit for bit and EM
      stays monotone exactly as in §15.2.
    - **Interior and boundary solutions.** $T$ is a monotone bijection of the open box, so
      interior fixed points of EM are the same on both scales (only the path differs, because
      the M-step is a partial Newton maximization). A boundary optimum cannot be reached:
      $z\to\pm\infty$, the natural-scale change per iteration shrinks geometrically, and EM
      stops by the $\max|\Delta\eta|<$ `tol` rule with the parameter close to but not at the
      bound, often after many more iterations than the active-set Newton. `HeywoodWarning`
      (§15.6) still uses the natural-scale distance $<10^{-4}$, so it fires only once the
      parameter has drifted that close; a slow drift that is stopped earlier by `tol` can end
      without the warning. That is why `"natural"` stays the default.
    - **SQUAREM** (§15.11–§15.12) is unchanged: it extrapolates the natural-scale $\eta$ and
      checks the same natural-scale bounds. `m_step_scale` only changes the EM map $F$ it calls.
    - **Not changed**: the ParameterTable (`value`, `lower`, `upper`), standard errors (Oakes,
      Hessian, cross-product and sandwich are all on natural $\eta$, so Wald intervals of $g$ are
      not confined to $(0,1)$ by this option), priors, and the stopping rule. `BAEM` is saved in
      the JSON (schema version 8 adds the `m_step_scale` key; versions $\le 7$ read as
      `"natural"`).
48. **Logit-normal prior (`LogitNormalPrior(mean_logit, sd_logit)`, #15, V45)**: a prior on a
    parameter in $(0,1)$ (usually $g$ or $u$) under which $\operatorname{logit}X\sim
    N(\mu,s^2)$ (Aitchison & Shen, 1980). On the natural scale, as every
    prior in §4.7, its density includes the Jacobian of the logit:
    $$\log\pi(x)=-\log x-\log(1-x)-\log s-\tfrac12\log(2\pi)-\frac{(\operatorname{logit}x-\mu)^2}{2s^2},\qquad 0<x<1,$$
    and $-\infty$ outside $(0,1)$. The support is $(0,1)$, so it is intersected with the box like
    `BetaPrior` (§15.16); the Core code is `PRIOR_LOGITNORMAL = 4` in `eval_log_prior`; a starting
    value outside the box moves to the prior median $\sigma(\mu)$.
    - **Relation to flexMIRT.** flexMIRT (Houts & Cai) estimates the 3PL lower asymptote on the
      logit metric, $\lambda=\log[g/(1-g)]$, and its `Prior ..., Guessing : Normal(μ, s);` (e.g.
      `Normal(-1.09, 0.5)` for $g\approx 1/4$) is a normal prior on $\lambda$. Its posterior mode
      maximizes $\ell+\log\phi(\lambda;\mu,s)$ over $\lambda$, which is the mode over $g$ of
      $\ell+\log\phi(\operatorname{logit}g;\mu,s)$ — the same normal kernel **without** the
      $-\log g-\log(1-g)$ term. `LogitNormalPrior` is the proper density on $g$, so its natural-scale
      MAP (§4.7) differs from flexMIRT's estimate by that term. The term grows towards both bounds,
      so the `LogitNormalPrior` mode lies further from $1/2$ than flexMIRT's; the prior itself (as a
      distribution of $g$) is the same.
      R mirt's `norm` prior on `g` acts on mirt's internal logit($g$) and behaves like flexMIRT's.
      `mode_scale="logit"` (below) reproduces that mode.
    - **`mode_scale: Literal["natural", "logit"] = "natural"`** (keyword-only). `"natural"` is the
      density above. `"logit"` drops the Jacobian and evaluates the normal log density of
      $\lambda=\operatorname{logit}x$ at the natural-scale value,
      $$\log\tilde\pi(x)=-\log s-\tfrac12\log(2\pi)-\frac{(\operatorname{logit}x-\mu)^2}{2s^2},\qquad 0<x<1,$$
      ($-\infty$ outside). Because $\operatorname{logit}$ is a bijection of $(0,1)$ and no Jacobian
      is added, $\arg\max_g[\ell+\log\tilde\pi]=\sigma(\arg\max_\lambda[\ell+\log\phi(\lambda;\mu,s)])$:
      the MAP is flexMIRT's `Guessing : Normal(μ, s)` and mirt's `PRIOR = (..., g, norm, μ, s)`
      estimate, and `log_posterior` equals their logit-metric log posterior (mirt's `logPrior` is
      $\sum\log\phi(\operatorname{logit}g)$). $\log\tilde\pi$ is not a density of $x$ (it is the
      density of $\lambda$); `log_pdf` returns it as the objective term, as for every prior (§4.7).
      - Core: a separate code `PRIOR_LOGITNORMAL_LOGIT = 5` in `eval_log_prior` with the same two
        parameters, so `PriorBlock` and the jit kernels are unchanged (branch-free double-`where`
        like the other codes). Support, box intersection and start value ($\sigma(\mu)$, which is
        also the logit-scale prior mode) are as for `"natural"`.
      - Standard errors use the same objective on natural $\eta$, so they are consistent with it:
        at a stationary point the natural-scale Hessian is $J^\top H_\lambda J$ with
        $J=\operatorname{diag}(g(1-g))$, hence $\mathrm{SE}(g)=g(1-g)\,\mathrm{SE}(\lambda)$, the
        delta-method SE of the logit-scale estimate. `m_step_scale` (§15.47) is independent: it
        changes the Newton path, not the objective.
      - JSON: the `"LogitNormalPrior"` object gains a `mode_scale` key within schema version 8,
        written always and optional on read (absent → `"natural"`).
    - The hyperparameters are validated like `LogNormalPrior` (`mean_logit` finite, `sd_logit`
      positive and finite). JSON registers it as `"LogitNormalPrior"` with keys `mean_logit` and
      `sd_logit` (schema version 8).

## 16. Implementation decisions (M5)
1. **Equality constraints across items (`Model.equal_across_items`, shared `free_id`s)**:
   `Model.equal_across_items(*, items, params)` makes each parameter in `params` of the selected
   items (at least two) equal within every group, by giving their free rows one `free_id` (as
   mirt's `CONSTRAIN = (items, param)`; in a multiple-group model the constraint applies within
   each group, as `CONSTRAIN` does in `multipleGroup`). It combines with `equal_across_groups`:
   all equalities of the model are merged by union–find over rows, so overlapping constraints
   share one id. Fixed rows are not affected. Rows sharing a `free_id` (from the model or from an
   edited table) may now belong to different items and different item models (e.g. a common
   slope for 2PL and GRM items), but within one item a `free_id` still maps to one parameter, and
   the rows must agree on bounds and prior (as before). The default starting value of a shared
   parameter is the mean of its rows' default starting values.
   **M-step**: the expected complete-data log posterior is
   $Q(\eta)=\sum_j Q_j(\xi_j(\eta))+\log\pi(\eta)$, so the item-wise decomposition (§4.3) holds
   between connected components of the item graph whose edges are shared `free_id`s, not between
   items. The Compile layer finds the components by union–find over items; a component with two
   or more items becomes a `CoupledBlock` (per member family: the static family index, the
   dynamic item positions and a `(G, n, P_f)` local index into the component's vector $u_c$ of
   free parameters; plus $u_c$'s `free_id`s and bounds). Its M-step maximizes
   $\sum_{j\in c}\sum_g Q_{jg}(\xi_j^{(g)}(u_c))+\sum_{s}\log\pi(u_{cs})$ over $u_c$ with the
   same masked, bounded, monotone Newton (§13.1, §15.2; gradient and Hessian by autodiff, each
   member family's items `vmap`ed), so each shared prior is counted once and EM stays monotone.
   Items of a component are masked out of the item-wise `vmap` (their slots are held) and the
   component step runs after it, reading the component's parameters from the previous iterate
   (they are disjoint from every other block's). Components are a static Python loop, so a model
   with many small components traces one Newton block per component. A common slope over $J$
   2PL items gives one block of $1+J$ parameters, whose dense Hessian is cheap at the sizes of
   interest.
   **Standard errors and SQUAREM**: Oakes' first term scatter-adds each item's Hessian block by
   `free_id`, and the cross term accumulates scores by `free_id`, so parameters shared across
   items are summed exactly with no change (likewise `information_hessian`, which differentiates
   the marginal log posterior directly, and the SQUAREM bounds). Delta-method SEs, `n_params`
   (counted once per `free_id`) and JSON (model field `equal_items`, schema version 3) follow
   from the table.
   **Identification**: the necessary conditions of §4.5 and §14.5 are unchanged (for example a
   common free slope with a free latent variance is still rejected, because no slope is fixed).
2. **DIF by multiple-group likelihood-ratio tests (`FitResult.dif`)**:
   `FitResult.dif(data, *, group, items, params=None, scheme="add", p_adjust="holm",
   item_names=None)` tests each
   item in `items` for differential item functioning across the groups of a fitted
   multiple-group model (Thissen, Steinberg & Wainer, 1993), as mirt's `DIF()`. For
   `scheme="add"` the fitted model is the baseline and each item's `params` (default: all of
   its parameters) are made equal across groups; for `scheme="drop"` they are freed (one
   `free_id` per group). Each modified model is refitted with the same `BAEM` settings and
   quadrature, starting from the baseline estimates, and
   $X^2=2(\ell_{\text{free}}-\ell_{\text{constrained}})$ with
   $df=|k_{\text{base}}-k_{\text{new}}|$ gives $p=P(\chi^2_{df}>X^2)$. The result has one row per
   item with `item`, `converged`, the differences (item free minus item constrained, mirt's sign,
   so negative values favour DIF) of AIC,
   SABIC ($-2\ell+k\log((N+2)/24)$), HQ ($-2\ell+2k\log\log N$) and BIC, `x2`, `df`, `p` and
   `adj_p` (adjusted over the tested items: Holm, 1979, by default as mirt's `p.adjust = "holm"`;
   `p_adjust="BH"` for Benjamini & Hochberg, 1995, or `"none"`). Changes are made in the parameter table, so identification is checked as
   for any fit (freeing the only anchors raises `IdentificationError`), and an item whose
   parameters are already equal (`"add"`) or already free (`"drop"`) raises
   `SpecificationError`. MAP fits are rejected (the statistic needs the marginal likelihood at
   the MLE). DIF is post-fit orchestration in the Result layer; only the refits run in `jit`.
3. **Limited-information goodness of fit (`FitResult.m2`, $M_2$ / $M_2^*$)**:
   `FitResult.m2(data, *, item_names=None, ci=0.90)` computes the limited-information
   goodness-of-fit statistic of Maydeu-Olivares & Joe (2005, 2006) on the first- and second-order
   item-score moments — the $J$ item means $E[y_j]$ and $\binom{J}{2}$ cross-products
   $E[y_iy_j]$ ($i < j$, in column-major pair order, matching mirt's default `M2(type = "M2*")`;
   for dichotomous items this is exactly $M_2$) — for single-group, complete-data
   MLE fits with a `Normal` latent distribution (bifactor / two-tier and `EmpiricalHistogram`
   fits raise `SpecificationError`; confirmatory multidimensional models on a product grid are
   covered by §16.19, and multiple groups, missing data and MAP fits by §16.23).
   With sample margins $\hat p_2$, model-implied margins $\pi_2(\hat\eta)$ integrated on the fitted
   quadrature grid with the normalized normal weights, asymptotic covariance $\Xi_2$ of
   $\sqrt{N}(\hat p_2-\pi_2)$ under local independence (using conditional first and second moments
   $m_{1,jq}=\sum_k kP_{jk}(\theta_q)$ and $m_{2,jq}=\sum_k k^2P_{jk}(\theta_q)$), and Jacobian
   $\Delta_2=\partial\pi_2/\partial\eta^\top$ by JAX forward-mode autodiff over the free vector
   $\eta$ (shared parameters appear once, matching column summation for equality constraints):
   $$M_2 = N\,(\hat p_2-\pi_2)^\top C_2\,(\hat p_2-\pi_2),\qquad
     C_2 = \Delta_c(\Delta_c^\top\Xi_2\Delta_c)^{-1}\Delta_c^\top,\qquad
     df = s - q,$$
   where $s = J + \binom{J}{2}$, $q = \text{n\_params}$ (all free parameters, including any free
   latent mean/variance), and $\Delta_c$ is an orthonormal basis for the orthogonal complement of
   $\operatorname{col}(\Delta_2)$ obtained from the SVD of $\Delta_2$. The asymptotic $\chi^2_{s-q}$
   distribution requires $df = s - q > 0$ and full column rank $\operatorname{rank}(\Delta_2) = q$
   (local identification from first- and second-order moments; Maydeu-Olivares & Joe, 2005,
   Theorem 1; 2006, Theorem 2). Whenever $df \le 0$ or $\operatorname{rank}(\Delta_2) < \operatorname{ncol}(\Delta_2)$
   (for example when a polytomous item has more free parameters than items, $P_j > J$, or a fixed
   zero slope, even if $df > 0$), flexpyirt emits a `FlexpyirtWarning` noting that the model is not
   locally identified with residual degrees of freedom from first- and second-order moments and
   returns `NaN` for `m2`, `p`, RMSEA, its CI, `tli` and `cfi` (keeping `df` and `srmsr` finite);
   it never pairs an $(s - r)$-dimensional quadratic form with $df = s - q$.
   **Free latent parameters in $\Delta_2$**: mirt 1.46.1 excludes free latent mean/variance columns
   from $\Delta_2$ while still subtracting them in $df$. Following Maydeu-Olivares & Joe (2005,
   2006), flexpyirt includes every free parameter (item and latent) in $\Delta_2$ by default; the
   internal helper `_m2.m2_statistic` accepts a private `_latent_in_delta: bool = True` flag so
   validation tests can verify both the theoretical definition and agreement with mirt when set to
   `False` (for models with no free latent parameters both are identical).
   **Auxiliary fit indices**:
   - $p = P(\chi^2_{df} > M_2)$ (survival function).
   - $\mathrm{RMSEA} = \sqrt{\max(0, M_2/df - 1)/(N - 1)}$ (`NaN` for $df \le 0$ or $N \le 1$), with
     $(100\,\mathrm{ci})\%$ confidence interval $\sqrt{\lambda/(N\,df)}$ by inverting the
     noncentral $\chi^2$ CDF (MacCallum, Browne & Sugawara, 1996; matching mirt's `RMSEA.CI`,
     returning 0 when $P(\chi^2_{df}\le M_2)$ is already at or below the target probability, and
     expanding the upper search bracket from $\max(N, 5M_2)$ if needed so the upper bound cannot
     spuriously collapse to 0). Column names follow `ci` (e.g. `rmsea_5`, `rmsea_95` for `ci=0.90`).
   - $\mathrm{SRMSR} = \sqrt{\binom{J}{2}^{-1}\sum_{i<j}(r_{ij} - \hat\rho_{ij})^2}$ from the
     observed and model-implied item-score correlation matrices (diagonal variance uses
     $E[y_j^2] - E[y_j]^2$).
   - **Null model, TLI and CFI**: for dichotomous models ($K_j = 2$ for every item), the
     independence (null) model has every slope fixed at 0 and $J$ free item intercepts whose MLE
     reproduces the observed proportions; its orthogonal complement simplifies analytically to
     $M_{2,\text{null}} = N\sum_{i<j} r_{ij}^2$ with $df_{\text{null}} = \binom{J}{2}$ (exact,
     closed-form, matching mirt's `M2(null.mod)`). When $M_{2,\text{null}} > M_2$ and $df > 0$,
     $\mathrm{TLI} = (M_{2,\text{null}}/df_{\text{null}} - M_2/df)/(M_{2,\text{null}}/df_{\text{null}} - 1)$
     (uncapped) and $\mathrm{CFI} = \operatorname{clip}(1 - (M_2 - df)/(M_{2,\text{null}} - df_{\text{null}}), 0, 1)$;
     when $M_{2,\text{null}} \le M_2$, a `FlexpyirtWarning` is emitted and both are `NaN`.
     For models containing any polytomous item ($K_j > 2$), `tli` and `cfi` are returned as `NaN`:
     under $M_2^*$'s item-score margins the $K_j - 1$ intercept columns of each item in the null
     model's $\Delta_{\text{null}}$ are strictly collinear ($\operatorname{rank}(\Delta_{\text{null}}) = J < \sum_j(K_j - 1)$),
     so the null $M_2^*$ with $df_{\text{null}} = s - \sum_j(K_j - 1)$ is ill-posed (mirt 1.46.1
     slices `qr.Q(qr(delta), complete = TRUE)` after `ncol(delta)` without rank checking, discarding
     arbitrary directions that depend on item order and floating-point roundoff).
   Moment integration and $\Delta_2$ use JAX; $\Xi_2$, the SVD complement, and all summary indices
   run in NumPy/SciPy outside `jit`.
4. **Davidian-curve latent distributions (`DavidianCurve`, §4.4, V19)**:
   `DavidianCurve(degree, *, mean=0.0, var=1.0, free_mean=False, free_var=False)` (plus
   `DavidianCurve.free(degree, *, mean=0.0, var=1.0)`) specifies a single-group unidimensional
   Davidian-curve density of degree $k\in\{1,\dots,10\}$ (Zhang & Davidian, 2001; Woods & Lin,
   2009). Multiple-group (`Model.with_groups`) and multidimensional (`Model.with_factors`) models
   with `DavidianCurve` raise `SpecificationError` (mirt 1.46.1 forbids free mean/variance
   hyper-parameters and standard errors for Davidian curves).
   **Density and analytical standardization**:
   Let $\psi_j(z)=He_j(z)/\sqrt{j!}$ ($j=0,\dots,k$) be the normalized probabilists' Hermite
   polynomials (orthonormal under $\varphi(z)=(2\pi)^{-1/2}e^{-z^2/2}$) and let
   $h_k(z;\boldsymbol\phi)=P_k(z;\boldsymbol\phi)^2\varphi(z)$ with
   $P_k(z;\boldsymbol\phi)=\sum_{j=0}^k c_j(\boldsymbol\phi)\psi_j(z)$, where the unit vector
   $\mathbf c(\boldsymbol\phi)\in S^k$ uses the cosine-first spherical chart
   $c_0=\prod_{m=1}^k\cos\phi_m$, $c_j=\sin\phi_j\prod_{m=j+1}^k\cos\phi_m$ ($j=1,\dots,k$).
   `dcurver`'s sine-first chart ($c_0=\sin\phi_1$) places $N(0,1)$ at the polar singularity
   $\phi_1=\pi/2$, where $\partial\mathbf c/\partial\phi_{2:k}=\mathbf 0$; this chart places
   $N(0,1)$ at $\boldsymbol\phi=\mathbf 0$ with identity Jacobian
   $\partial c_j/\partial\phi_m|_{\mathbf 0}=\delta_{jm}$ and nests degree $k-1$ at $\phi_k=0$.
   Both charts cover all of $S^k$ and therefore span the exact same family of densities
   (identified up to $\mathbf c\mapsto-\mathbf c$, which leaves $P_k^2$ unchanged). The chart alone
   does **not** make $N(0,1)$ a non-stationary point: the first-order perturbations
   $\psi_1(z)\varphi(z)$ and $\psi_2(z)\varphi(z)$ are pure location and scale changes, which
   analytical standardization removes (and which free items absorb in mirt's unstandardized
   parameterization), so the score in $\phi_1,\phi_2$ is identically zero at
   $\boldsymbol\phi=\mathbf 0$ for every data set (for $k=1$, $\phi=0$ is always stationary).
   mirt 1.46.1 stalls at $N(0,1)$ on V19b ($k=2$) for the same reason combined with the polar
   singularity in $\phi_2$; flexpyirt escapes it through the perturbed default starts below.
   **Singular information at $\boldsymbol\phi=\mathbf 0$**: consequently the Fisher information
   is singular in the $(\phi_1,\phi_2)$ directions at $\boldsymbol\phi=\mathbf 0$ (in $\phi_1$
   for $k=1$). When the true latent density is close to normal, $\hat{\boldsymbol\phi}$ is
   poorly determined, the likelihood can have several local maxima, the angle SEs (and the
   Wald tests built on them) are unreliable, and a likelihood-ratio test of `DavidianCurve`
   against `Normal` does not have its usual $\chi^2$ reference distribution (the null lies
   where the model is not first-order identified). Users should compare degrees with
   AIC/BIC (as in Woods & Lin, 2009) rather than with LR tests against `Normal`.
   From the Hermite three-term recurrence, $Z\sim h_k(\cdot;\boldsymbol\phi)$ has closed-form
   continuous moments
   $$\mu_Z(\boldsymbol\phi)=2\sum_{j=0}^{k-1}\sqrt{j+1}\,c_jc_{j+1},\qquad
     \sigma_Z^2(\boldsymbol\phi)=\sum_{j=0}^k(2j+1)c_j^2+2\sum_{j=0}^{k-2}\sqrt{(j+1)(j+2)}\,c_jc_{j+2}-\mu_Z(\boldsymbol\phi)^2>0.$$
   Standardizing analytically to $\theta=\mu+\sigma(Z-\mu_Z)/\sigma_Z$ gives the density
   $f_k(\theta;\mu,\sigma^2,\boldsymbol\phi)=(\sigma_Z/\sigma)h_k(z(\theta);\boldsymbol\phi)$
   with $z(\theta)=\mu_Z(\boldsymbol\phi)+\sigma_Z(\boldsymbol\phi)(\theta-\mu)/\sigma$, which
   has exact continuous mean $\mu$ and variance $\sigma^2$ for every $\boldsymbol\phi\in\mathbb R^k$
   (and reduces identically to `dcurver::ddc` when $\mu=\mu_Z$, $\sigma^2=\sigma_Z^2$).
   **Grid weights, M-step, and comparison with mirt**:
   Quadrature log-weights are normalized on the grid via
   $\log w_q=\operatorname{log\_softmax}_q(\log(P_k(z_q)^2)-\tfrac12 z_q^2)$ (with double-`where`
   guards for exact polynomial roots $P_k(z_q)=0$). Angles $\phi_1,\dots,\phi_k$ are unbounded in
   `(-inf, inf)` during optimization so Newton and SQUAREM are never blocked when a coefficient
   $c_j$ changes sign. The `ParameterTable` stores `mean`, `var`, and `phi_1..phi_k` under
   `item="__latent__"`, `model="Davidian"`; identification follows §4.5 (`mean`/`var` follow the
   same rules as `Normal`, and priors on latent parameters raise `SpecificationError`). Because
   analytical standardization cancels the first-order derivative in $\phi_1,\phi_2$ at
   $\boldsymbol\phi=\mathbf 0$, default starts apply a small skewness-aligned perturbation
   (`phi_1 = 0.1 * sign` for $k=1$; `phi_1 = -0.1 * sign, phi_2 = 0.1, phi_{3..k} = 0` for
   $k\ge 2$, where `sign` comes from empirical total-score skewness). These starts give the
   standardized start density a third standardized moment with the same sign as `sign`: to
   leading order it is $\approx 4\phi_1^3$ for $k=1$ (`+0.0039` at $\phi_1=0.1$) and
   $\approx -8.3\,\phi_1\phi_2$ for $k\ge 2$ (`+0.083` at $(\phi_1,\phi_2)=(-0.1,0.1)$), while
   $\phi_2=0.1$ moves the start off the stationary point at $\boldsymbol\phi=\mathbf 0$ (checked
   in `tests/test_m5_davidian.py`). The latent M-step maximizes
   $Q^{\text{lat}}=\sum_q \bar n_q\log w_q$ with the existing masked Newton (`_newton.py`),
   keeping EM monotone (non-decreasing), SQUAREM working without special cases, and Oakes == Hessian SEs
   exact to $\sim 10^{-12}$ (by contrast, mirt 1.46.1 applies Woods' linear-interpolation
   `standardizeQuadrature` to expected counts before calling `nlminb` on unnormalized `ddc`,
   which breaks EM monotonicity, disables `SE = TRUE`, and forbids free mean/variance; on V19a it
   stops at a point that is non-stationary for mirt's own discretized likelihood (score
   $\approx 56.7$ in `PHI_3`), and on V19b it stalls at the $N(0,1)$ stationary point; both
   with strictly lower log-likelihood; see VALIDATION.md V19).
   **Scoring, reporting, and JSON**:
   `FitResult.latent` displays `mean`, `var`, `phi_1..phi_k` and their SEs; `latent_weights`
   stays `None` (parametric). `score()` uses the fitted Davidian density for `"EAP"` (on the
   quadrature grid) and `"PV"` (on the 501-node fine grid), keeps `"ML"` and `"WLE"`
   likelihood-only, and rejects `"MAP"` with `SpecificationError` because $P_k(z)^2\varphi(z)$
   can be multimodal and vanish at real roots of $P_k$ (matching mirt's `use_dentype_estimate`).
   `item_fit()` rejects `DavidianCurve` until validated. `"DavidianCurve"` is registered in
   `_json.py` without changing the JSON payload shape (`schema_version` remains 3).
5. **Differentiable IRT scale linking (`link`, `LinkResult`, V20)**:
   Characteristic-curve linking of two separately calibrated unidimensional scales via common items
   (§1), solving for linear transformation constants $(A, B)$ with $A > 0$ such that
   $\theta^* = A\theta + B$ maps the `source` (new) scale onto the `target` (base) scale
   (Kolen & Brennan, 2014). Under $\theta = (\theta^* - B)/A$, slope–intercept item parameters
   transform with $a^* = a / A$ and:
   - `DichotomousKernel` (2PL/Rasch) and `FourPLKernel` (3PL/4PL): $d^* = d - aB/A$ (asymptotes
     $g, u$ unchanged);
   - `GradedKernel` (GRM): $d_k^* = d_k - aB/A$ for $k = 1, \dots, K-1$ (preserving $d_1 > \dots > d_{K-1}$);
   - `GPCMKernel` (GPCM/PCM): $d_k^* = d_k - k\,aB/A$ for $k = 1, \dots, K-1$ (with $d_0 = 0$),
     preserving $k a^* \theta^* + d_k^* = k a\theta + d_k$ and step difficulties $b_k^* = A b_k + B$.
   - `NominalKernel` (nominal response model; V40): the scoring coefficients are unchanged,
     $\mathrm{ak}_k^* = \mathrm{ak}_k$ ($k = 1, \dots, K-2$; $\mathrm{ak}_0 = 0$ and
     $\mathrm{ak}_{K-1} = K-1$ stay fixed), and $d_k^* = d_k - \mathrm{ak}_k\,aB/A$ for
     $k = 1, \dots, K-1$ (with $d_0 = 0$), preserving
     $\mathrm{ak}_k a^* \theta^* + d_k^* = \mathrm{ak}_k a\theta + d_k$. The transformation is the
     GPCM's with $k$ replaced by $\mathrm{ak}_k$; it stays inside mirt's identification
     ($\mathrm{ak}_0 = 0$, $\mathrm{ak}_{K-1} = K-1$, $d_0 = 0$) because only $a$ absorbs $1/A$.
   For the 4PL the upper asymptote $u$ is invariant like $g$ (both are probabilities, not
   locations on $\theta$), so only $a$ and $d$ change (V40).
   Because $P_j(y=k \mid \theta^*; \xi_j^*) = P_j(y=k \mid (\theta^* - B)/A; \xi_j)$ for any
   unidimensional `ItemKernel`, the criteria are evaluated directly from `kernel.log_prob` with no
   hand-written derivatives:
   - **Haebara** (Haebara, 1980; polytomous form Kim & Lee, 2006):
     $H_{\mathrm{fwd}}(A,B) = \sum_q w_q^{(t)} \sum_{j\in\mathcal{C}} \sum_{k=0}^{K_j-1} [P_{jk}^{\mathrm{tgt}}(\theta_q^{(t)}) - P_{jk}^{\mathrm{src}}((\theta_q^{(t)}-B)/A)]^2$.
     The sum runs over all categories, including both categories of a dichotomous item (the
     Kim & Lee, 2006 convention), so a dichotomous item contributes $2[P^{\mathrm{tgt}} - P^{\mathrm{src}}]^2$,
     twice Haebara's (1980) original term: the solution is unchanged when all common items are
     dichotomous, reported criterion values are doubled, and in mixed-format anchors dichotomous
     items carry the Kim & Lee weight relative to polytomous items.
   - **Stocking–Lord** (Stocking & Lord, 1983; polytomous form Kim & Lee, 2006):
     $\mathrm{SL}_{\mathrm{fwd}}(A,B) = \sum_q w_q^{(t)} [\mathrm{TCC}^{\mathrm{tgt}}(\theta_q^{(t)}) - \mathrm{TCC}^{\mathrm{src}}((\theta_q^{(t)}-B)/A)]^2$,
     where $\mathrm{TCC}(\theta) = \sum_{j\in\mathcal{C}}\sum_{k=0}^{K_j-1} k\,P_{jk}(\theta)$.
     A nominal item is scored by its category index $k$ like every other item (mirt's
     `expected.item()`), not by $\mathrm{ak}_k$: the criterion is still a valid loss (zero iff the
     curves coincide when the transformation holds exactly), but the scores of a nominal item
     depend on how its categories are coded, so with unordered categories Haebara, which does
     not use scores, is the natural choice.
   - **Symmetric variants** (`symmetric=True`; Kim & Lee, 2006): add the reverse-direction term
     comparing $P_{jk}^{\mathrm{src}}(\theta_q^{(s)})$ against $P_{jk}^{\mathrm{tgt}}(A\theta_q^{(s)} + B)$
     (likewise for $\mathrm{TCC}$). By default, $(\theta^{(t)}, w^{(t)})$ is the target group's
     fitted latent distribution on its quadrature grid and $(\theta^{(s)}, w^{(s)})$ is the source
     group's, making symmetric linking exactly reversible: swapping `source` and `target` yields
     $(1/A, -B/A)$. Explicit `nodes` / `weights` override the grid (weights are normalized to sum
     to 1; uniform if `nodes` is given without `weights`).
   **Optimization**: starting from Mean/Mean ($A_0 = \sum |a_j^{\mathrm{src}}| / \sum |a_j^{\mathrm{tgt}}|$,
   $B_0 = \bar b^{\mathrm{tgt}} - A_0 \bar b^{\mathrm{src}}$), a jitted, shape-static 2-D bounded
   monotone Newton minimizer (`_linking.solve_linking`, `lax.while_loop`) uses `jax.grad` and
   `jax.hessian` on `LinkingProblem` pytrees with $A > 0$ enforced by halfway-to-bound projection,
   Levenberg–Marquardt fallback when the Hessian is not positive definite, the Euclidean step
   norm capped at `_MAX_STEP = 1` (so a poor start cannot jump onto a flat plateau where the curves
   no longer overlap), 40 parallel backtracking step sizes, and a 1-ULP gradient-polish step near
   the optimum. Convergence is $\|\nabla F\|_\infty \le$ `tol` (absolute, default $10^{-10}$, not
   rescaled: $F$ uses weights summing to 1 and grows with the number of common items); a step
   that no longer decreases $F$ before reaching `tol` ends the loop with `converged = False` and a
   `ConvergenceWarning`. Mean/Mean is only a starting value: it sums absolute slopes and pools
   every display threshold of every common item, so polytomous items weigh more than in per-item
   averages. A nominal item has no display thresholds; it enters $\bar b$ with the single location
   $-d_{K-1}/((K-1)a)$ (where its first and last category curves cross), which transforms as
   $Ab + B$.
   **Standard errors (Ogasawara, 2001)**: from the stationarity condition
   $g(c, \eta_s, \eta_t) = \nabla_c F(c, \eta_s, \eta_t) = 0$ at $\hat c = (\hat A, \hat B)$, the
   implicit function theorem gives $J_s = -H_{cc}^{-1} H_{c,\eta_s}$ and
   $J_t = -H_{cc}^{-1} H_{c,\eta_t}$ via `jax.jacfwd(g, 0)` and `jax.jacrev(g, 1/2)` over the
   common items' unique free parameters (respecting fixed parameters and shared `free_id`s), so
   $\mathrm{Cov}(\hat A, \hat B) = J_s \Sigma_s J_s^\top + J_t \Sigma_t J_t^\top + J_s \Sigma_{st} J_t^\top + J_t \Sigma_{st}^\top J_s^\top$.
   Separate `FitResult`s are treated as independent calibrations ($\Sigma_{st} = 0$); when the
   same `FitResult` is passed as `source` and `target` (two groups of one multiple-group fit),
   $\Sigma_{st}$ is the cross block of its joint `vcov`, so equality-constrained anchors shared by
   both groups give $(A, B) = (1, 0)$ with zero covariance. Distinct `FitResult`s fitted to
   overlapping data are not detected. **Quadrature weights are held fixed.** By default they are
   the fitted latent distributions (estimates when a group's mean/variance is free or the
   distribution is EH), but this is exact to first order under a correctly specified model:
   $\partial g/\partial w_q$ is a weighted sum of curve residuals times curve derivatives and the
   residuals vanish at the population solution, so $H_{c,w} = 0$ there; under item parameter drift
   or DIF (non-zero population residuals) the weights' sampling variability is ignored. SEs are
   computed only at a converged solution; non-convergence, unavailable `vcov` on a side with free
   common-item parameters, or a non-PD $H_{cc}$ yields `vcov = None` and `NaN` SEs.
   **Transformed items and scores**: `LinkResult.items` holds every source item on the target
   scale; fixed-slope models (Rasch, PCM) get $a^* = 1/A$ (not 1), and `Model.fix_from` fixes
   that slope. `transform_theta` maps `theta` and every `pv*` column by $A\theta + B$ and `se` by
   $A\,\mathrm{se}$, treating $(A, B)$ as known (linking error is not added).
   **Validation in the Python shell**: multidimensional fits, multiple-group fits without explicit
   `source_group` / `target_group`, mismatched item kernels or category counts on common items, and
   invalid nodes/weights raise `SpecificationError`.
6. **Nominal response model (`Nominal(n_categories)`, `NominalKernel(n_categories, n_factors=1)`, V23)**:
   Bock's (1972) nominal response model in Thissen, Cai & Bock's (2010) overall-slope ×
   scoring-function parameterization, matching R mirt's `itemtype = "nominal"` (Chalmers, 2012):
   $$P(y=k\mid\boldsymbol\theta)=\frac{\exp\big(\mathrm{ak}_k\,\mathbf a^\top\boldsymbol\theta+d_k\big)}{\sum_{c=0}^{K-1}\exp\big(\mathrm{ak}_c\,\mathbf a^\top\boldsymbol\theta+d_c\big)},\qquad k=0,\dots,K-1,\quad \mathrm{ak}_0=0,\ \mathrm{ak}_{K-1}=K-1,\ d_0=0,$$
   for categories `0..K-1` (`n_categories >= 2` validated) and $D\in\{1,2,3\}$ (evaluated with
   `_linear_predictor(theta, params[:D], D)` multiplying the scoring vector $\mathrm{ak}$ and
   `jax.nn.log_softmax`).
   - **Parameters and bounds**: mirt's `coef(m)` / `pars = "values"` reports `a1` (`a1..aD`),
     `ak0..ak{K-1}`, and `d0..d{K-1}` with `ak0 = 0`, `ak{K-1} = K - 1`, and `d0 = 0` fixed
     (`est = FALSE`) and all bounds `(-Inf, Inf)`. As in `GPCMKernel` (§15.8), the three fixed
     identification constants are part of the kernel rather than `ParameterTable` rows (so
     `_check_identification` does not treat them as user-fixed slopes or intercepts):
     `param_names` is `("a", "ak1", ..., "ak{K-2}", "d1", ..., "d{K-1}")` for $D=1$ and
     `("a1", ..., "aD", "ak1", ..., "ak{K-2}", "d1", ..., "d{K-1}")` for $D\in\{2,3\}$, all free
     by default and unbounded `(-inf, inf)`. Fixing $\mathrm{ak}_k=k$ ($k=1,\dots,K-2$) reduces
     `Nominal(K)` to `GPCM(K)` identically (`Nominal(2)` has no free `ak` parameters and equals
     `GPCM(2)` / `TwoPL`).
   - **Starting values, identification, and estimability**: mirt's default starting values
     (`a1 = 0.851`, `ak1 = ... = ak{K-2} = 0.5`, `d1 = ... = d{K-1} = 0`) give interior
     categories $1,\dots,K-2$ identical predictors at iteration 0 and can cause mirt's undamped
     `optimizer = "NR"` to hit an indefinite joint Hessian and trap `a1 = 0` (e.g. on `Science`).
     flexpyirt starts `Nominal(n_categories)` at the GPCM starting point: $a_j^{(0)}$ from the item–rest
     correlation (§4.10; factor-specific item–rest correlations for $D>1$, §15.31), distinct
     scoring coefficients $\mathrm{ak}_{jk}^{(0)}=k$ ($k=1,\dots,K-2$), and GPCM starting
     intercepts $d_{jk}^{(0)}$ (`_gpcm_intercept_starts`, §15.9). This breaks category symmetry at
     iteration 0 while allowing items with non-monotone or negative true $\mathrm{ak}_k$ to
     converge to the unordered MLE. For §4.5 / §15.31 identification checks, `slope_params` is
     `("a",)` (`("a1", ..., "aD")`) and `intercept_params` is `("d1", ..., "d{K-1}")`;
     `ak1..ak{K-2}` carry relative category shape (fixing $\mathrm{ak}_k=k$ yields GPCM and does
     not identify a free latent variance or mean). Every category `0..K-1` must be observed for
     items with free parameters (`_check_estimable`).
   - **Display thresholds, FIPC, JSON, and ML scoring**: mirt's `coef(m, IRTpars = TRUE)` converts
     nominal items to Bock's (1972) zero-sum slopes and intercepts
     $a_k^*=a\,\mathrm{ak}_{k-1}-\overline{a\,\mathrm{ak}}$ (`a1..aK`) and $c_k^*=d_{k-1}-\bar d$
     (`c1..cK`) rather than $\theta$-scale difficulty thresholds $b_k$, and adjacent category
     intersections $-(d_k-d_{k-1})/[a(\mathrm{ak}_k-\mathrm{ak}_{k-1})]$ are singular whenever
     $\mathrm{ak}_k=\mathrm{ak}_{k-1}$. `NominalKernel` therefore sets `irt_names = ()` and
     `irt_dependency_indices = ()` (no `b` / `bk` columns in `FitResult.items`), and `Model.fix_from()`
     supports `Nominal` items with `parameterization="slope_intercept"` while raising
     `SpecificationError` for `parameterization="irt"` on `Nominal` items. `"Nominal"` is added to
     the polytomous item registry in `_json.py` (keeping schema version 3 and older versions 1–2
     readable). In 1-D ML scoring (`_scoring._pattern_flags`), an item's lowest- and
     highest-$\theta$ categories are determined from the effective category slopes
     $s_c=a\,\mathrm{ak}_c$ ($s_{y_j}=\min_c s_c$ and $s_{y_j}=\max_c s_c$ with $a\neq 0$), which
     handles non-monotone categories ($\mathrm{ak}_k<0$ or $\mathrm{ak}_k>K-1$) and reduces to
     $0$ and $K-1$ when $0\le\mathrm{ak}_k\le K-1$. This deliberately differs from mirt 1.46.1,
     whose `fscores(method = "ML")` returns $\mp\infty$ for every all-category-0 /
     all-top-category pattern regardless of $\mathrm{ak}$: with $\mathrm{ak}_1<0$ the all-0 pattern
     has an interior ML estimate, which flexpyirt returns (VALIDATION.md V23).
   - **Degenerate items**: `fit()` emits a `HeywoodWarning` for a `Nominal` item ($K\ge 3$) with a
     free slope whose slope norm is below $10^{-3}$ (the item carries no information and
     $\mathrm{ak}_1..\mathrm{ak}_{K-2}$ are unidentified: the saddle that traps mirt's NR from its
     default starts) or whose $|\mathrm{ak}_k|$ or $|d_k|$ exceeds 50 (diverging categories).
   - **Validation (V23)**: `validation/r/m5_nominal_reference.R` starts mirt's NR at the GPCM point
     (mirt's default nominal starting values stop NR at a saddle on `Science`) and restarts it from
     the best of five alternative mirt starts when that is higher (on the mostly non-monotone V23e
     data the GPCM-point NR stops at a local maximum). flexpyirt fits from its own defaults and
     reaches mirt's solution in all cases within the VALIDATION.md tolerances: unidimensional
     (`Science`, simulated with unordered / negative $\mathrm{ak}$, nominal + 2PL, mostly
     non-monotone items) and a two-factor confirmatory model (V23f). Three-factor nominal models
     are unit-tested only. mirt also leaves one true $+\infty$ V23e pattern at its `max_theta = 20`
     bound; ML extremes are checked against an independent R log-likelihood scan.

7. **Normal-mixture latent distributions (`NormalMixture`, §4.4, V22)**:
   `NormalMixture(n_components, *, mean=0.0, var=1.0, free_mean=False, free_var=False,
   means=None, variances=None, weights=None)` (plus `NormalMixture.free(n_components, ...)`)
   specifies a single-group unidimensional latent density that is a finite mixture of
   $k\in\{2,\dots,10\}$ normal components (Mislevy, 1984), with **one common set of item
   parameters**: a single population with a non-normal latent density. This is not a mixture
   IRT model with class-specific item parameters (Rost, 1990; Bolt, Cohen & Wollack, 2001),
   which is mirt's default for `dentype = "mixture-k"` and is out of scope.
   Multiple-group (`Model.with_groups`) and multidimensional (`Model.with_factors`) models with
   `NormalMixture` raise `SpecificationError`.
   **Raw (gauge) parameterization**: with $\boldsymbol\lambda=(0,\lambda_2,\dots,\lambda_k)$ and
   $\pi=\operatorname{softmax}(\boldsymbol\lambda)$,
   $$h(z)=\sum_{c=1}^k\pi_c\,\phi(z;\tilde\mu_c,\tilde\sigma_c^2),\qquad
     \tilde\mu_1=0,\ \tilde\sigma_1^2=1,$$
   with free `rmean_c` $=\tilde\mu_c$, `rvar_c` $=\tilde\sigma_c^2>0$ and `logit_c`
   $=\lambda_c=\log(\pi_c/\pi_1)$ for $c=2,\dots,k$. These are exactly mirt's `MEAN_1`, `COV_11`
   and `PI` of class `MIXTURE_c` in `multipleGroup(..., dentype = "mixture-k", invariance =
   c(<all items>, "free_means", "free_var"))`, where class 1 is fixed at $N(0,1)$ with `PI = 0`
   and the items are constrained equal across classes (mirt stores the mixing coefficients as
   log ratios to class 1). Raw component 1 only fixes the location/scale gauge of $z$; it is not
   a "reference class" of the population.
   **Analytical standardization**: the raw mixture has exact continuous moments
   $M=\sum_c\pi_c\tilde\mu_c$ and $V=\sum_c\pi_c(\tilde\sigma_c^2+\tilde\mu_c^2)-M^2>0$. As for
   Davidian curves (§16.4), $\theta=\mu+\sigma(Z-M)/\sqrt V$, so on the $\theta$ scale component
   $c$ is $N(\mu_c,\sigma_c^2)$ with
   $$\mu_c=\mu+\sigma\,\frac{\tilde\mu_c-M}{\sqrt V},\qquad \sigma_c^2=\sigma^2\,\frac{\tilde\sigma_c^2}{V},$$
   and $\sum_c\pi_c\mu_c=\mu$, $\sum_c\pi_c(\sigma_c^2+\mu_c^2)-\mu^2=\sigma^2$ exactly for every
   shape. `mean` $=\mu$ and `var` $=\sigma^2$ are fixed at 0 and 1 by default and follow the
   `Normal` identification rules of §4.5 when freed (`free_var=True` for Rasch/PCM,
   `NormalMixture.free(k)` under FIPC); the shape adds $3(k-1)$ free parameters.
   **Why standardize (Decision)**: mirt's own parameterization fixes class 1 at $N(0,1)$ and
   lets the item parameters carry the scale of class 1. Starting items are calibrated for an
   overall $N(0,1)$ (§4.10), which matches no component of a bimodal population, so EM from
   generic starts drifts through badly scaled regions and can stall at inferior stationary
   points (for example a free component absorbing nearly all the mass, $\pi_1\to 0$, where the
   scale is not identified). On the V22a data the same EM (node-level M-step, below) in mirt's
   parameterization stopped 12.7–47.6 log-likelihood points below the maximum from two of four
   starting shapes (component 2 at $+1.5$ or $-1.5$) and needed 3,848–4,505 plain EM iterations
   from the other two; standardized, it reached the maximum from all four in 602–661 plain EM
   iterations (28–93 with SQUAREM), while mirt itself needed 4,555 EM cycles at `TOL = 1e-10`
   from the generating values (VALIDATION.md V22). The
   item parameters are on the overall-standardized scale, which is the same convention as
   `Normal`, `EmpiricalHistogram` and `DavidianCurve` fits and therefore comparable with them.
   Equivalence with mirt: flexpyirt's model on the grid $\theta_q$ is mirt's model on the grid
   $z_q=M+\sqrt V\theta_q$ (for $\mu=0,\sigma^2=1$) with $a=a^{\text{mirt}}\sqrt V$ and
   $d_k=d_k^{\text{mirt}}+s_k\,a^{\text{mirt}}M$ ($s_k=1$ for 2PL/3PL/4PL/GRM, $s_k=k$ for
   GPCM/PCM), because each normalized component density is invariant under the affine map.
   **Grid weights**: each component is normalized on the grid and then mixed, as mirt's
   `dentype = "mixture-k"` stacks per-class normalized densities:
   $$\log w_q=\log\sum_{c=1}^k\pi_c\tilde w_{cq},\qquad
     \log\tilde w_{cq}=\operatorname{log\_softmax}_q\Big(b_q-\frac{(\theta_q-\mu_c)^2}{2\sigma_c^2}\Big),$$
   with the fixed node log base weights $b_q$ (0 on equispaced grids). Each component therefore
   gets exactly the weights a `Normal` with $(\mu_c,\sigma_c^2)$ would get on the same grid, so
   Gauss–Hermite grids (§15.44) are supported as for `Normal`, and $\pi_c$ is the exact grid mass
   of component $c$. Non-positive variances return NaN (rejected by the line search and
   SQUAREM).
   **M-step**: the complete data are (pattern, node), as for every latent distribution: the
   latent part $Q^{\text{lat}}=\sum_q\bar n_q\log w_q(\mu,\sigma^2,\tilde{\boldsymbol\mu},
   \tilde{\boldsymbol\sigma}^2,\boldsymbol\lambda)$ is maximized jointly by the existing masked,
   bounded, monotone Newton (`_newton.py`; gradient and Hessian by autodiff), so EM stays
   monotone, SQUAREM works unchanged, and Oakes' identity (§4.8) and the observed-information
   Hessian need no new code. mirt instead treats the class as further missing data (closed-form
   class proportions clipped to $[10^{-4},1-10^{-4}]$ plus per-class mean/variance updates);
   both EMs have the same fixed points, but the class augmentation adds missing information and
   converges more slowly.
   **Starting values**: `means`, `variances` and `weights` (length $k$; defaults
   $(0,1.5,-1.5,3,-3,\dots)$, all 1, and $1/k$) give the starting shape in any affine scale;
   they are mapped to the raw gauge by $\tilde\mu_c=(m_c-m_1)/\sqrt{v_1}$,
   $\tilde\sigma_c^2=v_c/v_1$, $\lambda_c=\log(\pi_c/\pi_1)$. Identical components are a
   stationary point (the first-order effect of separating them is a location/scale change, which
   the standardization removes, as for Davidian curves at $\boldsymbol\phi=\mathbf 0$), so the
   defaults are distinct.
   **Label switching and degenerate solutions**: the likelihood is invariant to permutations of
   the components (including which physical component is raw component 1) and can have several
   local maxima (McLachlan & Peel, 2000); EM is deterministic and monotone, so labels do not
   switch during a fit, and users should compare fits from several starting shapes. After
   fitting, `fit()` emits a `HeywoodWarning` when a component's weight is below $10^{-3}$ (its
   location and scale are then unidentified and the information matrix is near singular), when
   a component is narrower than the grid ($\sigma_c$ below half the smallest node spacing, so the grid
   cannot resolve it), when a component mean lies outside the grid, or when two components
   nearly coincide ($|\mu_c-\mu_{c'}|<10^{-3}$ and $|\sigma_c^2-\sigma_{c'}^2|<10^{-3}$, a
   collapsed mixture).
   **Reporting, scoring and other methods**: the `ParameterTable` stores `mean`, `var`,
   `rmean_c`, `rvar_c`, `logit_c` under `item="__latent__"`, `model="Mixture"` (priors on latent
   parameters raise `SpecificationError`, as for every latent distribution). `FitResult.latent`
   adds the standardized components `pi_c`, `mean_c`, `var_c` ($c=1,\dots,k$) with delta-method
   SEs (JAX forward-mode Jacobian of the map above); `latent_weights` stays `None`. `score()`
   uses the fitted mixture for `"EAP"` (quadrature grid) and `"PV"` (501-node fine grid, each
   component renormalized there), keeps `"ML"` and `"WLE"` likelihood-only, and rejects `"MAP"`
   with `SpecificationError` because a mixture prior can be multimodal (mirt 1.46.1 also rejects
   MAP for mixtures). `item_fit()` and `m2()` reject `NormalMixture` until validated, `dif()`
   needs a multiple-group fit, and `link()` uses the fitted mixture grid weights by default.
   `"NormalMixture"` is registered in `_json.py` without changing the payload shape
   (`schema_version` stays 3).
   **Validation (V22)**: `validation/r/m5_mixture_reference.R` re-codes the standardized
   discretized likelihood independently in R, polishes mirt's solution to its exact maximum
   (Richardson-Hessian SEs and EAPs in R), and refits built-in `mirt(dentype = "mixture-k")` on
   the image grid $z_q=M+\sqrt V\theta_q$ (asymmetric `theta_lim`), mapping its estimates and
   EAPs to flexpyirt's scale: 2PL $k=2$ (bimodal), GRM $k=2$ (small component) and GPCM $k=3$
   (121 grid points; VALIDATION.md V22). SEs are checked against the R Richardson Hessian, because mirt 1.46.1's
   `SE.type = "Oakes"` for mixtures is not the observed information (its latent SEs are several
   times too small, close to the complete-data SEs with known class membership).
8. **Dimensionality diagnostics (plan, decided 2026-11-24; nothing implemented yet)**:
   **Scope**: two questions are kept apart, because one majority vote over both is meaningless.
   (A) *How many dimensions?* returns an integer $k$ (parallel analysis, MAP, EKC, Hull, CD).
   (B) *Is essential unidimensionality (Stout, 1990) tenable?* returns a verdict, a $p$ value or a
   graded index (DIMTEST, DETECT, local-dependence statistics of a fitted model, $M_2$ / RMSEA,
   model-based parallel analysis, first-to-second eigenvalue ratio). No single method is trusted on
   its own (Auerswald & Moshagen, 2019), so the user-facing entry point is an ensemble.
   **Out of scope**: exploratory MIRT (the product-grid quadrature of §4 does not scale to the
   dimensions an exploratory model needs), so (A) is a diagnostic and does not feed a fit;
   learned predictors (Factor Forest; Goretzko & Bühner, 2020), because they ship a trained model
   that cannot be validated from formulas; the Kaiser criterion is reported for reference only and
   never counted. EGA (Golino & Epskamp, 2017) and bifactor indices (ECV, $\omega_h$, PUC;
   Rodriguez, Reise & Haviland, 2016) are deferred.
   **EFA (revised 2026-11-24)**: the plan was to call `factor_analyzer` as an optional extra, but
   its last release (0.5.1, February 2024) fails in `fit()` with scikit-learn 1.8 and later (it
   passes the removed `force_all_finite` argument to `check_array`). ULS and ML EFA are therefore
   implemented in flexpyirt (§16.14) with SciPy only and checked against EFAtools and `psych::fa`
   on the same matrices; there is no `efa` extra.
   **Stages** (each validated before the next; ROADMAP M5):
   D1 polychoric / tetrachoric correlations (two-step, Olsson, 1979; pairwise `vmap`) with
   smoothing of non-positive-definite matrices, against `psych::polychoric` and `lavaan::lavCor`.
   D2 eigenvalues, parallel analysis (Horn, 1965; polychoric, PCA and FA eigenvalues, percentile
   per Glorfeld, 1995; Garrido, Abad & Ponsoda, 2013), revised MAP (Velicer, Eaton & Fava, 2000) and
   EKC (Braeken & van Assen, 2017), against `psych::fa.parallel`, `psych::vss` and `EFAtools::EKC`
   (deterministic parts exactly; simulated reference eigenvalues by passing R's simulated data or
   with a Monte Carlo tolerance).
   D3 local dependence of a fitted unidimensional model, Q3 (Yen, 1984), Q3* (Christensen, Makransky
   & Horton, 2017), LD-$X^2$ (Chen & Thissen, 1997) and JSI (Edwards, Houts & Cai, 2018), against
   `mirt::residuals`; model-based parallel analysis (Drasgow & Lissak, 1983) against
   `ltm::unidimTest`.
   D4 DETECT, confirmatory and exploratory (Zhang & Stout, 1999; Zhang, 2007), with ASSI and $R$,
   against `sirt::conf.detect` / `expl.detect`; the partition search runs in the outer layer.
   D5 EFA (§16.14) and Hull (Lorenzo-Seva, Timmerman & Kiers, 2011), against EFAtools' `EFA` /
   `HULL` and `psych::fa`; comparison data (Ruscio & Roche, 2012) against `EFAtools::CD`.
   D6 DIMTEST with bootstrap bias correction (Stout, Froelich & Gao, 2001; Froelich & Habing, 2008).
   No CRAN reference exists, so it is accepted on a simulation study (type I error at the nominal
   level for unidimensional data, power for two-dimensional data) recorded in VALIDATION.md.
   **Ensemble** (`dimensionality()`, after D3): each method returns a common result (question A or
   B, the suggested $k$ or verdict, its statistic, the threshold used and its source). The summary
   lists every method; the consensus for (A) is the mode and median of $k$ with the agreement count
   (ties reported, not broken), and for (B) the count of methods supporting essential
   unidimensionality. No weighted voting until a simulation study supports weights. Thresholds
   (DETECT 0.2 / 0.4 / 1, Q3* 0.2, RMSEA) are arguments with their sources in docstrings. The
   polychoric matrix is computed once and shared; simulations use a JAX PRNG key. The ensemble
   itself is checked by a simulation study (unidimensional, two-dimensional and bifactor data).
9. **Polychoric / tetrachoric correlations (`polychoric()`, D1 of §16.8, V24; the JAX version,
   see §16.18 for the NumPy one)**:
   `flexpyirt.polychoric(data, *, smooth=True, zero_cell_correction=0.5, tol=1e-10, max_iter=100)`
   in `correlation.py` (shell) and `_polychoric.py` (jit kernels) returns a frozen
   `PolychoricResult`. **Data**: any numeric codes; each item is recoded to $0..K_j-1$ over its
   observed values (ascending), so unobserved codes are not categories (psych and lavaan do the
   same in effect: an empty category only adds zero-probability cells). An item with one observed
   category, a pair with no jointly observed respondent, or a pair in which one item is constant
   among them raise `SpecificationError`. **Thresholds**: $\hat\tau_{jk}=\Phi^{-1}(\hat P(Y_j\le k))$
   from every observed response to the item (available cases), shared by all pairs (two-step).
   Pair tables use respondents who answered both items. This is lavaan's `missing = "pairwise"`
   and psych's `global = TRUE`; psych switches to per-pair thresholds (`global = FALSE`) when
   items differ in category counts, so it is compared on complete data only.
   **Bivariate normal**: Genz's (2004) `BVNU` with the 20-point Gauss–Legendre rule for every $r$
   (the 6- and 12-point rules are only cheaper), Drezner & Wesolowsky's (1990) expansion for
   $|r|\ge0.925$, every branch evaluated and selected with `jnp.where`, infinite limits exact
   (finite stand-in $\pm40$ inside formulas). A `custom_jvp` supplies the analytic derivatives
   $\partial_r\Phi_2=\phi_2$ and $\partial_h\Phi_2=\phi(h)\Phi((k-rh)/\sqrt{1-r^2})$, so second
   derivatives for Newton are exact too. It agrees with Plackett's integral by adaptive quadrature
   to $10^{-14}$ in absolute value. **Per pair**: cells
   $\pi_{ab}=\Phi_2(\tau_a,\tau'_b)-\Phi_2(\tau_{a-1},\tau'_b)-\Phi_2(\tau_a,\tau'_{b-1})+\Phi_2(\tau_{a-1},\tau'_{b-1})$
   on a grid padded to $K_{\max}$ with $+\infty$ (padded cells have zero counts and probability),
   $\ell=\sum n_{ab}\log\pi_{ab}$ maximized in $z=\operatorname{atanh}\rho$ by Newton (gradient
   step when not concave, step halving, $|z|\le 8$), all pairs under `vmap` + `lax.while_loop`.
   Start: atanh of the clipped Pearson correlation of the codes. Converged when the projected
   gradient is $\le$ `tol` $\times n$ or the step is below $10^{-14}$. **Boundary**: when $\ell$
   increases monotonically to $|\rho|=1$ (e.g. a 2 x 2 table with an empty cell) its gradient in $z$
   decays exponentially and the loop stops early, so after the loop $\ell$ at $\pm8$ (the side of
   $\hat z$) is compared with $\ell(\hat z)$; if it is at least as large the estimate is
   $\pm\tanh 8$, `at_boundary` is set and `HeywoodWarning` is emitted. **Zero cells**: lavaan's
   default (`zero.add = c(0.5, 0)`, `zero.keep.margins = TRUE`): a 2 x 2 table with exactly one empty
   cell gets $+c$ on the empty cell's diagonal and $-c$ on the other, margins unchanged; larger
   tables are not corrected. The default $c=0.5$ follows lavaan because dimensionality methods need
   interior estimates; $c=0$ gives the plain ML. **Smoothing**: psych's `cor.smooth` exactly
   (eigenvalues below $10^{-12}$ set to $10^{-10}$ when the smallest is below machine epsilon,
   rescaled to sum to $p$, rebuilt, rescaled to unit diagonal), `NotPositiveDefiniteWarning`
   (new in `errors`). Not reported: standard errors of $\hat\rho$ (not validated; the
   dimensionality methods do not use them).
   **Validation (V24)**: `flexpyctt/validation/r/m5_polychoric_reference.R` (lavaan 0.7-3, psych 2.6.9):
   simulated tetrachoric (8 items), mixed 2–5 categories with two factors, the same with 10% MCAR
   (pairwise), extreme binary items with one-empty-cell 2 x 2 tables, LSAT7 and Science. lavaan
   agrees to $\le 3.1\times10^{-8}$ (its `nlminb` tolerance; flexpyirt's pair gradients vanish to
   $10^{-7}n$), thresholds to $10^{-12}$; psych to $\le 2\times10^{-5}$ on complete data without
   corrected pairs (its `optimize()` tolerance); `cor.smooth` to $10^{-12}$.
10. **Parallel analysis, MAP and EKC (`parallel_analysis()`, `velicer_map()`,
    `empirical_kaiser()`, D2 of §16.8, V25)**: in `dimensionality.py` (shell; NumPy for the
    $p\times p$ linear algebra, JAX `vmap` for normal reference data; now in flexpyctt with NumPy
    random numbers and `rng` instead of `key`, §16.18). All three answer question
    (A) and return frozen results with `n_factors`. Matrix inputs (DataFrame, array or
    `PolychoricResult`) must be square, finite, symmetric, unit-diagonal, $p\ge3$.
    **Counting**: leading eigenvalues strictly above the reference, stopping at the first miss
    (psych's and EFAtools' sequential rule). **SMC eigenvalues**: $R$ with $h_j=1-1/(R^{-1})_{jj}$
    on the diagonal (Humphreys & Ilgen, 1969); the EFA-communality variant of psych /
    EFAtools waits for the EFA bridge (D5). **Parallel analysis references**: `"permutation"`
    (default) permutes each item independently with a JAX PRNG key (`(R, p, N)` permutations by
    nested `vmap`), so marginals and missingness are kept, then recomputes the correlation with
    the validated `polychoric()` (or pairwise Pearson) per replication, warnings silenced; a
    replication whose permuted pair is not estimable (missing data) is skipped and counted, with a
    `FlexpyirtWarning`, and fewer than two usable replications raise `SpecificationError`.
    `"normal"`: i.i.d. $N(0,1)$ data of the same $N\times p$ (rows of incomplete data count),
    Pearson correlations, chunked `vmap`. Quantiles use NumPy's default (linear interpolation, R's
    type 7). Default quantile 0.95 (Glorfeld, 1995), PCA eigenvalues and the permutation
    reference with polychoric correlations, following Garrido, Abad & Ponsoda (2013).
    **MAP**: $f_m$ for $m=0,\dots,p-1$ with principal-component loadings
    $V_m\Lambda_m^{1/2}$ (negative eigenvalues clipped at 0); power 4 is elementwise (O'Connor,
    2000), and the default because the revised test is the recommended one (Velicer, Eaton &
    Fava, 2000). The loop stops with NaN once a partial variance is $\le10^{-10}$ and the minimum
    ignores NaN. EFA.dimensions' fourth-power statistic is a matrix power (documented, not
    followed). **EKC**: Braeken & van Assen (2017) as in EFAtools (observed eigenvalues in the
    remaining-variance term); with a `PolychoricResult` and no `n_obs`, the smallest pairwise
    $n$ is used (conservative under missing data).
    **Validation (V25)**: `flexpyctt/validation/r/m5_dimensionality_reference.R` feeds the six lavaan
    polychoric matrices of V24 to EFAtools 1.1.0 (`EKC`), psych 2.6.9 (`vss()$map`),
    EFA.dimensions 0.1.9.3 (`MAP`) and an independent R implementation of both MAP powers, plus
    `eigen()` for PCA / SMC eigenvalues: all agree to $10^{-11}$, including the selected numbers of
    factors. Normal reference eigenvalues (3 sizes x PCA / SMC, 20000 data sets each) agree with
    EFAtools' `PARALLEL` within Monte Carlo error ($|z|\le4.5$ for the means with a fixed key,
    $6\times10^{-3}$ for the 95th percentiles). The permutation path is checked by properties
    (independent permutations, recovery of two factors, reproducibility by key).
11. **Local dependence and model-based parallel analysis (`FitResult.local_dependence()`,
    `model_parallel_analysis()`, D3 of §16.8, V26)**: both need a single-group unidimensional
    model with a `Normal` latent distribution (`FitResult._require_unidimensional_normal`).
    `FitResult._traces(prob, nodes)` evaluates every item's category probabilities at arbitrary
    nodes by replacing `CompiledProblem.nodes` (the kernels only read the nodes), so the same
    code serves the fitted grid, EAP scores and simulated $\theta$. **Q3**: residuals
    $x_{ij}-\sum_k kP_{jk}(\hat\theta_i)$ with EAP $\hat\theta$ from `score()` (missing responses
    allowed), pairwise-complete Pearson correlations (mirt's `na.omit` per pair). **Q3\***: Q3 minus
    the mean off-diagonal Q3 (Christensen, Makransky & Horton, 2017), so `max_q3_star` is their
    $Q3_{\max}-\overline{Q3}$. **LD** (Chen & Thissen, 1997, as mirt's `residuals(type = "LD")`):
    per pair the full $K_i\times K_j$ observed table over respondents answering both, the expected
    table $n_{ij}\sum_q P_{ia}P_{jb}w_q$ with the fitted grid weights (identical to mirt's
    normalized normal density on its 61-point grid), $X^2$, $G^2$ (cells with $O>0$), $p$ from
    $\chi^2_{(K_i-1)(K_j-1)}$, and $V=\operatorname{sign}(\gamma_O-\gamma_E)\sqrt{X^2/(n
    \min(K_i-1,K_j-1))}$ with Goodman–Kruskal $\gamma$ (sign $+$ on a tie, $V$ NaN when a $\gamma$
    is undefined). **Model-based PA** (Drasgow & Lissak, 1983): statistic = second eigenvalue of the
    observed smoothed polychoric matrix; `_simulate_responses` draws $\theta\sim N(\hat\mu,
    \hat\sigma^2)$ and responses by inverting uniforms against cumulative category probabilities,
    and copies the data's missingness; $B$ replications through `polychoric()` (warnings
    silenced, unusable ones skipped with a `FlexpyirtWarning`); $p=(1+\#\{T_b\ge T\})/(B+1)$.
    `eigen_type="max"` (default) sets the whole diagonal to the largest off-diagonal correlation,
    as ltm's `unidimTest` does; `"smc"` and `"pca"` reuse §16.10. ltm draws $\theta$ from
    $N(\mathrm{EAP}_i,\mathrm{SE}_i^2)$ when given a fitted object; flexpyirt uses the fitted
    latent distribution (the reference generating model), which ltm does when given parameters.
    The parameters come from the same data, so the $p$-value is conservative under a correct
    model (30 simulated 2PL data sets, $B=50$: mean $p$ 0.66, 1 of 30 below 0.05); the docstring
    and the guide say so.
    **JSI** (Edwards, Houts & Cai, 2018): item 17 below.
    **Validation (V26)**: `validation/r/m5_local_dependence_reference.R` (mirt 1.46.1, ltm 1.2.0):
    LSAT7 2PL, Science GRM and a simulated 2PL with one locally dependent pair and 5% MCAR.
    flexpyirt refits the models (log-likelihoods agree to $10^{-9}$ relative) and agrees with
    `residuals()` to $10^{-5}$ (relative, $X^2$, $G^2$) and $10^{-6}$ ($V$, Q3; observed $\le
    4\times10^{-8}$). ltm's `unidimTest` on LSAT7 with mirt's estimates (`thetas = cbind(d, a1)`,
    $B=4000$): observed eigenvalues to $2\times10^{-4}$ (polycor's `optimise()` tolerance; observed
    $2.4\times10^{-5}$), the simulated second-eigenvalue mean within $|z|\le4.5$ (observed
    $|z|<1$ with 2000 replications on two keys) and its 95th percentile to 0.02.
    **Extension beyond single-group unidimensional Normal models (M7, issue #81, V26)**:
    `local_dependence()` takes `group=` (as `item_fit()`) and works on the compiled problem of
    the fitted model, so one code path serves every latent distribution and grid: the weights
    are `_latent_log_weights` (`(G, Q)`; the EH weights when the model has them), the traces
    `_item_log_probs` (`(G, J_f, Q, K)`), and the EAP `_core.eap` with each respondent's own
    group as the prior. (a) **Multiple groups**: every statistic is computed within a group,
    from that group's respondents, item parameters and latent weights, as mirt's
    `residuals()` does for a `multipleGroup` fit (it loops over the groups' single-group
    objects); `pairs` gets a leading `group` column, `q3` a `(group, item)` row index, `q3_mean`
    is a Series indexed by group, Q3\* subtracts the group's own mean and `max_q3_star` is the
    largest over all groups. Groups without respondents in `data` are skipped. The tables are
    always $K_i\times K_j$, also when a category is not observed in a group (mirt's
    `table()` drops it and fails there). (b) **`EmpiricalHistogram`, `DavidianCurve`,
    `NormalMixture`**: $E_{ab}$ uses the fitted grid weights and Q3 the EAP under the fitted
    distribution. mirt's default for `dentype = "EH"` / `"Davidian-k"` fits is different: it
    takes a $N(0,1)$ prior in both; the fitted density is used with
    `residuals(type = "LD", use_dentype_estimate = TRUE)` and, for Q3, with
    `Theta = fscores(m, use_dentype_estimate = TRUE)` (`type = "Q3", use_dentype_estimate =
    TRUE` evaluates the residuals at the grid nodes instead of the scores in mirt 1.48 and is
    not usable). flexpyirt follows the fitted density because the Normal prior contradicts the
    fitted model; the docstring states the mirt calls that agree. (c) **Confirmatory 2–3
    dimensional models**: $E_{ab}=n\sum_qP_{ia}(\boldsymbol\theta_q)P_{jb}(\boldsymbol\theta_q)
    w_q$ on the product grid with the fitted multivariate Normal weights, and
    $d_{ij}=x_{ij}-E(X_j\mid\hat{\boldsymbol\theta}_i)$ at the vector EAP; `_traces` evaluates
    the kernels at `(N, D)` nodes. The EAP uses the fitted grid, where mirt's `fscores()` uses
    its own default grid whatever the fit used (121 / 61 / 31 points per dimension for 1 / 2 /
    3 factors), so the two agree when the grids are the same. $df$ stays $(K_i-1)(K_j-1)$ in every case (mirt's), which
    ignores estimated parameters; the statistics are diagnostic (Chen & Thissen, 1997).
    **Not extended**: bifactor / two-tier models (after the bifactor PR), `jsi()`,
    `model_parallel_analysis()` and `dimensionality(fit=...)`, which keep
    `_require_unidimensional_normal`. mirt's JSI needs one factor, has no SE for Davidian
    curves (`"No standard error method currently supported"`), does not support mixtures, and
    for `multipleGroup` refits each group alone as a single-group $N(0,1)$ model while taking
    the slopes from the multiple-group fit, which is not the refit of the fitted model; there
    is no reference to validate an extension against. **Validation (V26, M7)**:
    `validation/r/m7_local_dependence_reference.R` (mirt 1.48), nine cases (two groups with one
    and two factors, EH, Davidian, mixture, two and three factors), with mirt's estimates put
    into a `FitResult` with every parameter fixed: $X^2$ / $G^2$ to $10^{-7}$ relative
    (observed $\le8\times10^{-9}$), $V$ / Q3 to $10^{-8}$ (observed $\le3\times10^{-15}$);
    details and mirt's differing defaults in VALIDATION.md.
12. **The `dimensionality()` ensemble (§16.8)**: one polychoric matrix (with `fit`, the data are
    first coerced to the fitted items) feeds question (A): `parallel_analysis()` (defaults),
    `velicer_map()` (power 4) and `empirical_kaiser()`; the key is split into one key for each
    simulation-based method. Question (B), only with `fit`: `model_parallel_analysis()` ($p\ge
    \alpha=0.05$), largest Q3\* $<0.2$ and `m2()` RMSEA $\le0.05$ (Maydeu-Olivares & Joe, 2014);
    an `m2()` `SpecificationError` (e.g. missing data) is reported as an unavailable row
    (`supports_unidimensionality=None`), not hidden. Consensus (A): mode (every value on a tie),
    median and number agreeing; (B): number of available methods supporting. No weights, and the
    two questions are never merged (§16.8). Thresholds are arguments with their sources in the
    docstring; the 0.2 rule for Q3\* is labelled as a rule of thumb. `DimensionalityResult.summary()`
    prints every method, its detail and both consensus lines. The module and the function share
    the name `dimensionality` (as `fit`), so `flexpyirt.dimensionality` is the function; tests
    reach the module with `importlib.import_module`. Checked in `tests/test_m5_ensemble.py` by
    recomputing each row with the individual functions and the same keys.
13. **DETECT (`detect()`, `exploratory_detect()`, D4 of §16.8, V27)**: `_detect.py` (NumPy; now
    in flexpyctt, §16.18).
    **Conditional covariances**: Zhang & Stout (1999): for each pair the ML covariance within
    groups of equal total score $S$ and within groups of equal rest score $S-X_i-X_j$, each
    averaged with the group proportions, then the two averaged (the total-score estimate is biased
    negatively, the rest-score one positively). Groups are formed with `np.unique` and
    `np.bincount`, one pass per pair. Raw item scores are used (polytomous items allowed);
    complete data are required (the scores are undefined otherwise), so `dimensionality()`
    reports DETECT as unavailable with missing data. **Indices** (sirt's `detect.index`):
    DETECT $=100\,\overline{\delta\,\mathrm{ccov}}$, ASSI, RATIO, MADCOV100, MCOV100, unweighted
    and weighted by $\sqrt{n_{ij}}$ (identical with complete data; kept for sirt's layout). The
    smoothed (kernel) conditional covariances of sirt's default `conf.detect` are not
    implemented. **Exploratory**: rows split into halves (`floor(N/2)` estimation rows from a JAX
    key, or given), Ward clustering of the estimation-half matrix
    $d=\max(\mathrm{ccov})-\mathrm{ccov}$ (the maximum includes the zero diagonal, as sirt's
    `create.ccov`). R's `hclust(method = "ward.D")` applies the Lance–Williams Ward update to $d$;
    SciPy's `"ward"` applies it to squared distances, so SciPy gets $\sqrt d$ (merge heights are
    R's after squaring). `cutree` labels clusters by first appearance; flexpyirt relabels SciPy's
    `fcluster` output the same way. Each partition $k=2..$`max_clusters` is scored on both halves;
    `n_clusters` maximizes the estimation-half DETECT (sirt's choice). Zhang & Stout's genetic
    partition search is not implemented. **Thresholds** (docstrings, guide): DETECT $<0.2$ / $0.4$
    / $1$, ASSI $0.25$, RATIO $0.36$ (Jang & Roussos, 2007; Zhang, 2007; §16.8 had 0.1 / 0.5 / 1,
    corrected here). **Ensemble**: `dimensionality()` adds the validation-half DETECT of the selected
    partition as a question-(B) row (supports when $<$ `detect_threshold` $=0.2$), available
    without `fit`; the key is split three ways (parallel analysis, model-based PA, DETECT).
    **Validation (V27)**: `flexpyctt/validation/r/m5_detect_reference.R` (sirt 4.2.133): simulated
    two-dimensional (true and wrong partitions), unidimensional and Science (polytomous) data with
    `conf.detect(use_sum_score = TRUE, bias_corr = TRUE, smooth = FALSE)`, and a three-cluster data
    set with `expl.detect(use_sum_score = TRUE, estsample = )`: conditional covariances,
    all five indices (both weightings), partitions and both halves' indices agree to
    $10^{-12}$ (observed $10^{-14}$), and Ward merge heights to $10^{-12}$. With the default
    `smooth = TRUE`, sirt's `conf.detect(use_sum_score = TRUE)` standardizes the total score and then
    subtracts raw item scores for the rest score, so its rest-score groups are wrong (DETECT 6.73
    instead of 1.60 for `sim_2d`); recorded on the accuracy page, not followed.
14. **EFA and the Hull method (`efa()`, `hull()`, D5 of §16.8, V28)**: `_efa.py` (NumPy / SciPy;
    now in flexpyctt, §16.18).
    **Estimation**: unrotated solutions from the uniquenesses $\Psi$, minimized by SciPy L-BFGS-B
    (`ftol=1e-15`, `gtol=1e-11`) from $\psi_i=1-\mathrm{SMC}_i$ within $[0.005,\max(1,\mathrm{SMC})]$
    (psych / EFAtools' bounds). ULS: $F=\tfrac12\lVert R-\Psi-\Lambda\Lambda^\top\rVert_F^2$, with
    $\Lambda$ the leading eigenvectors of $R-\Psi$ scaled by $\sqrt{\max(\lambda,0)}$, gradient
    $\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)$ (psych's `FAgr.minres`, which is the exact
    gradient of this $F$); this is psych's minres and EFAtools' ULS (psych's `fit.residuals` sums
    only off-diagonal residuals, which has the same minimizer in the interior). ML: Jöreskog's
    concentrated discrepancy on the eigenvalues of $\Psi^{-1/2}R\Psi^{-1/2}$ with gradient
    $\operatorname{diag}(\Lambda\Lambda^\top+\Psi-R)/\psi^2$ (psych's `FAfn` / `FAgr`). Columns are
    signed to non-negative sums; no rotation. A uniqueness at the floor is a Heywood case
    (`HeywoodWarning` from `efa()`). The tolerances are tighter than R's `optim()` defaults, so the
    flexpyirt objective is never above EFAtools' at its solution (checked in the tests).
    **Fit indices** (EFAtools' `.gof`): CAF $=1-$KMO of the residual matrix with unit diagonal;
    $\chi^2$ with Bartlett's multiplier $N-1-(2p+5)/6-2k/3$ and its $p$; CFI and RMSEA from the
    uncorrected $F(N-1)$ and the null $-\log|R|(N-1)$, $df_0=p(p-1)/2$; $df=((p-k)^2-(p+k))/2$.
    **Hull** (EFAtools' `HULL` / `.hull_calc`): rows $k=0..J$ with fit CAF, CFI or $1-$RMSEA
    ($k=0$: $1-$KMO($R$), 0, $1-$RMSEA$_0$; $df_0$); drop solutions whose fit does not increase,
    remove interior points on or below the chord of their neighbours until none is, scree values
    $st$ on the remaining interior, select the largest $st$ (fewer than three hull points: the best
    fit, with a warning). Default $J$: one more than parallel analysis with SMC eigenvalues of normal
    data and the mean rule (EFAtools' `efa_parallel` defaults, here 1000 replications from a JAX key),
    capped at the largest $k$ with $df>0$ and raised to at least $\min(3,\text{cap})$; with
    `max_factors` given, $J$ is taken as is. At least 6 variables. The table records `heywood` and
    `converged` per solution, and a `HeywoodWarning` is emitted when the selected one is
    inadmissible (as EFAtools). **Ensemble**: `dimensionality()` adds Hull (CAF, ULS, default $J$ from a
    fourth split of the key) to question (A) when there are at least 6 items.
    **Validation (V28)**: `flexpyctt/validation/r/m5_efa_hull_reference.R` (EFAtools 1.1.0, psych 2.6.9) on the
    V24 polychoric matrices with $\ge6$ variables and two simulated Pearson matrices (3 factors,
    $N=500$; 2 weak factors, $N=300$): every $k\le J$ with ULS and ML — implied
    $\Lambda\Lambda^\top$ and communalities to $10^{-4}$ (observed $\le7\times10^{-5}$, the largest at
    Heywood solutions), CAF / CFI / RMSEA to $2\times10^{-4}$, $\chi^2$ to $10^{-4}$ relative, $df$
    exactly; psych's `fa()` on the 3-factor matrix to $10^{-4}$. Hull for every matrix x estimator x
    fit index with EFAtools' recorded $J$: the selected $k$ and the hull membership identical, the
    fit values to $10^{-5}$ (observed $\le1.4\times10^{-6}$), $st$ to 1% relative (a ratio of small
    differences; observed $\le3.5\times10^{-3}$ absolute). Comparison data is §16.15.
15. **Comparison data (`comparison_data()`, D5b of §16.8, V29; now in flexpyctt with `rng`
    instead of `key`, §16.18)**: `_comparison_data.py` (NumPy;
    random draws from a JAX key are passed in, so the generator is deterministic given them).
    For $k=1..K$ ($K$ = largest $k$ with positive $df$, or `max_factors`): resample each variable's
    observed values `n_population` times and sort them; draw $F$ $(N,k)$ and $E$ $(N,p)$ standard
    normal; generate the population (Ruscio & Kaczetow, 2008, as EFAtools'
    `.simulate_cfm_empirical`): principal axis factoring of an intermediate $R^*$ (EFAtools'
    `.paf_iter`: unit starting communalities, absolute eigenvalues, criterion $10^{-3}$ on the
    largest change, `max_iter` 50), loadings clipped to $[-1,1]$ and flipped when the first is
    negative, scores $F\Lambda^\top+E\,\mathrm{diag}(\sqrt{\max(1-h^2,0)})$, then for each column in
    turn the rows are reordered by that column (stable sort) and the column replaced by its sorted
    resampled values; $R^*$ starts at $R$ and moves by the residual $R-R_{\text{rep}}$ while the RMSE
    of the lower-triangle residuals improves (from 1), otherwise halves the step from the best
    $R^*$, stopping after 5 failures in a row. `n_samples` bootstrap samples of the data's size are
    drawn from the population, and the RMSE of their Pearson-correlation eigenvalues against the
    observed ones is recorded. $k$ is kept while the one-sided Mann–Whitney test (normal
    approximation with continuity and tie corrections, as R's `wilcox.test` for $n\ge50$) says
    the RMSE with $k$ factors is smaller than with $k-1$ at $\alpha=0.30$ (Ruscio & Roche's
    recommendation); the result is the last kept $k$. Pearson correlations only (as Ruscio & Roche
    and EFAtools); complete data. PAF loadings are signed to non-negative column sums: EFAtools
    takes LAPACK's signs, which only matter for $k\ge2$ and do not change the population's
    distribution. **Ensemble**: `dimensionality()` adds CD to question (A) with complete data
    (fifth split of the key). **Validation (V29)**: `flexpyctt/validation/r/m5_comparison_data_reference.R`
    (EFAtools 1.1.0): PAF loadings for $k=1..3$ to $10^{-12}$ in $\Lambda\Lambda^\top$; the one-factor
    population from EFAtools' own random draws (reproduced with the same seed and exported)
    identical; the rank-sum $p$-values on EFAtools' RMSE columns to $10^{-10}$ relative; `CD()` on a
    3-factor, a 2-factor ordinal and a 1-factor binary data set: the same suggestion for two keys
    each and mean RMSEs within 15% (different random populations).
16. **DIMTEST (`dimtest()`, D6 of §16.8, V30)**: `_dimtest.py` (NumPy; random draws are
    arguments; now in flexpyctt with `rng` instead of `key`, §16.18). Binary complete data, at least 4 items. **Statistic** (Nandakumar & Stout,
    1993): respondents grouped by the PT score; groups smaller than `min_group` (20) or with
    $S_k\le0$ dropped; $T_L=K^{-1/2}\sum_k(\hat\sigma_k^2-\hat\sigma_{U,k}^2)/S_k$ with
    $\hat\sigma_k^2$ the (divide-by-$J_k$) variance of the AT score, $\hat\sigma_{U,k}^2=\sum_i\hat
    p_{ik}(1-\hat p_{ik})$ and $S_k^2=(\hat\mu_{4,k}-\hat\sigma_k^4+\hat\delta_{4,k})/J_k$,
    $\hat\delta_{4,k}=\sum_i\hat p_{ik}(1-\hat p_{ik})(1-2\hat p_{ik})^2$. **Bias correction**
    (Stout, Froelich & Gao, 2001): kernel-smoothed IRFs of the test sample (abilities
    $\Phi^{-1}(r/(N+1))$ of total-score ranks with ties broken by uniform draws, Gaussian kernel
    with bandwidth $1.1N^{-1/5}$ on 161 points of $[-4,4]$; Ramsay, 1991), $B$ (100) unidimensional
    data sets of the test sample's size with $\theta\sim N(0,1)$ and IRFs interpolated linearly,
    $T=(T_L-\bar T_G)/\sqrt{1+1/B}$, $p=1-\Phi(T)$. **AT selection**: with `assessment_items` the
    whole sample is tested; otherwise a random third (`training_fraction`) selects the AT and the
    rest is tested (Froelich & Habing, 2008, use a separate subsample): among the clusters of the
    Ward clusterings into 2–6 groups of the training conditional covariances (§16.13), with 2 to
    `max_assessment_size` (half the items) members, the one whose split {AT, rest} has the largest
    DETECT. This is in the spirit of Froelich & Habing's conditional-covariance AT selection, not
    their exact ATFIND, and is documented as such. **Ensemble**: `dimensionality()` adds DIMTEST
    ($p\ge\alpha$) to question (B) for complete binary data (sixth split of the key), otherwise an
    unavailable row. **Validation (V30)**: no implementation outside the stand-alone DIMPACK, so the
    test is accepted on `flexpyctt/validation/simulation/dimtest_study.py` (results in
    `dimtest_study.json`): 2PL data, slopes $U(0.8,1.8)$, intercepts $N(0,0.8)$, items in two halves
    on dimensions correlated $\rho$, 200 replications per condition, defaults throughout:

    | Data | $N$ | $J$ | $\rho$ | mean $T$ | SD $T$ | reject at .05 | reject at .10 |
    |---|---|---|---|---|---|---|---|
    | unidimensional | 1000 | 20 | 1 | -0.07 | 0.72 | 0.005 | 0.020 |
    | unidimensional | 2000 | 20 | 1 | -0.12 | 0.71 | 0.010 | 0.030 |
    | unidimensional | 2000 | 30 | 1 | -0.03 | 0.73 | 0.010 | 0.025 |
    | two-dimensional | 1000 | 20 | 0.7 | 4.85 | 0.97 | 0.990 | 0.995 |
    | two-dimensional | 2000 | 20 | 0.7 | 6.94 | 0.88 | 1.000 | 1.000 |
    | two-dimensional | 2000 | 30 | 0.8 | 6.50 | 0.86 | 1.000 | 1.000 |

    (Rerun on 2026-11-25 with flexpyctt's NumPy generator, §16.18. With the JAX keys used before
    the move the rejection rates at .05 were 0.020, 0.005, 0.000 and 0.990, 1.000, 1.000.)

    The test is conservative: $\bar T_G$ removes the bias of $T_L$ (mean $T$ near 0) but $T_L$'s
    standard deviation under the null is about 0.72 rather than 1, so the type I error is below the
    nominal level; power against two dimensions correlated 0.7–0.8 is 0.99–1. A small $p$ is
    therefore strong evidence of multidimensionality, a large $p$ weak evidence for unidimensionality
    (stated in the docstring and the guide). Standardizing by the bootstrap SD of $T_G$ instead would
    recalibrate it but departs from the published statistic; not done. The pieces are unit-tested
    against direct computations, and `flexpyctt/tests/test_dimtest.py` reruns a small version of the
    study (20 null and 10 alternative replications).
17. **Jackknife slope index (`FitResult.jsi()`, V31)**: `_jsi.py`. **Definition** (Edwards, Houts
    & Cai, 2018, as mirt's `residuals(type = "JSI")`): the jackknife is over items, not
    respondents (an earlier version of item 11 said "subgroups"; corrected). For each item $j$ the
    model is re-estimated without item $j$ and $\mathrm{JSI}_{ji}=(\hat a_i-\hat a_i^{(-j)})/
    \mathrm{SE}(\hat a_i^{(-j)})$ for every other item $i$, the SE from the refit; the matrix is not
    symmetric, and mirt's `fold = TRUE` adds its transpose (the `folded` column of `pairs`). mirt
    gives no cutoff and neither does flexpyirt; the docstring says to look for values that stand
    out. **Removing an item = masking**: item $j$'s responses are set to missing (`y[:, j] = -1`,
    all-zero one-hot rows), so it drops out of $\ell$ while every array keeps its shape; the
    removed item index is a traced argument and one jit compilation serves every $j$ (§5). The
    parameters kept are the free ids used by any other row (other items and the latent
    distribution), so across-item equality constraints and priors carry over; item $j$'s own ids
    stay at their estimates and are ignored. Items without a free slope (Rasch, PCM, fixed
    slopes) give NaN columns. Single-group unidimensional `Normal` models only (mirt requires one
    factor); at least three items. **Methods**, all from the full estimates $\hat\eta$:
    `"refit"` (default): Newton–Raphson on $\ell_{(-j)}+\log\pi$ over the kept ids with step
    halving and a ridge when the information is not positive definite, until the largest step is
    below `tol` ($10^{-8}$; `ConvergenceWarning` after `max_iter`); the SE is from the observed
    information (`information_hessian`, equal to Oakes' at any point) of the last evaluation.
    Newton from a warm start converges in about five steps; EM would take tens. `"newton"`:
    `newton_steps` steps of the same iteration, the SE from the information of the last
    evaluation (the start for one step). `"one_step"` (the Julia sketch the owner proposed):
    $\hat\eta^{(-j)}\approx\hat\eta+V\nabla\ell_{(-j)}(\hat\eta)$ with the full model's $V$
    (`FitResult.vcov`) and the full model's SE in the denominator; the $J$ gradients come from one
    `vmap`, so it costs about $J$ E-steps. One EM step was rejected: it moves only the fraction
    $1-$(missing information) of the way (JSI shrunk by about a third in a pilot study).
    **Accuracy of the approximations** (largest absolute difference from the refit; 2PL / GRM
    with one locally dependent pair):

    | Data | $J$ | refit | `"newton"` 1 step | 2 steps | `"one_step"` |
    |---|---|---|---|---|---|
    | 2PL, $N=1000$ | 15 | 0.7 s | 0.012 (0.20 s) | 0.010 | 0.021 (0.02 s) |
    | 2PL, $N=2000$ | 30 | 4.6 s | 0.002 (1.2 s) | <0.001 | 0.002 (0.06 s) |
    | 2PL, $N=3000$ | 60 | 45 s | 0.001 (15 s) | <0.001 | <0.001 (0.2 s) |
    | GRM(4), $N=2000$ | 30 | 17 s | 0.002 (5.1 s) | <0.001 | 0.002 (0.06 s) |
    | LSAT7 2PL | 5 | | 2.0 | 0.88 | 1.3 |
    | Science GRM | 4 | | 2.7 | 2.9 | 2.5 |
    | GPCM(4), $N=800$ | 8 | | 0.11 | 0.45 | 0.39 |

    With few items removing one is not a small perturbation and the Taylor approximations fail
    (two steps can be worse than one before Newton's quadratic phase); with 15 or more items
    `"one_step"` is within 0.02 of the refit at a few hundredths of its cost. Below 10 items
    (`_JSI_APPROXIMATION_MIN_ITEMS`) the approximations warn and point to the refit.
    **Validation (V31)**: `validation/r/m5_jsi_reference.R` (mirt 1.46.1, TOL $=10^{-10}$ in the
    fits and the refits): LSAT7 2PL, Science GRM, a simulated 2PL with two dependent pairs and 5%
    MCAR, and a simulated GPCM with one dependent pair; the refit agrees to $10^{-5}$ (observed
    $\le1.2\times10^{-7}$). The tests also check the refit against `fit()` on the data without the
    item, the approximations against the refit on a 15-item data set (`"newton"` 1 step 0.03, 2
    steps 0.02, `"one_step"` 0.05) and the vmapped gradients against single ones.
    **Scope (decided 2026-11-26, #81)**: JSI is defined only for single-group unidimensional
    models (one slope per item; Edwards, Houts & Cai, 2018). Its multidimensional /
    multiple-group extension (including bifactor and two-tier models) and its use with non-Normal
    latent distributions have not been studied, and there is no reference implementation (mirt's
    `type = "JSI"` needs one factor; see also item 11 and §17.3), so it is not implemented:
    `jsi()` raises `SpecificationError` for multiple-group, multidimensional and non-`Normal`
    latent models, with a message saying why.
18. **Package split: `flexpyctt` and `flexpyequate` (decided 2026-11-25, issue #87)**: tools that need no fitted IRT
    model are released as separate, JAX-free distributions from this repository. flexpyirt keeps
    BA-EM estimation and everything that uses its result.

    > **Superseded in part (2026-11-27, §16.24).** The two packages now have their own
    > repositories and versions. The paragraphs on layout, the single version, tooling, the
    > documentation site and where tests and references are describe the state before that;
    > the rest still holds.

    | Distribution | Scope | Depends on |
    |---|---|---|
    | `flexpyirt` | estimation; tools that take a `FitResult` (`local_dependence()`, `jsi()`, `model_parallel_analysis()`, `link()`, IRT equating); the `dimensionality()` report | JAX, `flexpyctt` |
    | `flexpyctt` | model-free item and test analysis: `polychoric()`, `parallel_analysis()`, `velicer_map()`, `empirical_kaiser()`, `efa()`, `hull()`, `comparison_data()`, `detect()`, `exploratory_detect()`, `dimtest()`; observed-score DIF (`mantel_haenszel()`, `mantel()`, `generalized_mantel_haenszel()`, `logistic_dif()`, §16.20); classical item analysis (`score_responses()`, `item_analysis()`, `coefficient_alpha()`, `coefficient_omega()`, `distractor_analysis()`, §16.21) | NumPy, SciPy, pandas |
    | `flexpyequate` | equating of score distributions only: mean, linear and equipercentile equating, the anchor-test methods, log-linear presmoothing, kernel equating (its own design document: `packages/flexpyequate/docs/DESIGN.md`) | NumPy, SciPy, pandas |

    **Layout**: a uv workspace. flexpyirt stays at the repository root (`pyproject.toml`,
    `src/flexpyirt/`, `tests/`); the issue's `packages/flexpyirt/` is postponed, because moving
    the directory would conflict with every open branch. Members are `packages/*`:
    `packages/flexpyctt/` has its own `pyproject.toml`, `README.md` (the PyPI page, absolute
    links), `LICENSE`, `src/flexpyctt/`, `tests/` (with `tests/reference/`) and `validation/`.
    flexpyirt depends on flexpyctt (`[tool.uv.sources] flexpyctt = { workspace = true }`), so
    `pip install flexpyirt` still provides every tool of those two. `packages/flexpyequate/` has
    the same layout (plus `docs/DESIGN.md`; its R scripts are `validation/r/equate_*.R` at the
    root). **flexpyequate is independent**: it imports neither JAX, flexpyirt nor flexpyctt, and
    neither of them depends on it or re-exports it (`pip install flexpyirt` does not install it).
    The root lists it in the `dev` dependency group only, so that `uv sync` installs it for the
    tests; it is not in flexpyirt's sdist. Equating that needs item parameters stays in
    flexpyirt. **One version**: all three read the same
    `vX.Y.Z` tag through hatch-vcs (`raw-options.root = "../.."` in the members). The release
    workflow builds each with `uv build --package` and publishes flexpyctt before flexpyirt;
    flexpyequate has a PyPI job of its own that waits for neither.
    flexpyirt imports a few private helpers of flexpyctt (below), so the two are meant to be
    installed at the same version; the dependency carries no version bound yet (open point: pin
    `flexpyctt==<same version>` with a build hook).

    **flexpyctt never imports JAX**, and has no 64-bit requirement (NumPy is float64). A CI job
    installs it alone, asserts that `jax` is absent and runs its tests. A second job does the
    same for flexpyequate (`jax`, `flexpyirt` and `flexpyctt` absent).
    **Tooling**: ruff and mypy for flexpyirt and flexpyctt are configured in the root
    `pyproject.toml`. flexpyequate keeps its own `[tool.ruff]` and `[tool.mypy]` with the same
    rules (ruff picks them up for that directory when run from the root; mypy runs from
    `packages/flexpyequate/`), because its tests are a package named `tests`, like the root's,
    and one mypy run cannot hold two packages of that name. **Documentation site**: flexpyctt's
    tools are documented in the user guide and the API reference (flexpyirt re-exports them);
    flexpyequate has one page, `docs/equating.md`, into which `scripts/mkdocs_hooks.py` puts the
    package README (the PyPI page), so the text exists once.

    **Two polychoric implementations.** `flexpyctt.polychoric()` is a new NumPy / SciPy
    implementation of the estimator of §16.9 (same definition, options, defaults, warnings and
    `PolychoricResult`):
    - $\Phi_2$ by Owen's (1956) $T$ function (`scipy.special.owens_t`):
      $\Phi_2(h,k;\rho)=\tfrac12\Phi(h)+\tfrac12\Phi(k)-T(h,a_h)-T(k,a_k)-\beta$ with
      $a_h=(k-\rho h)/(h\sqrt{1-\rho^2})$, $a_k=(h-\rho k)/(k\sqrt{1-\rho^2})$, $\beta=0$ if
      $hk>0$ or ($hk=0$ and $h+k\ge0$), else $\tfrac12$; $\Phi_2(0,0;\rho)=\tfrac14+
      \arcsin\rho/(2\pi)$; infinite limits are exact. This is independent of the JAX kernel's
      Genz (2004) quadrature, so the agreement of the two is a real check.
    - Newton's method in $z=\operatorname{atanh}\rho$ for all pairs at once (array operations,
      no Python loop over pairs), with the analytic derivatives
      $\partial_\rho\pi_{ab}=\Delta\phi_2$ (Plackett, 1954) and
      $\partial_\rho\phi_2=\phi_2\,[\rho/s+(hk(1+\rho^2)-\rho(h^2+k^2))/s^2]$, $s=1-\rho^2$; the
      same step rule (Newton when concave, else a half step along the gradient; steps clipped to
      $\pm2$; step halving), bound $|z|\le8$, convergence criterion and boundary check as §16.9.

    `flexpyirt.polychoric()` stays the JAX version (`_polychoric.py`, `correlation.py`), public
    and unchanged in behaviour; it returns the same `PolychoricResult` class (defined in
    flexpyctt), so either result can be passed to `velicer_map()`, `efa()` etc. Every moved tool
    uses the NumPy version, as do `model_parallel_analysis()` and `dimensionality()`.
    **Validation**: the NumPy version passes V24 with the same tolerances (lavaan $10^{-6}$,
    thresholds $10^{-12}$, psych $10^{-3}$), its $\Phi_2$ agrees with Plackett's integral to
    $10^{-14}$, and a test in flexpyirt requires the two versions to agree to $10^{-9}$ on every
    V24 case, with equal flags.

    **Random numbers.** flexpyctt functions take `rng` (a `numpy.random.Generator` or an integer
    seed) where the JAX versions took `key`; `rng` is required where `key` was. The streams
    differ from the JAX ones, so results with reference data change within Monte Carlo error;
    the deterministic reference values (V24, V25, V27–V29) are unchanged. The functions that stay
    in flexpyirt keep `key`: `model_parallel_analysis()` simulates with it, and `dimensionality()`
    splits it into six keys as before and turns each into a generator for the flexpyctt call
    (`dimensionality._rng_from_key`: `numpy.random.default_rng` seeded with the key's data).

    **Errors and warnings.** flexpyctt defines `FlexpycttError`, `SpecificationError`,
    `FlexpycttWarning`, `ConvergenceWarning`, `HeywoodWarning` and `NotPositiveDefiniteWarning`
    (`flexpyctt.errors`). `flexpyirt.errors` re-exports the same class objects, and
    `FlexpyirtError` / `FlexpyirtWarning` are aliases of the flexpyctt base classes, so
    `except flexpyirt.errors.SpecificationError` (or `FlexpyirtError`) also catches what a
    flexpyctt function raises. `IdentificationError` and `PrecisionError` stay in flexpyirt.

    **Re-exports (decided: yes).** `flexpyirt` re-exports every moved function and result class
    (`flexpyirt.parallel_analysis is flexpyctt.parallel_analysis`), so code written against
    `develop` keeps its imports; the only API change is `key=` → `rng=`. The exception is
    `polychoric`: `flexpyirt.polychoric` is the JAX version. Documentation presents the moved
    tools under flexpyctt.

    **Where things are.** `flexpyctt.correlation` (`polychoric`, `PolychoricResult`,
    `smooth_correlation`), `flexpyctt._polychoric` (kernels), `flexpyctt.dimensionality` (the
    nine functions and their result classes), `flexpyctt._efa`, `_detect`, `_dimtest`,
    `_comparison_data` (moved unchanged). `flexpyirt.dimensionality` keeps `dimensionality()`,
    `model_parallel_analysis()`, `LocalDependenceResult`, `JSIResult`,
    `ModelParallelAnalysisResult`, `DimensionalityResult` and imports flexpyctt's private
    helpers `_coerce_ordinal`, `_observed_correlation`, `_eigenvalues` and
    `_descending_eigenvalues`. **Tests and references**: the tests of the moved tools and
    `v24`, `v25`, `v27`, `v28`, `v29` are in `packages/flexpyctt/tests/` (no JAX), their R
    scripts and the DIMTEST study in `packages/flexpyctt/validation/`. flexpyirt's tests keep
    the JAX polychoric tests (reading `v24` from the flexpyctt directory), the two-version
    agreement test, V26 and the ensemble tests.

19. **$M_2$ / $M_2^*$ and S-X2 for confirmatory multidimensional models (M6, #77, V39)**:
    `FitResult.m2()` (§16.3) and `FitResult.item_fit()` (§15.40) accept single-group models
    declared with `Model.with_factors()` ($D\in\{2,3\}$, product grid, §15.29–§15.30). Both
    statistics are integrals of products of item category probabilities over the latent density,
    so nothing in their definition depends on $D$: the node index $q$ runs over the
    $Q=n^D$ nodes $\boldsymbol\theta_q$ of the fitted product grid and the weights are the
    normalized multivariate-normal weights
    $w_q(\boldsymbol\rho)=\phi_D(\boldsymbol\theta_q;\mathbf 0,\mathbf R)/\sum_{q'}\phi_D(\boldsymbol\theta_{q'};\mathbf 0,\mathbf R)$
    of the fit (§15.30), with $P_{jk}(\boldsymbol\theta)$ from the same item kernels
    ($\boldsymbol a_j^\top\boldsymbol\theta$ in the linear predictor; unloaded slopes fixed at 0).
    **$M_2^*$.** With $m_{1,jq}=\sum_k kP_{jk}(\boldsymbol\theta_q)$ and
    $m_{2,jq}=\sum_k k^2P_{jk}(\boldsymbol\theta_q)$,
    $$\pi_j=\sum_q m_{1,jq}w_q,\qquad \pi_{ij}=\sum_q m_{1,iq}m_{1,jq}w_q\quad(i<j),$$
    and $\Xi_2$ is built from the same sums as in §16.3 (local independence holds given the
    whole vector $\boldsymbol\theta$, so $E[y_iy_j\mid\boldsymbol\theta]=m_{1,i}m_{1,j}$ for
    $i\ne j$ whether or not the two items load on the same factor). The Jacobian
    $\Delta_2=\partial\pi_2/\partial\eta^\top$ has, besides the item columns
    $$\frac{\partial\pi_j}{\partial\xi_j}=\sum_q\frac{\partial m_{1,jq}}{\partial\xi_j}w_q,\qquad
      \frac{\partial\pi_{ij}}{\partial\xi_i}=\sum_q\frac{\partial m_{1,iq}}{\partial\xi_i}m_{1,jq}w_q,$$
    one column per free factor correlation $\rho_{ab}$ ($a<b$):
    $$\frac{\partial\pi_{ij}}{\partial\rho_{ab}}=\sum_q m_{1,iq}m_{1,jq}\frac{\partial w_q}{\partial\rho_{ab}},\qquad
      \frac{\partial w_q}{\partial\rho_{ab}}=w_q\Big(g_{ab,q}-\sum_{q'}w_{q'}g_{ab,q'}\Big),\qquad
      g_{ab,q}=\big[\mathbf R^{-1}\boldsymbol\theta_q\boldsymbol\theta_q^\top\mathbf R^{-1}-\mathbf R^{-1}\big]_{ab}$$
    (and likewise for $\pi_j$). In the code all columns come from one `jax.jacfwd` of the
    margins with respect to $\eta$ (the latent parameters enter through `_latent_log_weights`),
    so no formula is written per parameter type; the expressions above are what the R reference
    recomputes independently. $df=J+\binom J2-q$ with $q$ = `n_params`, which includes the free
    correlations; RMSEA, its interval, SRMSR, and (dichotomous models) TLI / CFI against the
    independence model are unchanged, the null model having no latent structure. **Difference
    from mirt**: as for a free mean or variance (§16.3), mirt's `M2()` sets the `GroupPars`
    estimation flags to `FALSE` when it builds $\Delta_2$, so a free correlation has no column
    there although it is subtracted in $df$. flexpyirt keeps every free parameter in $\Delta_2$
    (Maydeu-Olivares & Joe, 2005, 2006); its $M_2$ is therefore slightly smaller than mirt's for
    a model with free correlations ($3\times10^{-7}$ to $7\times10^{-4}$ relative in V39). The
    private `_latent_in_delta=False` reproduces mirt's value. **Identification from the
    margins** is checked as before (rank of $\Delta_2$, $df>0$): for example a three-category
    GRM item has more free parameters than a small test has rows involving it, and a factor
    measured by a single item is not identified.
    **S-X2.** The summed score is the total over all items (not per factor), as in mirt. The
    Lord–Wingersky recursion is run at every node of the product grid,
    $$L_j(s\mid\boldsymbol\theta_q)=\sum_{k=0}^{K_j-1}L_{j-1}(s-k\mid\boldsymbol\theta_q)P_{jk}(\boldsymbol\theta_q),$$
    and the expected proportions integrate over all factors,
    $$E_j[s,k]=\frac{\sum_qP_{jk}(\boldsymbol\theta_q)L_{-j}(s-k\mid\boldsymbol\theta_q)w_q}{\sum_qL(s\mid\boldsymbol\theta_q)w_q}.$$
    `_itemfit.lord_wingersky` and `s_x2_expected_tables` take a list of `(Q, K_j)` arrays and a
    weight vector and never look at the node coordinates, so they are used unchanged with the
    flattened grid ($O(J^2SQ)$ work, about a second for $Q=15^3$ and nine items). Collapsing
    (`mirt:::collapseCells`) and $df_j$ = kept cells − rows − free parameters of item $j$ − free
    latent parameters (the free correlations; an item with a cross-loading has one more free
    slope) are as in §15.40. mirt's `itemfit(fit_stats = "S_X2")` does the same on
    `thetaComb(seq(-6, 6, length.out = quadpts), nfact)` with `mirt_dmvnorm` weights; its default
    `quadpts` (`select_quadpts(nfact)`: 31 for two factors, 15 for three) and limits are those of
    `mirt()` and of `Quadrature.default()`, but mirt's `itemfit()` and `M2()` do not read the
    grid of the fit, so a fit made with another `quadpts` or `theta_lim` must pass them again in
    R (flexpyirt always uses the grid of the fit).
    **Not supported** (each `SpecificationError`): bifactor and two-tier models (their
    reduced E-step has no joint grid; the statistics need the block structure of §17), and,
    for S-X2, multiple-group multidimensional models (not validated; $M_2$ handles them, §16.23).
    Unidimensional results are unchanged bit for bit (the same code path with $D=1$).
    **Validation (V39)**: mirt 1.48 on the four V14 models, see VALIDATION.md.
20. **Observed-score DIF in flexpyctt (`mantel_haenszel()`, `mantel()`,
    `generalized_mantel_haenszel()`, `logistic_dif()`, `DIFResult`; issue #102, V47)**: DIF
    screening without an IRT model, in `flexpyctt.dif` (public layer: input checks, matching
    scores, purification, tables) and `flexpyctt._dif` (NumPy kernels for one studied item). The
    API is the one approved on #102, with the changes listed at the end.

    **Common arguments.** `data`, keyword-only `group` (one label per row), `focal` (one label;
    a list for the generalized test; every other row is the reference group, as difR's
    `focal.name`), `items=None` (tested items), `anchor_items=None` (items whose total is the
    matching score), `matching=None` (external matching variable; not with `anchor_items` or
    `purify`), `purify=False`, `max_iter=10`, `p_adjust="none"` (`"holm"`, `"BH"`; R's
    `p.adjust` on the finite p-values), `alpha=0.05`. **Matching**: the total of the anchor
    items, plus the studied item when it is not an anchor (difR's rule); every distinct value is
    a stratum (no thick matching). **Missing values** on the tested or anchor items raise
    `SpecificationError` (owner decision 3 of #102: error, not silent row deletion; difR's
    `rowSums(na.rm = TRUE)` would mix totals over different item sets).

    **Purification** (difR `purify = TRUE`, `puriadjType = "simple"`): step 0 tests every item
    on the given anchors; while the flagged set (unadjusted $p<\alpha$) is non-empty, the next
    step uses the anchors minus the flagged tested items; it stops when the flagged set equals
    the previous one (`converged=True`) or after `max_iter` steps (`ConvergenceWarning`; the table
    is the last step, whose anchors exclude the items flagged one step earlier, as difR). Nothing
    flagged at step 0 gives `n_iter=0`, `converged=True`. With user `anchor_items`, purification
    removes flagged items from that set only (difR ignores `purify` when `anchor` is given; this
    is the natural extension, tested against a direct call with the final anchors). The final
    `flagged` is `adj_p < alpha`. `flag_history` keeps difR's `difPur` (one row per step).

    **Mantel–Haenszel** (Mantel & Haenszel, 1959; Holland & Thayer, 1988): strata with
    $T_k>1$; $\chi^2_{MH}=(\lvert\sum A-\sum E(A)\rvert-c)^2/\sum\operatorname{Var}(A)$ with
    $c=1/2$ under `continuity_correction=True` **always** (Holland & Thayer, difR; R's
    `mantelhaen.test()` sets $c=0$ when the difference is below $1/2$, which changes the statistic
    of items without DIF only); $\hat\alpha_{MH}=\sum AD/T/\sum BC/T$ with $A,B$ the reference
    right / wrong counts (so $\hat\alpha_{MH}>1$ favours the reference group); Phillips & Holland
    (1987) variance of $\ln\hat\alpha_{MH}$; $\Delta_{MH}=-2.35\ln\hat\alpha_{MH}$,
    $SE(\Delta)=2.35\,SE(\ln\hat\alpha)$. **ETS classes** (owner decision 2: the full rule by
    default; Zieky, 1993; Zwick, 2012): A if $\lvert\Delta\rvert<1$ or $\chi^2_{MH}$ is not
    significant ($p\ge\alpha$ on the `p_adjust`-adjusted p-value, the one `flagged` uses, so B
    and C items are always flagged; without adjustment it is the plain $p$); C if
    $\lvert\Delta\rvert\ge1.5$ and $(\lvert\Delta\rvert-1)/SE(\Delta)>z_{1-\alpha}$ (a one-sided
    test that $\lvert\Delta\rvert>1$, 1.645 at $\alpha=0.05$; some authors use 1.96); B otherwise. `classification="effect_size"`: A $<1\le$ B $<1.5\le$ C
    (difR's printed codes use right-closed intervals; only exact ties differ). A zero or infinite
    $\hat\alpha_{MH}$ gives a `FlexpycttWarning` (difR's "sparse data" warning).

    **Mantel** (1963) for ordered items: the focal score sum against its hypergeometric
    expectation, $df=1$, item values as scores; equal to $\chi^2_{MH}$ without correction for 0 / 1
    items. **Liu–Agresti** (1996) cumulative common odds ratio
    $\hat\psi=\sum_k\sum_{j<L}b_{jk}c_{jk}/T_k\big/\sum_k\sum_{j<L}a_{jk}d_{jk}/T_k$ ($a,c$: reference
    / focal counts at or below category $j$), **oriented as $\hat\alpha_{MH}$** (it reduces to
    $\hat\alpha_{MH}$ for 0 / 1 items; difR's `liu_agresti_ccor()` reports the reciprocal). Its
    SE is **Liu & Agresti's (1996) eq. 3**, which generalizes Robins, Breslow & Greenland (1986)
    and is consistent both with a fixed number of large strata and with many sparse strata
    ("dually consistent"). With the focal group as their row 1,
    $\widehat{\operatorname{Var}}(\ln\hat\psi)=\sum_k\hat\xi_k\big/(\sum_k\sum_jb_{jk}c_{jk}/T_k)^2$,
    $\hat\xi_k=\sum_j\hat\phi_{jjk}+2\sum_{j<s}\hat\phi_{jsk}$,
    $\hat\phi_{jsk}=\frac{n_{Fk}n_{Rk}}{T_k^2}\{\hat\psi d_{sk}a_{jk}/n_{Fk}\,[1+(\hat\psi-1)a_{sk}/n_{Rk}]
    +c_{jk}b_{sk}/n_{Rk}\,[\hat\psi-(\hat\psi-1)c_{sk}/n_{Fk}]\}$ ($j\le s$). Matching on a
    total score gives many small strata, where a large-strata delta method
    ($\operatorname{Var}(S-\hat\psi R)/S^2$ with plug-in multinomial variances, used before
    review) is far too small: about 40% at four respondents per stratum. The estimator is not
    invariant to reversing the category order; Liu & Agresti note that one can average the two
    orderings and that the difference is trivial in practice. flexpyctt uses the categories in
    increasing order as published, without averaging. difR's own SE is wrong (VALIDATION V47).

    **Generalized MH** (Landis, Heyman & Koch, 1978; Somes, 1986; Penfield, 2001): per stratum
    ($T_k>1$) the $G\times C$ table of groups x observed categories; observed focal counts for
    categories $1..C-1$ against $T\,\mathbf p\otimes\mathbf q$, covariance
    $T^2/(T-1)\,(D_p-pp^\top)\otimes(D_q-qq^\top)$ restricted to those cells; $df=(G-1)(C-1)$.
    A singular summed covariance (a category or group absent from every informative stratum) is
    handled with the pseudo-inverse and $df=$ rank, with a `FlexpycttWarning`. It serves both
    several focal groups with 0 / 1 items (difR `difGMH()`) and nominal categories
    (`mantelhaen.test()` on an $I\times J\times K$ table).

    **Logistic regression** (Swaminathan & Rogers, 1990; Zumbo, 1999): models
    $S$, $S+G$, $S+G+SG$ fitted by Newton–Raphson in NumPy (no new dependency), from
    $\beta=0$, until the max relative step is below $10^{-12}$ (100 iterations at most; a
    singular information matrix, as under separation, stops it with a `ConvergenceWarning` and a
    pseudo-inverse covariance). LRT = difference of deviances; Wald on the added coefficients of
    the larger model (difR's `criterion = "Wald"`). Nagelkerke $R^2$ with $N$ = rows, $\Delta R^2$
    = larger minus smaller model. Zumbo–Thomas (0.13, 0.26) and Jodoin–Gierl (0.035, 0.07)
    classes on $\Delta R^2$ only, as difR prints them (the published rules add significance,
    which is the `flagged` column); unlike the ETS default, no significance is folded in because
    the cut points were proposed as effect sizes to read next to the test.

    **Changes from the API proposed on #102.** (1) `mantel()`: `cumulative_odds_ratio_se` became
    `log_cumulative_odds_ratio_se`, next to a new `log_cumulative_odds_ratio`, because the SE is of
    the log (as `log_odds_ratio_se` of `mantel_haenszel()`). (2) `mantel()`, `generalized_mantel_haenszel()`
    and `logistic_dif()` have a `df` column (also for the fixed-$df$ methods, so tables of
    different methods line up). (3) `DIFResult` also has `flag_history`, `method` and the property
    `flagged_items`. (4) The logistic coefficient columns are named `group_coefficient`,
    `group_coefficient_se`, `interaction_coefficient`, `interaction_coefficient_se`. (5) Owner
    decision 4 (standardization, STD P-DIF): not included, as in the scope list of #102.
    **Not in this PR**: ordinal logistic DIF (the second PR of #102).
21. **Classical item analysis (flexpyctt, #101, V46)**: `flexpyctt/item_analysis.py` (NumPy /
    SciPy / pandas), re-exported by flexpyirt. The API is the one proposed in the issue and approved
    by the owner (2026-11-26): `score_responses()`, `item_analysis()`, `coefficient_alpha()`
    (`AlphaResult`), `coefficient_omega()` (`OmegaResult`), `distractor_analysis()`. The names do not
    clash with flexpyirt's model-based `reliability()`. **Changes from the issue's sketch** (no
    signature changed): `AlphaResult` also carries `confidence_level` and `if_deleted` has both
    `alpha` and `standardized_alpha`; `OmegaResult` also carries the `correlation` matrix it was
    fitted to and the `reversed_items` reversed before computing omega; `score_responses()` also
    accepts the key as a sequence in column order.
    **Missing data** (owner decision 2: the issue's `"pairwise"` default): everything is computed
    from the pairwise-complete covariance matrix $C$ (R's `cov(use = "pairwise")`) and, for
    standardized alpha and omega, the pairwise-complete correlation matrix (R's
    `cor(use = "pairwise")`, not `cov2cor(C)`), as psych's `alpha()`. `"listwise"` drops incomplete
    rows first. Rows with no response are dropped in both cases, so `n_obs` (the Feldt $n$) counts
    respondents with at least one response; psych uses `nrow(x)`.
    **Item–total and item–rest correlations** are $c_j/\sqrt{C_{jj}\,\mathbf 1^\top C\mathbf 1}$ and
    $(c_j-C_{jj})/\sqrt{C_{jj}(\mathbf 1^\top C\mathbf 1-2c_j+C_{jj})}$ with $c_j=\sum_l C_{jl}$,
    the Pearson correlations with $T$ and $T-x_j$ on complete data. psych's `r.drop` is the same
    formula; its `raw.r` correlates with `rowSums(x, na.rm = TRUE)`, which treats a missing
    response as 0 in the total and so is not the quantity alpha is built from. flexpyctt keeps the
    $C$-based definition for both columns so that they, `alpha` and `alpha_if_deleted` describe
    the same matrix; the difference from `raw.r` under missing data is tested and documented.
    **Polyserial / biserial** (`correlation="polyserial"`): the two-step estimator of Olsson,
    Drasgow & Dorans (1982), $\hat\rho=r\,\sigma_j/\sum_k\phi(\tau_{jk})$, with $\sigma_j$ the
    item's SD (denominator $n$) and $\tau_{jk}=\Phi^{-1}$ of its cumulative proportions, both over
    the item's observed scores; $r$ is the Pearson item–total (–rest) correlation above. This is
    polycor's `polyserial(ML = FALSE)`, psych's `polyserial()` on complete data and, for two
    categories, $r\sqrt{pq}/\phi(\Phi^{-1}(p))$. The issue's "thresholds from the margins, then
    $\rho$" means this estimator: polycor's two-step version does not maximize over $\rho$ (the
    maximum-likelihood polyserial, `ML = TRUE`, is not provided). psych's `biserial()` divides by
    the $n-1$ SD of the total while using $p,q$, so it is this value times $\sqrt{(n-1)/n}$
    (tested). Estimates outside $[-1,1]$ are set to $\pm1$ with a `HeywoodWarning`, as psych does.
    `difficulty` is the mean over the largest observed score, so scores must be $\ge0$.
    **Omega**: $\omega_t=1-\sum_j(1-\lambda_j^2)/\mathbf s^\top R\mathbf s$ from the one-factor ULS
    solution (`_efa.fit_factors`, the kernel of `efa()`), with $s_j=\operatorname{sign}(\lambda_j)$
    ($+1$ for a zero loading), i.e. psych's `omega.tot` (Revelle & Zinbarg, 2009) with its default
    `flip = TRUE` and the observed total variance in the denominator. Items that load negatively
    are reversed, as psych does (review of #127): otherwise one reverse-keyed item makes
    $\mathbf 1^\top R\mathbf 1$ small and omega describes a total nobody would use. The reversal is
    not silent: a `FlexpycttWarning` names the items and `OmegaResult.reversed_items` lists them,
    while `loadings` and `correlation` stay those of the items as scored; reverse-scoring the items
    first gives the same omega (tested). The model-implied form
    $(\sum\lambda)^2/((\sum\lambda)^2+\sum\psi)$ equals it only under perfect fit and is not
    reported. Calling the kernel instead of `efa()` admits three items (a just-identified model,
    which `efa()` rejects for having no degrees of freedom). The Pearson matrix is smoothed with
    `smooth_correlation` when not positive definite (pairwise matrices may not be), as
    `polychoric()` does. lavaan's one-factor CFA (listed in the issue) was not used as a reference:
    it fits ML to the covariance matrix, a different estimator on a different scale.
    **Keyed scoring and options**: options are compared with `==` per cell, so codes of any type
    work; a key option never chosen warns (usually a key in the wrong codes or type). In
    `distractor_analysis()` a missing response scores 0 in $T$; each option's statistics use the
    item's respondents; the key is listed even if nobody chose it. Score groups follow R's
    `cut(T, quantile(T, 0:G/G), include.lowest = TRUE)` (type-7 quantiles, ties never split); equal
    quantiles raise instead of silently merging groups.
    **Validation (V46)**: `flexpyctt/validation/r/ctt_item_analysis_reference.R`, see
    VALIDATION.md. **Left out** (as in the issue): hierarchical $\omega$, split-half, Guttman
    $\lambda$, glb, the upper–lower 27% index, plots (#103), weights.
22. **DIF workflow: screening with anchor purification, Wald tests, effect sizes (#104, V49)**:
    post-fit orchestration in the Result layer (`dif.py`, `_dif.py`); only the refits run in
    `jit`. Scope: unidimensional multiple-group models (two or more groups, every item family,
    every latent family), MLE only (no priors), no sampling weights, no equality constraints
    across items in the specification. Each test is an omnibus test across all groups.
    - **Wald tests (`FitResult.dif(..., test="wald")`)**: for each tested item, with
      $\boldsymbol\eta$ the free parameters of the fit and $\mathbf V$ = `vcov`, the
      $r=\sum_p(m_p-1)$ contrasts $\eta_{p,u_1}-\eta_{p,u_k}$ ($k=2..m_p$) between the $m_p$
      distinct `free_id`s $u_k$ of parameter $p$ across groups form $\mathbf L$ and
      $W=(\mathbf L\boldsymbol\eta)^\top(\mathbf L\mathbf V\mathbf L^\top)^{-1}
      (\mathbf L\boldsymbol\eta)\sim\chi^2_r$ (Wald, 1943; Lord, 1980), as mirt's
      `DIF(Wald = TRUE)`. No refit; needs `se=` at fit time; only `scheme="add"` (the
      parameters must be free across groups). The contrasts are on flexpyirt's natural
      parameters, so for 3PL/4PL `g`/`u` (mirt tests them on the logit scale) the statistic
      differs from mirt's; 2PL, GRM, GPCM, Rasch and nominal parameters are the same as mirt's.
      The columns are `item`, `x2` (the statistic $W$), `df`, `p`, `adj_p`.
    - **`dif_screen(spec, data, *, group, items, anchor_items, strategy, test, params,
      p_adjust="BH", alpha=0.05, max_iter=10, method, quadrature, se, item_names)`** builds the
      models itself from `spec` (its `equal_across_groups()` calls are replaced): every
      parameter of an untested item (`anchor_items` and items not in `items`) and every
      parameter outside `params` is equal across groups; the `params` of a tested item are equal
      ("constrained") or free per group ("free") according to the state of the procedure. With
      $F$ the set of free tested items, the test of item $i$ in state $F$ compares
      $F\cup\{i\}$ with $F\setminus\{i\}$, i.e. item $i$ with every other item as in $F$:
      likelihood ratio $X^2=2(\ell_{F\cup\{i\}}-\ell_{F\setminus\{i\}})$,
      $df=k_{F\cup\{i\}}-k_{F\setminus\{i\}}$, or the Wald statistic of the fit of
      $F\cup\{i\}$. Fits are cached by $F$, and each refit starts at the estimates of the model
      of the current round. Each round adjusts the latest $p$ of every tested item over the
      tested items (`p_adjust`, default `"BH"` for screening) and flags $\tilde p<\alpha$.
      Strategies (Candell & Drasgow, 1988; Kopf, Zeileis & Strobl, 2015; mirt's `DIF()`
      schemes):
      - `"all_others"` (mirt `scheme = "drop"` from the fully invariant model): one round with
        $F=\varnothing$, each item tested with all others as anchors.
      - `"sequential_free"` (`"drop_sequential"`): round 1 as `"all_others"`; then, while the
        flags change and at most `max_iter` rounds, $F$ = flagged items and the unflagged items
        are retested against $F$ (flagged items keep their latest statistic).
      - `"sequential_constrain"` (`"add_sequential"`): needs at least one untested anchor item;
        round 1 tests every item with $F$ = all tested items (only the anchors constrained);
        then $F$ = flagged items and the flagged items are retested against $F$.
      The procedure has converged when a round reproduces the flags of the round before
      (round 1 is compared with the starting $F$); otherwise a `ConvergenceWarning` after
      `max_iter` rounds. mirt's `max_run = r` equals `max_iter = r - 1`.
      The result `DIFScreenResult` holds `table` (one row per tested item: `item`, `x2`, `df`,
      `p`, `adj_p`, `flagged`, `converged`), `flagged_items`, `anchor_items` (all items not
      flagged, in model order), `history` (one row per round and tested item: `round`, `item`,
      `retested`, `x2`, `df`, `p`, `adj_p`, `flagged`), `fit` (the final model, $F$ = flagged
      items, with `se=`), `n_rounds` and `converged`. For `"all_others"` the table is round 1;
      for the sequential strategies it is the test of every item against the final $F$ (each
      item against the purified anchors; for flagged items this is mirt's final output), with
      `adj_p` adjusted over the table; `flagged` is the decision of the procedure. A
      refit that does not converge is recorded in `converged` (no warning per refit).
    - **`dif_effect_sizes(result, *, data=None, group=None, focal=None, items=None,
      density="focal", item_names=None)`** (`result` a multiple-group `FitResult` or a
      `DIFScreenResult`): for each focal group $f$ (default every group after the first, the
      reference $r$) and item $j$, with expected item scores $E_{gj}(\theta)=\sum_kkP_{gjk}(\theta)$
      and $D_j(\theta)=E_{fj}(\theta)-E_{rj}(\theta)$ on the fitted grid $\theta_q$ with the
      normalized fitted latent weights $w_q$ of the focal (or, `density="reference"`, reference)
      group:
      $\text{signed}_j=\sum_qw_qD_j(\theta_q)$, $\text{unsigned}_j=\sum_qw_q|D_j(\theta_q)|$
      (Chalmers, 2018; mirt `DRF(DIF = TRUE, best_fitting = TRUE)` `sDIF` / `uDIF`). With
      `data`, also the expected-score standardized difference (Meade, 2010; mirt
      `empirical_ES()` `ESSD`) at the EAP scores $\hat\theta_i$ of the $n$ focal respondents:
      $\big(\overline{E_{f j}(\hat\theta)}-\overline{E_{r j}(\hat\theta)}\big)/
      \sqrt{(s^2_{f}+s^2_{r})/2}$ (sample variances, denominator $n-1$; the pooled SD of two
      samples of equal size), and the root mean squared deviation of each of the two groups
      $g$ (`rmsd_reference`, `rmsd_focal`): with the E-step expected counts $\bar r_{gjkq}$
      of the group, $\hat E_{gj}(\theta_q)=\sum_kk\bar r_{gjkq}/\sum_k\bar r_{gjkq}$ and
      $\mathrm{RMSD}_{gj}=\big(\sum_qw_{gq}(\hat E_{gj}(\theta_q)-E_{gj}(\theta_q))^2\big)^{1/2}$
      (expected-score form; for dichotomous items the category-1 RMSD of von Davier, 2005, and
      Köhler, Robitzsch & Hartig, 2020; mirt `RMSD_DIF(probfun = FALSE)`). The RMSD measures
      how well each group's parameters fit that group, so it detects DIF for items whose
      parameters are equal across groups (an invariant or partially invariant fit). One more
      row per focal group (`level="test"`, `item="test"`) sums $D_j$ over `items`
      (signed / unsigned DTF, mirt `DRF(DIF = FALSE)`; the expected test score standardized
      difference, mirt `ETSSD`; RMSD NaN). Columns: `group`, `level`, `item`,
      `signed_difference`, `unsigned_difference` and, with `data`,
      `expected_score_standardized_difference`, `rmsd_reference`, `rmsd_focal`. Signs are focal
      minus reference (mirt's). Not in scope: mirt's default DRF density (the posterior counts
      of the data), DRF's sampling-based tests and intervals, pairwise comparisons of focal
      groups, multidimensional models.
23. **$M_2$ / $M_2^*$ for multiple groups, missing data and MAP fits (M6, #76, V42)**:
    `FitResult.m2(data, *, group=None, item_names=None, ci=0.90)` extends §16.3 / §16.19 to
    multiple-group models (unidimensional or confirmatory multidimensional, `Normal` latent
    distributions), to responses with missing values, and to fits with priors. All three rest on
    one property of the statistic: with $\Delta_c$ an orthonormal basis of
    $\operatorname{col}(\Delta)^\perp$, $\Delta_c^\top\{\hat p-\pi(\hat\eta)\}
    =\Delta_c^\top\{\hat p-\pi(\eta_0)\}+O_p(N^{-1})$ for **any** $\sqrt N$-consistent
    $\hat\eta$, so the null distribution depends only on the asymptotic covariance of the sample
    margins $\hat p$, never on that of $\hat\eta$ (Maydeu-Olivares & Joe, 2005, 2006).
    **What mirt 1.46.1 does** (read from the source of `M2()`): (i) on a `MultipleGroupClass` it
    computes margins, $\Xi_2$ and $\Delta$ per group, stacks the residuals
    $\sqrt{N_g}(\hat p_g-\pi_g)$, makes $\Xi$ and $\Delta$ block-diagonal over groups, sums the
    columns of parameters constrained equal across groups, and uses $df=G\,s-q$ and
    RMSEA with the total $N$; SRMSR is reported per group; (ii) the `impute` argument of earlier
    versions is gone (`impute <- 0` in the body): with missing values every margin is the mean
    over the respondents who answered the item (pair), $N_a$ of them, and the statistic is
    $\sum_{a,b}(N_a/\sqrt{N})\,d_a\,C_{ab}\,(N_b/\sqrt N)\,d_b$ with $d=\hat p-\pi$ and $C$ the
    complete-data $C_2$ (the `N.ratio` rescaling), which is not the inverse covariance of these
    residuals; (iii) priors are ignored: the statistic is evaluated at the MAP estimates with
    every free parameter counted in $df$.
    **Multiple groups.** With $N=\sum_gN_g$ and $\rho_g=N_g/N$, the stacked residual
    $r=(\sqrt{N_g}\{\hat p_g-\pi_g(\hat\eta)\})_g$ has
    $r\approx(\sqrt{N_g}\{\hat p_g-\pi_g(\eta_0)\})_g-\Delta^\ast\sqrt N(\hat\eta-\eta_0)$ with
    $\Delta^\ast=(\sqrt{\rho_g}\,\Delta_g)_g$, the rows of group $g$ scaled by $\sqrt{\rho_g}$,
    and covariance $\Gamma=\operatorname{blockdiag}(\Gamma_g)$ (groups independent). flexpyirt
    computes
    $$M_2=r^\top\Delta_c^\ast(\Delta_c^{\ast\top}\Gamma\Delta_c^\ast)^{-1}\Delta_c^{\ast\top}r,\qquad
      df=G\,s-q,$$
    with $\Delta_c^\ast$ the complement of $\operatorname{col}(\Delta^\ast)$ and the columns of
    $\Delta=\partial(\pi_1,\dots,\pi_G)/\partial\eta^\top$ from one `jax.jacfwd` over the shared
    free vector (a parameter equal across groups is one column, as for §16.1). This equals the
    usual form $N e^\top\Delta_c(\Delta_c^\top\Gamma_N\Delta_c)^{-1}\Delta_c^\top e$ with
    $\Gamma_N=\operatorname{blockdiag}(\Xi_g/\rho_g)$. The row scaling changes the column space
    only for a column shared by groups of different sizes: mirt omits it, so its value differs
    from flexpyirt's when parameters are equal across groups of unequal size (and, as in §16.3,
    when the latent mean / variance / correlation of a group is free, since mirt drops those
    columns). The private `_mirt_convention=True` of `_m2.m2_statistic` reproduces mirt's
    $\Delta$ (unscaled rows; no latent columns is `_latent_in_delta=False`) and its missing-data
    weighting, for validation only. RMSEA uses the total $N$; `srmsr` is reported per group (`srmsr_<group>` columns, as
    mirt's `SRMSR.<group>`); the independence model for TLI / CFI has its own intercepts in every
    group (mirt's `computeNullModel(group =)`), so $df_{\text{null}}=G\binom J2$ and, for complete
    data, $M_{2,\text{null}}=\sum_gN_g\sum_{i<j}r_{g,ij}^2$.
    **Missing data: pairwise-available margins with their exact covariance under MCAR.** With
    $o_{na}=1$ when respondent $n$ answered every item of margin $a$ (an item or a pair),
    $N_a=\sum_no_{na}$, $\hat p_a=N_a^{-1}\sum_no_{na}y_{na}$ and
    $N_{ab}=\sum_no_{na}o_{nb}$ (respondents who answered every item of $a\cup b$; the matrix
    $O^\top O$). If the responses are independent of the missingness pattern (MCAR), then
    conditionally on the patterns $\hat p$ is unbiased for $\pi$ and
    $$\operatorname{Cov}(\hat p_a,\hat p_b)=\frac{N_{ab}}{N_aN_b}\,\Xi_{ab},\qquad
      \Gamma_{ab}=N\,\frac{N_{ab}}{N_aN_b}\,\Xi_{ab}$$
    for $\sqrt N(\hat p-\pi)$, exactly (each respondent contributes $\Xi_{ab}$ to the
    covariance of the two sums when it observes both margins). The FIML estimate $\hat\eta$ is
    consistent under MAR $\supseteq$ MCAR, so by the property above
    $M_2=N e^\top\Delta_c(\Delta_c^\top\Gamma\Delta_c)^{-1}\Delta_c^\top e$ is asymptotically
    $\chi^2_{s-q}$ under MCAR; with complete data $N_{ab}=N_a=N$ and §16.3 is recovered. Under
    MAR that is not MCAR the available-case margins are biased for $\pi$ and the statistic has
    no reference distribution; this is documented, not detected. Alternatives considered: the
    multiple imputation of earlier mirt versions (imputed $\theta$ and responses, statistics
    averaged over draws) has no $\chi^2$ reference for the averaged statistic and adds Monte
    Carlo error; mirt 1.46.1's `N.ratio` weighting above is not the inverse covariance of the
    residuals; model-based expected margins (replacing missing responses by their conditional
    expectations) leave the covariance of the resulting $\hat p$ dependent on $\hat\eta$. A margin
    nobody observed ($N_a=0$, e.g. two items never given together in a booklet design) has no
    estimate, and `m2()` raises `SpecificationError` naming the pair: limited-information fit
    on second-order margins is not defined for such designs. SRMSR follows mirt: the
    pairwise-available covariance $\sum_no_{n,ij}y_{ni}y_{nj}/N_{ij}-\hat p_i\hat p_j$ (with
    $\hat p_i$ the item's available-case mean) converted to a correlation. The null model is
    the independence model with intercepts at the available-case means (its FIML estimate),
    and its $M_2$ uses the same $\Gamma$ (closed form no longer applies). RMSEA uses the number
    of rows $N$.
    **MAP fits.** Priors leave $\hat\eta$ $\sqrt N$-consistent (the log-prior is $O(1)$ against an
    $O(N)$ log-likelihood), so by the property above $M_2$ keeps its $\chi^2_{s-q}$ reference;
    flexpyirt evaluates it at the MAP estimates with $q=$ `n_params` (every free parameter,
    whether or not it has a prior), exactly what mirt does. In finite samples a strong prior
    that keeps a weakly identified parameter (for example a 3PL lower asymptote) near its prior
    mean removes less than one degree of freedom, so the test is then conservative; and a
    parameter identified only by its prior makes $\Delta$ rank-deficient, which gives the
    `NaN` + warning of §16.3. The docstring says both.
    **Still not supported** (`SpecificationError`): `EmpiricalHistogram` and `NormalMixture`
    latent distributions, multiple-group `DavidianCurve` models, bifactor / two-tier models,
    weighted fits (§17.4).
    **Validation (V42)**: mirt 1.46.1 `M2()` at mirt's estimates for two-group 2PL with unequal
    groups and anchor items, two-group GRM, two-group two-factor 2PL, single-group and
    two-group 2PL with MCAR missing values, and 2PL MAP fits; `_mirt_convention=True`
    reproduces mirt's raw output, and the default is compared with an independent R computation
    of the formulas above (see VALIDATION.md).

24. **One repository per package (decided 2026-11-27; replaces the layout, versioning and release
    parts of §16.18)**: flexpyctt and flexpyequate move out of this repository into
    `takuizum/flexpyctt` and `takuizum/flexpyequate` (history of `packages/<name>/` kept). The
    scope of each package, the two polychoric implementations, `rng`, the shared error classes
    and the re-exports of §16.18 are unchanged.

    **Why.** With one tag for three distributions, every release gave all three a new version
    whether they changed or not, none could be released on its own, and flexpyequate (which
    shares no code with the other two) lived in a repository named after another package. The
    cost of splitting is that a change spanning flexpyctt and flexpyirt takes two steps; it is
    accepted because such changes are rare now that the tools have moved.

    **Versions.** Each repository tags its own `vX.Y.Z` (hatch-vcs); flexpyctt and flexpyequate
    start at 0.1.0. The versions are unrelated: nothing requires flexpyirt 0.3 to go with
    flexpyctt 0.3.

    **The dependency.** flexpyirt requires a range, `flexpyctt>=0.X,<0.(X+1)` (now
    `>=0.1,<0.2`), not an exact version: an exact pin would force a flexpyirt release for every
    flexpyctt bug fix and would conflict with a user's own requirement on flexpyctt. Before 1.0
    the minor version of flexpyctt is the unit of compatibility:
    - a flexpyctt patch release (`0.x.Y`) keeps the public API, and also the four private helpers
      that `flexpyirt.dimensionality` imports (`_coerce_ordinal`, `_observed_correlation`,
      `_eigenvalues`, `_descending_eigenvalues`); a test in flexpyctt checks their signatures.
      This closes the open point of §16.18 (pinning the same version with a build hook);
    - a flexpyctt minor release may change them. flexpyirt then raises the range in a PR of its
      own, and its CI is the check that the two fit;
    - `tests/test_packaging.py` checks that the range has a lower bound and stays inside one
      minor version, and that flexpyctt is not taken from a workspace or a local path.

    **Re-exports.** `flexpyirt/__init__.py` lists the re-exported names explicitly, so a name
    added in a flexpyctt patch release does not appear in flexpyirt until it is added here.
    `tests/test_m5_ensemble.py` fails when flexpyctt exports a name that flexpyirt does not
    re-export, which is the reminder to add it.

    **Order of work for a change in both.** Change flexpyctt in its repository and release it;
    then change flexpyirt (and raise the range if the minor version went up). A flexpyirt release
    needs a flexpyctt version inside the range on PyPI: the `publish` job installs the built
    wheel with flexpyctt from PyPI.

    **Reference values.** flexpyirt's tests use two of flexpyctt's reference files (V24 for the
    JAX polychoric tests and the two-version agreement test, V27 for one ensemble test). They are
    copied unchanged to `tests/reference/flexpyctt/v24_polychoric.json` and `v27_detect.json`; the scripts
    that write them are in flexpyctt's `validation/r/`. Copy them again when they are regenerated
    there. In this document and in VALIDATION.md, a path that starts with `flexpyctt/` is a path
    in the flexpyctt repository.

    **Documents.** The design notes of flexpyctt's tools stay in this document (§16.8–§16.16,
    §16.18, §16.20, §16.21) and their validation cases in VALIDATION.md, and flexpyctt's
    repository links here; moving them is postponed. flexpyequate has its own design document
    in its repository. On the documentation site flexpyctt's tools stay in the user guide and
    the API reference (flexpyirt re-exports them); the page for flexpyequate is removed, and the
    home page links to its repository.

    **CI and release here.** No workspace: `uv build` builds flexpyirt only, the release workflow
    has one `pypi` job, and the CI jobs that tested the other two without JAX moved to their
    repositories. flexpyequate stays in the `dev` dependency group (`flexpyequate>=0.1,<0.2`,
    from PyPI), because `tests/test_equating.py` compares the equipercentile kernel with it;
    it is not a runtime dependency.

## 17. Implementation decisions (M7)

1. **Bifactor models with Gibbons–Hedeker dimension reduction (`Model.with_bifactor`, V32)**:
   **Specification.** `Model.with_bifactor(*, specific, general="general")` declares one general
   factor on which every item loads and $S\ge1$ specific factors (`specific: Mapping[str,
   ItemSelector]`); each item loads on at most one specific factor, items listed under none load
   on the general factor only, and every specific factor needs at least two items (a one-item
   specific slope is not identified). The model reuses the confirmatory representation of §15.29:
   `factors = ((general, all items), (s_1, items), …)`, $D=1+S$ slopes `a1..aD` with `a1` general
   and unloaded slopes fixed at 0, matching the column order of mirt's `bfactor()`; a new field
   `Model.bifactor = True` selects the reduced E-step. Factors are orthogonal: the latent rows are
   `mean_1..D` (fixed 0) and `var_1..D` (fixed 1) with no `cor_ij` rows
   (`Model.latent_parameter_names()`). Item kernels accept up to `MAX_KERNEL_FACTORS = 32`
   dimensions (product grids stay limited to 3 by `with_factors()` and `Quadrature.grid()`).
   Not supported yet, each with `SpecificationError`: multiple groups, `with_latent()`, combining
   with `with_factors()`, Rasch / PCM items (the general and specific variances would have to be
   free) and free latent parameters in an edited table. Two-tier models (correlated primary
   factors) are item 2 below.
   **Likelihood** (Gibbons & Hedeker, 1992; Cai, 2010). Item $j$ depends on
   $(\theta_0,\theta_{s(j)})$ only, and the factors are independent, so with one-dimensional
   grids $\{\theta_g\}_{g=1}^{n}$ (general) and $\{\theta_t\}_{t=1}^{n}$ (each specific),
   $$L_p=\sum_g w^{(0)}_g\,\prod_{s=1}^{S}\Big[\sum_t w^{(s)}_t\prod_{j\in s}P_j(y_{pj}\mid\theta_g,\theta_t)\Big]
   \prod_{j\in s_\varnothing}P_j(y_{pj}\mid\theta_g),$$
   where $s_\varnothing$ is the set of general-only items and
   $w^{(m)}=\mathrm{softmax}(b+\log\phi(\cdot;\mu_m,\sigma^2_m))$ is the factor's normalized
   one-dimensional weight (§4.4, §15.44). The cost is $O(S\,n^2)$ per pattern instead of
   $O(n^{1+S})$. **Implementation.** General-only items are put in an extra block
   $s_\varnothing$ whose "specific" weights are uniform, so every block is handled alike (their
   log-probabilities do not depend on $\theta_t$, so the inner sum is exact). Block $s$ has an
   embedded node set of shape $(n^2, D)$ — column 0 is $\theta_g$, column $s$ is $\theta_t$, the
   rest 0 — so the existing kernels evaluate $P_j$ unchanged (the slopes on other specific factors
   are fixed at 0). With $\ell^{(s)}_{pgt}=\sum_{j\in s}\log P_j$,
   $L^{(s)}_{pg}=\operatorname{logsumexp}_t(\ell^{(s)}_{pgt}+\log w^{(s)}_t)$ and
   $\lambda_{pg}=\log w^{(0)}_g+\sum_s L^{(s)}_{pg}$, the marginal log-likelihood is
   $\sum_pc_p\operatorname{logsumexp}_g\lambda_{pg}$ and the posterior of block $s$ is
   $h^{(s)}_{pgt}=\exp(\lambda_{pg}-\log L_p)\exp(\ell^{(s)}_{pgt}+\log w^{(s)}_t-L^{(s)}_{pg})$.
   The expected counts of item $j$ are on its block's $(n^2)$ grid,
   $\bar r_{jkgt}=\sum_pc_ph^{(s(j))}_{pgt}\mathbf 1[y_{pj}=k]$, so the per-item Newton M-step
   (§4.3), across-item equalities (§16.1) and priors are unchanged apart from taking the item's
   block nodes; $\bar n$ stores the general marginal and each block's specific marginal
   $(G, 1+B, n)$ for a later latent M-step. `CompiledProblem.reduction` (a `BifactorBlock`:
   block nodes, the 1-D nodes and base weights, and each family's block index) switches the
   Core functions; `CompiledProblem.nodes` holds the general grid embedded in $D$ columns.
   **Standard errors.** `"hessian"` differentiates the reduced marginal log-likelihood
   (chunked forward-over-reverse, §15.20). `"oakes"` uses Oakes' identity in the form
   $I(\hat\eta)=-\,\mathrm d F(\eta)/\mathrm d\eta$ with
   $F(\eta)=\nabla_\eta Q_{\mathrm{post}}(\eta\mid\eta')\big|_{\eta'=\eta}$ (the complete-data
   gradient with the E-step weights computed at the same point; differentiating it in both
   arguments gives exactly the two Oakes terms), by chunked forward-mode autodiff through the
   reduced E-step. The structured cross term of §15.20 assumes one joint posterior over all
   nodes and is not used. **Scoring.** EAP is exact on the factorized posterior:
   $\hat\theta_{p0}=\sum_g\theta_g h_{pg}$ and
   $\hat\theta_{ps}=\sum_{g,t}\theta_t h^{(s)}_{pgt}$ (posterior SDs likewise). MAP, ML, WLE and
   PV, item fit, $M_2$, local dependence, JSI, DIF and linking raise `SpecificationError` for
   bifactor models (they are multidimensional). **Quadrature.** One `Quadrature` gives the
   one-dimensional grid for every factor; the default is mirt's
   `select_quadpts(2) = 31` points on $[-6, 6]$ (mirt's `bfactor()` uses
   `select_quadpts(nfact - nspec + 1)`). **JSON** (schema version 5): the model stores
   `"bifactor": true|false`; versions 1–4 must not contain it and load as `false`. On load a
   bifactor structure is rebuilt through `with_bifactor()`.

2. **Two-tier models (`Model.with_two_tier`, V33)** (Cai, 2010):
   **Specification.** `Model.with_two_tier(*, primary, specific, correlated=True)` declares
   $P\in\{1,2\}$ primary factors and $S\ge1$ specific factors (both `Mapping[str, ItemSelector]`).
   Every item loads on at least one primary factor (cross-loadings among primary factors are
   allowed) and on at most one specific factor; every specific factor needs at least two items.
   Factors are ordered primary first: slopes `a1..aP` are primary and `a(P+1)..aD` specific
   ($D=P+S$), the column order of mirt's `bfactor(data, model, model2 = )`. Primary factors may
   correlate with each other (`correlated=True` frees `cor_ij`, $i<j\le P$); specific factors are
   orthogonal to everything. The latent rows are `mean_1..D` (fixed 0), `var_1..D` (fixed 1) and
   `cor_ij` for primary pairs only. `Model.n_primary_factors = P` marks the structure (0 for every
   other model); a bifactor model is the case $P=1$ with the primary factor on every item, and
   both share the reduced Core path (`Model.is_reduced`, `Model.n_primary`). $P=3$ would need a
   four-dimensional grid per block and is rejected, in line with the three-dimension limit of
   product grids (§4.9). Not supported, each with `SpecificationError`: as for bifactor models
   (multiple groups, `with_latent()`, Rasch / PCM items, free means or variances), and combining
   with `with_factors()` / `with_bifactor()`.
   **Likelihood.** With the primary product grid $\{\boldsymbol\theta_g\}_{g=1}^{n^P}$ and its
   normalized multivariate-normal weights $w^{(0)}_g(\boldsymbol\rho)$ (§15.30),
   $$L_p=\sum_g w^{(0)}_g\,\prod_{s=1}^{S}\Big[\sum_t w^{(s)}_t\prod_{j\in s}P_j(y_{pj}\mid\boldsymbol\theta_g,\theta_t)\Big]
   \prod_{j\in s_\varnothing}P_j(y_{pj}\mid\boldsymbol\theta_g),$$
   which is §17.1 with the general node replaced by the primary node: cost $O(S\,n^{P+1})$ per
   pattern instead of $O(n^{P+S})$. **Implementation.** `BifactorBlock` is generalized: it holds
   the primary grid `(Q_p, P)` ($Q_p=n^P$) with its base weights, block nodes of shape
   $(B, Q_p n, D)$ (columns $0..P-1$ the primary node, column $P+b$ the specific node, primary
   index major) and the static `n_primary`. All formulas of §17.1 hold with $g$ running over the
   $Q_p$ primary nodes. $\bar n$ is stored flat, `(G, Q_p + B n)`: the primary marginal followed
   by each block's specific marginal. **Latent M-step.** The latent part of $Q$ is
   $Q^{\text{lat}}=\sum_g\bar n^{(0)}_g\log w^{(0)}_g(\boldsymbol\rho)+\sum_{s,t}\bar n^{(s)}_t\log w^{(s)}_t$;
   only the first term depends on the free correlations, and it is maximized by the masked Newton
   of §4.4 over the primary multivariate-normal parameters (exact for the discretized likelihood,
   so EM stays monotone; a non-positive-definite candidate gives NaN and is rejected by the line
   search). Bifactor models have no free latent parameter and skip it as before.
   **Identification.** The rules of §15.31 apply to the primary factors (a free correlation needs
   fixed slopes; the zero slopes of the factor structure count). **Standard errors, scoring.**
   As §17.1: Hessian of the reduced log-likelihood, Oakes as the total derivative through the
   E-step (now including the correlation), EAP from the factorized posterior with the primary
   means taken on the primary grid. **Quadrature.** `Quadrature.default(P+1)` points per
   dimension: 31 for $P=1$, 15 for $P=2$ (mirt's `select_quadpts(nfact - nspec + 1)`).
   **JSON** (schema version 6): the model stores `"n_primary_factors"` (0 unless two-tier);
   versions 1–5 must not contain it. On load a two-tier structure is rebuilt through
   `with_two_tier()`.
3. **Local dependence for bifactor and two-tier models (`FitResult.local_dependence`, V38,
   issue #81)**: the statistics of §16.11 on the reduced grids of items 1–2, without the
   $n^{P+S}$ product grid. Notation as above: primary nodes $\boldsymbol\theta_g$ with weights
   $w^{(0)}_g$ ($g=1..Q_p$, $Q_p=n^P$), specific nodes $\theta_t$ with weights $w^{(s)}_t$,
   $s(j)$ the specific factor of item $j$ ($\varnothing$ when it has none) and
   $P_{ja}(\boldsymbol\theta_g,\theta_t)$ its category probabilities on its block grid.
   **Expected tables.** Integrating the specific factor out of one item gives its trace given
   the primary factors,
   $$\bar P_{ja}(\boldsymbol\theta_g)=\sum_t w^{(s(j))}_t\,P_{ja}(\boldsymbol\theta_g,\theta_t)$$
   ($\bar P_{ja}=P_{ja}$ when $s(j)=\varnothing$; the uniform weights of that block give the same
   value). Because the specific factors are independent of each other and of the primary
   factors, the model-implied table of a pair answered by $n_{ij}$ respondents is
   $$E_{ab}=n_{ij}\sum_g w^{(0)}_g\sum_t w^{(s)}_t\,P_{ia}(\boldsymbol\theta_g,\theta_t)
   P_{jb}(\boldsymbol\theta_g,\theta_t)\qquad s(i)=s(j)=s,$$
   $$E_{ab}=n_{ij}\sum_g w^{(0)}_g\,\bar P_{ia}(\boldsymbol\theta_g)\,\bar P_{jb}(\boldsymbol\theta_g)
   \qquad s(i)\ne s(j)\ \text{or}\ s(i)=s(j)=\varnothing .$$
   Both are exact rearrangements of the sum over the full product grid with weights
   $w^{(0)}_g\prod_s w^{(s)}_{t_s}$ (every factor that neither item loads on sums to 1), so they
   equal mirt's `residuals(type = "LD")`, which builds that grid (`thetaComb(theta, nfact)`),
   when the one-dimensional grid is the same. Cost: $O(Q_p n K_iK_j)$ per pair instead of
   $O(n^{P+S}K_iK_j)$. $X^2$, $G^2$, $df=(K_i-1)(K_j-1)$, the $p$-values and the signed
   Cramér's $V$ follow from $O_{ab}$ and $E_{ab}$ as in §16.11.
   **Q3.** $d_{ij}=x_{ij}-E(X_j\mid\hat{\boldsymbol\theta}_i)$ at the vector EAP of all $D=P+S$
   factors from the factorized posterior (`_bifactor_eap`); the kernels are evaluated at the
   `(N, D)` scores with the reduction switched off (`CompiledProblem.reduction = None`, so
   `_family_nodes` returns the scores). Q3\* as in §16.11.
   **Implementation.** `_local_dependence.ld_pairs` is split: `ld_pairs_from_tables(y,
   pair_probabilities)` computes the statistics from a function returning the $K_i\times K_j$
   joint probabilities of a pair, `ld_pairs` (shared grid) and `reduced_pair_probabilities`
   (the two formulas above, from the traces on the block grids `(Q_p, n, K_j)`, the weights of
   `_bifactor_log_weights` and each item's block) supply that function. Bifactor and two-tier
   models are single-group (items 1–2), so there is no group loop here beyond the existing one.
   **Interpretation.** A pair within a specific factor is tested for dependence *beyond* that
   factor; a testlet left out of `specific` shows up as large positive $V$ within it (unit-tested on simulated data).
   **Validation (V38).** mirt's estimates of a bifactor 2PL (with general-only items and
   missing data), a bifactor GRM and a two-tier 2PL (`bfactor(model2 = )`) are put into a
   `FitResult` with every parameter fixed and compared with `residuals(type = "LD" / "LDG2" /
   "Q3")` (Q3 with `Theta = fscores(m, quadpts = <the fit's>)`, because `fscores()` otherwise
   uses its own grid); a unit test compares the factorized tables with the brute-force product
   grid. **Not extended.** `jsi()` keeps raising `SpecificationError` for these models: the
   statistic is defined for one slope per item (Edwards, Houts & Cai, 2018), an item here has a
   primary and a specific slope, and mirt's `type = "JSI"` requires one factor, so both the
   definition and a reference are missing.

4. **Sampling (survey) weights (`fit(..., weights=...)`, V34)**: with a weight $w_i\ge0$ for every
   respondent, `fit()` maximizes the weighted marginal log-likelihood (pseudo-likelihood; Skinner,
   1989; mirt's `survey.weights`)
   $$\ell_w(\eta)=\sum_i w_i\log L_i(\eta)=\sum_p c_p\log L_p(\eta),\qquad c_p=\sum_{i:\,y_i=p}w_i,$$
   plus the log prior for a MAP fit. This is the likelihood of §4.2 with the pattern frequency
   $c_p$ replaced by the sum of the weights of the rows with pattern $p$ (within a group), so the
   Core is unchanged: the E-step, the item and latent M-steps, SQUAREM, multiple groups, FIPC,
   every latent distribution and both information matrices already work on real-valued
   `CompiledProblem.counts`. Only the Compile layer changes (`coerce_weights`,
   `compress_patterns(..., weights)` summing with `np.bincount`).
   - **Argument**: `weights` is an array-like with one number per row of `data` (NumPy array,
     list, pandas / polars Series). A column name is **not** accepted: `data` must hold exactly
     the model's items, and `group=` is array-like only, so `weights=` follows the same rule.
     `SpecificationError` for a wrong length or shape, non-numeric, NaN / infinite or negative
     values, a zero total, and a declared group whose weights are all zero.
   - **No rescaling**: the weights are used as given (as in mirt). `loglik`, `log_posterior`, AIC
     and the information matrix are therefore on the scale of the weights; weights that sum to
     the number of rows keep them comparable with an unweighted fit. With priors, the prior has
     the same absolute weight whatever the scale of $w$, so rescaling the weights changes a MAP
     estimate (not an MLE).
   - **Zero weights**: rows with $w_i=0$ are dropped right after validation, before the starting
     values and the estimability check (§14.5), so they affect nothing (a category seen only in
     zero-weight rows is "never observed").
   - **Integer weights** are frequency weights: the fit equals the fit to the data with row $i$
     repeated $w_i$ times (same likelihood, estimates and information; unit-tested).
   - **What "N" means**: `n_obs` is the number of rows with a positive weight. BIC uses
     $\log W$ with $W=\sum_i w_i$ in place of $\log N$, as mirt does (its $N$ is the sum of the
     pattern frequencies), so BIC agrees with mirt and with row replication for integer weights.
     AIC does not use $N$. $W$ is kept as the private `FitResult._sum_weights` (`None` for an
     unweighted fit) and shown in `summary()`. The **starting values** (§4.10) are computed
     without weights from the rows with a positive weight: they change the path of EM, not its
     limit, and an unweighted moment is a good enough start. The identification checks and the
     Heywood / degenerate-solution warnings do not use $N$.
   - **Standard errors**: `se="oakes"` (default) and `se="hessian"` invert
     $-\nabla^2\ell_w(\hat\eta)$. For frequency weights that is the usual information. For
     sampling weights it is **not** a valid variance estimate: the design-consistent variance is
     the sandwich $H^{-1}\big(\sum_i w_i^2 s_i s_i^\top\big)H^{-1}$ (Binder, 1983), which comes
     with the robust standard errors (item 6). Decision: the default stays `"oakes"` (the values
     equal mirt's `SE.type = "Oakes"` with `survey.weights`, so they can be validated), and `fit()`
     emits a `FlexpyirtWarning` whenever `weights` is given and `se` is `"oakes"`, `"hessian"` or
     `"crossprod"`; the warning names `se="sandwich"` as the design-based alternative (item 6).
   - **After the fit**: scoring (`score()`) depends only on the estimates and is unchanged.
     `link()` uses the estimates and `vcov` (its SEs inherit the caveat above). The data-based
     diagnostics have no validated weighted form (their statistics and reference distributions
     assume unweighted counts), so `item_fit()`, `m2()`, `local_dependence()`, `jsi()`, `dif()`,
     `model_parallel_analysis()` and `dimensionality(fit=...)` raise `SpecificationError` for a
     weighted fit (`FitResult._require_unweighted`).
   - **JSON**: `schema_version` 7 adds the optional top-level key `sum_weights` (present only
     for a weighted fit; a positive finite number); it is rejected in files of version $\le6$,
     which still load. The weights themselves are not stored (like the data).
   - **Validation (V34)**: `validation/r/m7_weights_reference.R` (mirt 1.48, `survey.weights`,
     TOL $=10^{-10}$, Oakes): a 2PL with non-integer weights that do not sum to $N$ and five zero
     weights, the Science GRM with weights normalized to $N$, two groups with anchors, a free
     focal latent distribution and 5% missing responses, FIPC with a free latent
     distribution, and a bifactor 2PL (`bfactor()`). Bifactor and two-tier models (§17.1, §17.2)
     need nothing special: their dimension-reduced E-step also works on `counts`; the two-tier
     case is checked against row replication only. Estimates, log-likelihood, AIC, BIC and Oakes SEs agree within the common
     tolerances (VALIDATION.md V34).
5. **Robust standard errors: cross-product and sandwich (`fit(se="crossprod" | "sandwich")`, V35, #85)**:
   **Definitions.** With $L_p(\eta)=\sum_q w_{g(p)q}(\eta)\prod_j P_j(y_{pj}\mid\theta_q;\eta)$ the
   marginal likelihood of response pattern $p$ (count $c_p$) and $s_p=\nabla_\eta\log L_p(\hat\eta)$
   its score,
   $$B=\sum_p c_p\,s_p s_p^\top,\qquad B_{\mathrm{post}}=B-\nabla^2_\eta\log\pi(\hat\eta),$$
   $$V_{\mathrm{crossprod}}=B_{\mathrm{post}}^{-1},\qquad
     V_{\mathrm{sandwich}}=H^{-1}B_{\mathrm{post}}H^{-1},$$
   where $H=-\nabla^2_\eta[\ell(\hat\eta)+\log\pi(\hat\eta)]$ is the observed (posterior)
   information of §4.8, computed by Oakes' identity (`information_oakes`; identical to
   `information_hessian`). $B$ is the outer-product-of-gradients estimate of the Fisher information
   (Berndt, Hall, Hall & Hausman, 1974); the sandwich is Huber's (1967) and White's (1982)
   covariance of a maximum-likelihood estimator under misspecification. The unit of independence
   is the person: a pattern observed $c_p$ times contributes $c_p$ identical terms, so compressed
   and row-replicated data give the same $B$. The scores are not centred ($\sum_p c_p s_p=0$ at
   the ML estimate, and mirt does not centre them either).
   **Priors.** The prior curvature is added to the cross-product, exactly as it is added to $H$
   (§15.17): both `oakes`/`hessian` and `crossprod` then estimate the same posterior information,
   and when the information equality $B=-\nabla^2\ell$ holds the sandwich reduces to
   $H^{-1}$, the existing MAP covariance. This is mirt's convention (`SE.simple`:
   `info <- Igrad + Iprior` for `"crossprod"`, the same matrix as the middle of `"sandwich"`), so
   it can be validated. The alternative $H^{-1}BH^{-1}$ with the likelihood scores only (the
   repeated-sampling variance of a penalized estimator, which for a log-concave prior is smaller than $H^{-1}$ even under
   a correct model) is not offered. A fixed parameter is not in $\eta$; a parameter shared across
   groups or items (one `free_id`) has one score coordinate that sums its occurrences.
   **Multiple groups and latent parameters.** $\eta$ holds every group's item and latent
   parameters; a pattern of group $g$ has a zero score for the parameters that enter only other
   groups, so $B$ is the sum of the groups' cross-products without any special handling.
   **Empirical histogram.** As for `oakes`/`hessian` (§15.35), the histogram weights are held at
   their estimates: $s_p$ and $H$ are derivatives with respect to the free item parameters only,
   and `vcov` keeps its `n_free_items × n_free_items` shape. The sandwich is therefore robust for
   the item parameters given the estimated histogram, not for the estimation of the histogram.
   **Not covered.** `DavidianCurve` and `NormalMixture` latent distributions raise
   `SpecificationError` for both options: mirt computes neither for them ("No standard error
   method currently supported for Davidian curves"; "Only Oakes and complete SE.types current
   supported for mixture models"), so there is no reference to validate against (principle 1).
   Bifactor and two-tier models (§17.1, §17.2) also raise `SpecificationError`: their likelihood
   is evaluated by dimension reduction (`CompiledProblem.reduction`), while the score kernel
   below works on the node tables of the full grid. mirt does compute both types for `bfactor()`,
   so this is an implementation gap that can be validated later, not a missing reference.
   Sampling weights and the cluster-robust version are item 6.
   **Computation** (`_robust_se.py`, a separate Core module so that `_core.py` is untouched).
   `_fold_pattern_scores` evaluates the node tables $\log P_{gjqk}(\eta)$ and $\log w_{gq}(\eta)$
   once and linearizes them with `jax.vjp`. Patterns are processed in chunks by `lax.scan`: for a
   chunk, `vmap(jax.grad)` differentiates each pattern's log-likelihood with respect to the
   tables and the stored pullback maps each cotangent to $\eta$, giving a `(B, n_free)` block of
   scores from which $\sum c_p s_p s_p^\top$ is accumulated. Neither `(P, n_free)` nor
   `(P, G, J, Q, K)` is materialized; the chunk size is the largest (at most 1024) that keeps
   `chunk × G × Q × (1 + Σ_f J_f K_f (P_f + 1))` under $8\times10^6$ elements (about 64 MB), and
   chunks of zero-count padding rows are skipped with `lax.cond`. Every derivative is autodiff;
   `score_sum` ($\sum_p c_p s_p$) is checked against `jax.grad(marginal_loglik)` and the
   cross-product against a dense `jax.jacfwd` of the pattern log-likelihoods in the unit tests.
   `fit()` builds the sandwich in NumPy from the inverse of $H$ (the existing positive-definiteness
   check: `FlexpyirtWarning` and NaN standard errors if $H$ is not positive definite; for
   `crossprod` the same check applies to $B_{\mathrm{post}}$, which is singular when there are
   fewer distinct patterns than free parameters).
   **Consumers of `vcov`.** `FitResult.vcov`, the `se` column, the delta-method display SEs
   (`b_se`, `bk_se`, latent `sd`/`cor` SEs; §15.18) and `link()` (§16.5) read `vcov` and so use the
   chosen covariance without change. `FitResult.jsi(method="one_step")` uses `vcov` as the inverse
   observed information of a Newton step, which $V_{\mathrm{sandwich}}$ and
   $V_{\mathrm{crossprod}}$ are not; it raises `SpecificationError` unless the fit used `oakes` or
   `hessian`.
   **Option names and JSON.** `se` takes `"sandwich"` and `"crossprod"` (mirt's `SE.type`
   spellings; `"cross_product"` would be the spelling of the naming rule of §15.46 and is left to
   the owner). `se_method` in the JSON file stores the same strings; the two values are additions,
   no stored name changes, and `schema_version` stays as it is (as when new latent kinds were
   added in §16.4 and §16.7). mirt's `"sandwich.Louis"` is the same estimator with $H$ computed by
   Louis' (1982) formula instead of Oakes'; both formulas give the observed information, so it is
   not a separate option here.

6. **Robust standard errors under sampling weights and clusters (`fit(..., weights=, cluster=, se="sandwich")`, V57, #84, #85)**:
   **Weighted sandwich.** The weighted estimator of item 4 solves $\sum_i w_i s_i(\hat\eta)=0$
   (plus the prior gradient for MAP), with $s_i=\nabla_\eta\log L_i(\hat\eta)$ the score of row
   $i$. Its design-based (linearization) covariance is (Binder, 1983; Skinner, 1989; Asparouhov,
   2005)
   $$V_{\mathrm{sandwich}}=H_w^{-1}B_wH_w^{-1},\qquad
     B_w=\sum_i w_i^2\,s_i s_i^\top-\nabla^2_\eta\log\pi(\hat\eta),\qquad
     H_w=-\nabla^2_\eta\big[\ell_w(\hat\eta)+\log\pi(\hat\eta)\big].$$
   Rows with the same pattern in the same group have the same score, so
   $B_w=\sum_p c^{(2)}_p s_p s_p^\top-\nabla^2\log\pi$ with $c^{(2)}_p=\sum_{i:\,y_i=p}w_i^2$; with
   $w_i=1$ this is item 5. Without priors, $V_{\mathrm{sandwich}}$ does not change when every
   weight is multiplied by the same constant ($H_w$ scales by $k$, $B_w$ by $k^2$). The prior
   curvature is added to the meat as in item 5 (mirt's convention; with priors the matrix depends
   on the scale of the weights, like the MAP estimate itself, item 4).
   **This is not mirt's weighted sandwich.** mirt 1.48 builds the middle matrix of
   `SE.type = "sandwich"` (and `"crossprod"`) from the pattern frequencies `Data$Freq`, which
   with `survey.weights` are the sums of the weights (`computeInfo` in `dpars.cpp`:
   `add2outer(Igrad, grad, rs(g, pat))`), so its meat is $\sum_i w_i s_i s_i^\top$. That is the
   frequency-weight cross-product: multiplying the weights by $k$ divides mirt's sandwich by $k$,
   so it is not a design-based variance (mirt's own `estfun.AllModelClass` documentation says the
   scores must be weighted by $\sqrt{w_i}$ to reproduce that matrix). Decision: follow Binder
   (1983). The validation (V57) therefore checks (a) the weighted scores against mirt's
   `SE.type = "crossprod"` with `survey.weights` (same definition as flexpyirt's, below) and
   (b) the design-based sandwich and the cluster version against an R computation from mirt's
   weighted Oakes covariance (the bread) and per-person scores by `numDeriv`.
   **`crossprod` with weights.** `se="crossprod"` inverts
   $\sum_p c_p s_p s_p^\top-\nabla^2\log\pi$ with $c_p=\sum_{i:\,y_i=p}w_i$ (mirt's definition).
   Through the information equality this estimates the information of the weighted
   log-likelihood, so, like `oakes` and `hessian`, it is a valid variance only for frequency
   weights; it is allowed and emits the same `FlexpyirtWarning`. ($B_w^{-1}$ would be no variance
   at all: it is the inverse of the variance of the weighted score, not the variance of
   $\hat\eta$.)
   **Cluster-robust sandwich.** `fit(..., cluster=labels)` takes one label per row (array-like
   like `group=`: NumPy array, list, pandas / polars Series; any hashable labels; a column name
   is not accepted; missing labels (NaN / None) raise `SpecificationError`). The scores of the
   rows of a cluster (for example a school) are summed before the outer product:
   $$u_c=\sum_{i\in c}w_i s_i,\qquad
     B_{\mathrm{cl}}=\frac{G}{G-1}\sum_{c=1}^{G}u_c u_c^\top-\nabla^2_\eta\log\pi(\hat\eta),\qquad
     V=H_w^{-1}B_{\mathrm{cl}}H_w^{-1},$$
   with $w_i=1$ when `weights` is not given. This treats the clusters as independent primary
   sampling units drawn with replacement in a single stratum (Binder, 1983; Williams, 2000).
   **Small-sample factor: decision $G/(G-1)$.** It is the factor of the with-replacement
   first-stage variance estimator (Binder, 1983; R's survey package, `svyrecvar`: $n_h/(n_h-1)$
   within a stratum, one stratum here) and of Stata's `vce(cluster)` for maximum-likelihood
   estimators; without it the estimator is biased downward by $(G-1)/G$, which matters with tens
   of schools. The extra $(N-1)/(N-K)$ of the linear-regression version is not used. The scores
   are not centred (as in item 5; $\sum_c u_c=0$ at an MLE). Consequence: with one row per
   cluster, $B_{\mathrm{cl}}$ is $N/(N-1)$ times the unclustered $B_w$ (unit-tested), not equal to
   it; the unclustered sandwich keeps no factor so that it stays equal to mirt's unweighted one.
   **Scope.** Strata, finite-population corrections and multi-stage designs are not supported
   (one stage, clusters as PSUs). Clusters may span the groups of a multiple-group model. At
   least two clusters (after dropping zero-weight rows with their labels) are required, else
   `SpecificationError`. `cluster=` without `weights=` is allowed. `cluster=` with any `se` other
   than `"sandwich"` raises `SpecificationError` (no other method uses it). Not covered, as in
   item 5: `DavidianCurve`, `NormalMixture`, bifactor and two-tier models (`SpecificationError`).
   **Computation** (`_robust_se.py`). Unclustered: the meat uses a copy of `CompiledProblem` with
   `counts` replaced by $c^{(2)}_p$ on the same patterns, and `score_cross_product`. Clustered:
   the rows are compressed by (group, cluster, pattern) (`compress_patterns(..., cluster=)`, at
   most $N$ rows, padded as in §4.1), the counts are weight sums, and
   `score_cluster_cross_product` accumulates the `(G, n_free)` matrix $U$ with
   `U.at[cluster].add(c_p s_p)` in the same pattern-chunked fold (`_fold_pattern_scores` now
   chunks a per-pattern integer label with the counts), then returns $U^\top U$. A pattern that
   occurs in several clusters is differentiated once per cluster, so the cost is at most that of
   an uncompressed data set. $G$ is static for `jit` (one compilation per number of clusters).
   **Default with weights: decision, still `"oakes"`.** `se` does not change its default
   depending on another argument; `"oakes"` is the method validated against mirt's
   `survey.weights` and the only one available for every latent distribution and bifactor /
   two-tier models. The `FlexpyirtWarning` of item 4 now recommends `se="sandwich"` and is not
   emitted for it. Left to the owner: default to `"sandwich"` whenever `weights` or `cluster` is
   given.
   **Result and JSON.** `FitResult._n_clusters` is the number of clusters (`None` without
   `cluster=`), shown by `summary()` as `sandwich (cluster-robust, G clusters)`.
   `schema_version` 9 adds the optional top-level key `n_clusters` (an integer $\ge2$, only with
   `se_method` `"sandwich"`); it is rejected in files of version $\le8$, which still load. The
   labels themselves are not stored (like the weights and the data). The consumers of `vcov`
   (the `se` column, the delta-method display SEs, `link()`) use the chosen covariance without
   change (item 5); `jsi(method="one_step")` still requires `oakes` or `hessian`.
   **Validation (V57)**: `validation/r/v57_weighted_cluster_se_reference.R` (GitHub mirt 1.48.5,
   `survey.weights`, TOL $=10^{-10}$): a 2PL with simulated schools (a school effect in $\theta$)
   and unequal weights, and a two-group 2PL with anchors and clusters that span the groups; the
   reference sandwiches are computed in R from mirt's Oakes covariance and per-person scores
   (`numDeriv::jacobian` of the per-person marginal log-likelihood on mirt's grid, checked
   against `estfun.AllModelClass`), summed by school for the cluster version.

## 18. Summed-score distribution kernel, information and reliability (M7)

1. **Lord–Wingersky recursion as a Poisson-binomial kernel (`summed_score_distribution`, V36)**:
   **What it computes.** Given $\theta$, the item scores $X_j\in\{0,\dots,K_j-1\}$ are independent
   with category probabilities $P_{jk}(\theta)$, and the kernel returns the distribution of the
   summed score $S=\sum_jX_j$, $s=0,\dots,S_{\max}=\sum_j(K_j-1)$. For dichotomous items $S$ is a
   sum of independent non-identical Bernoulli variables, i.e. the **Poisson-binomial
   distribution** with success probabilities $p_j(\theta)$ (González, Wiberg & von Davier, 2016);
   for polytomous items it is the distribution of a sum of independent categorical scores. The
   Lord–Wingersky (1984) recursion (Thissen, Pommerich, Billeaud & Williams, 1995, for
   polytomous items) is the direct convolution algorithm for this distribution, the same as the
   "RF" recursion of the Poisson-binomial literature (Hong, 2013).
   **API (Decision, owner 2026-11-25).** `summed_score_distribution(probs, *, method="recursion",
   log=False)`. `probs` has shape `(..., J, K)`: category probabilities of $J$ items, rows summing
   to 1, items with fewer than $K$ categories padded with zeros; leading axes (quadrature nodes,
   persons, ...) are batch axes. The result has shape `(..., J(K-1)+1)`; scores above an item
   set's own maximum have probability 0. The function is a pure JAX kernel: `jit`, `vmap` and
   autodiff work through every method, and values are not checked (a caller inside `jit` could
   not be checked). Only the shape and `method` are validated (`SpecificationError`).
   **Methods.**
   - `"recursion"` (default, exact): $L_j(s)=\sum_kL_{j-1}(s-k)P_{jk}$, $L_0(0)=1$, as a
     `lax.scan` over items on a fixed-length vector, $O(JKS_{\max})$ per batch element. Every
     term is non-negative, so there is no cancellation and the relative error of each
     probability is a few ulp times $J$, tails included. With `log=True` the recursion runs in
     log space, $\log L_j(s)=\operatorname{logsumexp}_k[\log L_{j-1}(s-k)+\log P_{jk}]$, which
     keeps tail probabilities below the smallest double ($\approx10^{-308}$) that the linear
     recursion underflows to 0 (long tests, extreme $\theta$).
   - `"fft"` (exact up to round-off): the DFT of the characteristic function (DFT-CF; Hong,
     2013), $\varphi(\omega_l)=\prod_j\sum_kP_{jk}e^{i\omega_lk}$ at
     $\omega_l=2\pi l/(S_{\max}+1)$ and $P(S=s)=\frac1{S_{\max}+1}\sum_l\varphi(\omega_l)e^{-i\omega_ls}$,
     one batched FFT. Its error is **absolute**, about $10^{-16}$, whatever the size of the
     probability: probabilities below $\sim10^{-15}$ have no correct digits (negative round-off
     is clipped at 0). It must not be used where tails matter (S-$X^2$, log-likelihoods of
     extreme scores); `log=True` only takes the logarithm of the result.
   - `"normal"`, `"refined_normal"`, `"poisson"` (approximations, dichotomous items only,
     $K=2$; other $K$ raise `SpecificationError` because only the dichotomous case has an R
     reference): with $\mu=\sum_jp_j$, $\sigma^2=\sum_jp_j(1-p_j)$ and
     $\gamma=\sigma^{-3}\sum_jp_j(1-p_j)(1-2p_j)$, the normal approximation with continuity
     correction is $P(S\le s)\approx\Phi(x)$, $x=(s+0.5-\mu)/\sigma$; the refined normal
     approximation is $P(S\le s)\approx G(x)$, $G(x)=\Phi(x)+\gamma(1-x^2)\phi(x)/6$ clipped to
     $[0,1]$ (Volkova, 1996; Hong, 2013); the probability of $s$ is the difference of
     consecutive values, so the result need not sum to exactly 1. The Poisson approximation is
     $P(S=s)\approx e^{-\mu}\mu^s/s!$ (Le Cam, 1960), with total-variation error at most
     $\sum_jp_j^2$; it is useful only when every $p_j$ is small. The approximations cost
     $O(J+S_{\max})$ and are meant as optional fast paths; nothing inside flexpyirt uses them.
   **Reuse.** `_itemfit.lord_wingersky` (S-$X^2$) now calls the same kernel; the V15 references
   are unchanged. Summed-score EAP and IRT observed-score equating will use it too (#82).
   **Measured accuracy and speed** are recorded in VALIDATION.md (V36).

2. **Information, expected score, reliability and summed-score EAP (#82, V37)**:
   **Scope.** Unidimensional models, single and multiple groups. The information matrix and
   directional information of multidimensional models are item 3; reliability and the
   summed-score functions raise `SpecificationError` for a multidimensional model.
   **API (Decision, owner 2026-11-25).** Module-level functions that take the fit first, like
   `link()` and `dimensionality()`:
   `item_information(res, theta, *, group=None)`, `test_information(res, theta, *, group=None)`,
   `expected_score(res, theta, *, group=None)`,
   `reliability(res, data=None, *, group=None, item_names=None)`, `summed_score_table(res)`, and
   `FitResult.score(data, method="EAPsum")`. The three functions of $\theta$ return a DataFrame
   indexed by `theta` (so `.plot()` draws the curves); in a multiple-group model `group=` (a
   group name) is required, because items may differ across groups. `reliability()` and
   `summed_score_table()` return one row (block) per group with a leading `group` column in a
   multiple-group model, as `item_fit()` does; in `reliability()` `group=` are the group labels
   of the rows of `data`, as in `score()`.
   **Item and test information.** For every item family, from JAX autodiff of the category
   log-probabilities of the item kernel (no hand-written derivatives, the same expression WLE
   scoring uses in §15.24):
   $$I_j(\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\theta)\Big(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\Big)^2,\qquad
   I(\theta)=\sum_jI_j(\theta),\qquad \mathrm{SE}(\theta)=I(\theta)^{-1/2}$$
   (Birnbaum, 1968; Samejima, 1969). mirt's `iteminfo()` evaluates the algebraically equal
   $\sum_k[(P'_{jk})^2/P_{jk}-P''_{jk}]$ ($\sum_kP''_{jk}=0$) from hand-written derivatives.
   `item_information` has one column per item; `test_information` has the columns
   `information` and `se` (the conditional standard error of measurement of an efficient
   estimator of $\theta$).
   **Expected score.** $E_j(\theta)=\sum_kk\,P_{jk}(\theta)$ per item and the test
   characteristic curve $T(\theta)=\sum_jE_j(\theta)$ (column `total`; an item named `total`
   raises `SpecificationError`). Categories are scored $0,\dots,K_j-1$, the coding of the
   data, which is mirt's `expected.item()` / `expected.test()` when the lowest observed
   category is 0.
   **Marginal reliability.** For group $g$ with latent distribution $N(\mu_g,\sigma_g^2)$ and
   test information $I_g$,
   $$\bar\rho_g=\int\frac{\sigma_g^2I_g(\theta)}{\sigma_g^2I_g(\theta)+1}\,\phi(\theta;\mu_g,\sigma_g^2)\,d\theta ,$$
   the expectation over the latent distribution of the conditional reliability
   $\sigma^2/(\sigma^2+I(\theta)^{-1})$ of a score with error variance $1/I(\theta)$ (Green,
   Bock, Humphreys, Linn & Reckase, 1984; Thissen & Wainer, 2001). mirt's `marginal_rxx()`
   computes $\int\frac{I(\theta)}{I(\theta)+\sigma^2}\phi(\theta;0,1)\,d\theta$: the same
   number when $\mu=0$, $\sigma^2=1$ (every single-group model with the default
   identification), but for other $(\mu,\sigma^2)$ it neither integrates over the fitted
   density nor is invariant under a linear change of the latent scale. flexpyirt's definition
   is invariant: with $\theta^*=(\theta-\mu)/\sigma$ the information is
   $I^*(\theta^*)=\sigma^2I(\theta)$ and $\bar\rho$ becomes $\int I^*/(I^*+1)\,\phi(\theta^*)\,d\theta^*$.
   So for $\sigma^2\ne1$ or $\mu\ne0$ (a fixed non-standard `Normal`, the focal groups of a
   multiple-group model) the reference is mirt's `marginal_rxx()` on the *standardized
   equivalent model* ($a^*=a\sigma$, $d^*=d+a\mu$ with $N(0,1)$), together with a direct
   `integrate()` of the formula above with mirt's `testinfo()`. The integral is evaluated on a
   dedicated grid, not the fitted quadrature: 801 equispaced points on $\mu_g\pm8\sigma_g$ with
   weights $\propto\phi$ (rectangle rule). The integrand is analytic and the Gaussian weight
   vanishes at the ends, so the rule converges geometrically (truncation $2\Phi(-8)\approx10^{-15}$);
   the result does not depend on the quadrature the model was fitted with. mirt uses
   `integrate()` with R's default tolerance (`rel.tol` $\approx1.2\times10^{-4}$), so the
   reference also stores a recomputation with `rel.tol = 1e-12`.
   **Empirical reliability.** From the EAP scores $\hat\theta_i$ and posterior SDs
   $\mathrm{SE}_i$ of the respondents in `data` (`score(method="EAP")`, on the fitted grid),
   $$\hat\rho=\frac{s^2_{\hat\theta}}{s^2_{\hat\theta}+\overline{\mathrm{SE}^2}},\qquad
   s^2_{\hat\theta}=\frac1{N-1}\sum_i(\hat\theta_i-\bar{\hat\theta})^2,\quad
   \overline{\mathrm{SE}^2}=\frac1N\sum_i\mathrm{SE}_i^2 ,$$
   which is mirt's `empirical_rxx()` with its default `T_as_X = FALSE`: the variance of the EAP
   scores estimates the true-score variance and the mean posterior variance the error
   variance, their sum estimating $\sigma^2$ (Thissen & Wainer, 2001). In a multiple-group
   model it is computed within each group from that group's respondents (mirt pools all
   respondents of a `multipleGroup` fit; the per-group value is `empirical_rxx()` of that
   group's rows). Rows with incomplete responses are scored as usual and kept, as in mirt. A
   group with fewer than two respondents gives NaN. `reliability(res)` returns the column
   `marginal`; with `data` it adds `empirical`.
   **Summed-score EAP** (Thissen, Pommerich, Billeaud & Williams, 1995). With the fitted
   quadrature nodes $\theta_q$ and latent weights $w_{gq}$ (the weights the EM integrates
   with) and $L_g(s\mid\theta_q)$ from `summed_score_distribution` (item 1, exact recursion),
   $$P_g(s)=\sum_qL_g(s\mid\theta_q)w_{gq},\qquad
   \hat\theta_g(s)=\frac{\sum_q\theta_qL_g(s\mid\theta_q)w_{gq}}{P_g(s)},\qquad
   \mathrm{SD}_g(s)=\Big(\frac{\sum_q(\theta_q-\hat\theta_g(s))^2L_g(s\mid\theta_q)w_{gq}}{P_g(s)}\Big)^{1/2}.$$
   `summed_score_table(res)` returns `score`, `theta`, `se`, `probability` ($P_g(s)$, the
   model-implied summed-score distribution) for $s=0,\dots,\sum_j(K_j-1)$. mirt's
   `fscores(method = "EAPsum", full.scores = FALSE)` is the same table with
   `expected` $=N\,P(s)$; its observed counts and $X^2$ need data and are not reproduced. A
   score whose $P_g(s)$ underflows to 0 gets NaN `theta` and `se`.
   `score(data, method="EAPsum")` looks each respondent's summed score up in the table of
   the respondent's group. **Incomplete patterns**: a respondent with any missing response gets
   NaN `theta` and `se`, because the summed score over the observed items is not on the scale
   of the table; this is what mirt's `fscores(method = "EAPsum", full.scores = TRUE)` returns
   (`NA`). Scoring the sub-test of observed items instead (one table per missingness pattern)
   is left out.
   **Latent distributions.** Information and expected score do not involve the latent
   distribution and are available for every unidimensional model. Reliability, the
   summed-score table and EAPsum scoring are validated for `Normal` latent distributions only
   (any mean and variance, single and multiple groups); `EmpiricalHistogram`, `DavidianCurve`
   and `NormalMixture` raise `SpecificationError` (mirt's `marginal_rxx()` has no counterpart
   for a fitted non-Normal density, and mirt's EAPsum uses a normal prior by default).
   **Nothing new is stored** in `FitResult`; `schema_version` is unchanged.
   **Measured agreement** is recorded in VALIDATION.md (V37).

3. **Information matrix and directional information of multidimensional models (#82, V43)**:
   **Scope.** Confirmatory models from `with_factors` (2–3 factors, any loading pattern
   including cross-loadings), bifactor models (`with_bifactor`) and two-tier models
   (`with_two_tier`), single and multiple groups. `item_information`, `test_information` and
   `expected_score` accept them; the new `information_matrix` returns the matrix itself.
   `reliability`, `summed_score_table` and `score(method="EAPsum")` stay unidimensional (a
   multidimensional summed-score EAP and a marginal reliability of a composite are separate
   questions without a mirt reference).
   **Information matrix.** With $\boldsymbol\theta\in\mathbb R^D$, item $j$'s Fisher information
   matrix and the test information matrix are
   $$\mathbf I_j(\boldsymbol\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\boldsymbol\theta)\,
     \nabla_{\boldsymbol\theta}\log P_{jk}(\boldsymbol\theta)\,
     \nabla_{\boldsymbol\theta}\log P_{jk}(\boldsymbol\theta)^\top,\qquad
     \mathbf I(\boldsymbol\theta)=\sum_j\mathbf I_j(\boldsymbol\theta)$$
   (Reckase, 2009, ch. 5), from the same forward-mode autodiff of the item kernel's category
   log-probabilities as item 2, now with respect to all $D$ coordinates (`jax.jacfwd` gives
   the $K_j\times D$ Jacobian). For every built-in family $P_{jk}$ depends on
   $\boldsymbol\theta$ only through $\mathbf a_j^\top\boldsymbol\theta$, so $\mathbf I_j=c_j(\boldsymbol\theta)\,\mathbf a_j\mathbf a_j^\top$
   has rank 1; the code does not use this (user-defined kernels need not have it). For a
   unidimensional model $D=1$ and the matrix is the scalar of item 2; there is one code path.
   In bifactor and two-tier models the kernels are evaluated at the full $D$-vector; the
   dimension reduction of §17 concerns only the integration over $\boldsymbol\theta$ and plays
   no role here.
   **Directional information.** Along a unit vector $\mathbf u$ ($\|\mathbf u\|=1$),
   $$I_{j,\mathbf u}(\boldsymbol\theta)=\mathbf u^\top\mathbf I_j(\boldsymbol\theta)\mathbf u,\qquad
     I_{\mathbf u}(\boldsymbol\theta)=\mathbf u^\top\mathbf I(\boldsymbol\theta)\mathbf u$$
   (Reckase & McKinley, 1991; Ackerman, 1994), the information about $\boldsymbol\theta$ when
   it moves along $\mathbf u$. mirt's `iteminfo(..., degrees = )` / `testinfo(..., degrees = )`
   take the angles $\alpha_m$ to the axes and use $u_m=\cos\alpha_m$ without normalizing;
   flexpyirt takes the direction vector and **normalizes it**, so the two agree whenever the
   angles are consistent ($\sum_m\cos^2\alpha_m=1$), and mirt's angles are
   `direction=np.cos(np.radians(degrees))`.
   **API (Decision, 2026-11-26).**
   - `theta` of a multidimensional model is a `(T, D)` array-like, or a DataFrame whose columns
     are exactly the factor names (in any order; reordered to the model's order); a 1-D array of
     length $D$ is a single point. Unidimensional models keep the 1-D `theta` of item 2. The
     result is indexed by a `MultiIndex` whose level names are the factor names (index name
     `theta` for unidimensional models, as before).
   - `item_information(res, theta, *, direction=None, group=None)` and
     `test_information(res, theta, *, direction=None, group=None)`: for a multidimensional model
     `direction` (1-D, length $D$, finite, non-zero; normalized) is **required**, because there
     is no natural default direction (mirt also requires `degrees`); one column per item,
     respectively the columns `information` and `se`. For a unidimensional model `direction`
     must be `None`. One direction per call; several directions are several calls.
   - `test_information`'s `se` is $\sqrt{\mathbf u^\top\mathbf I(\boldsymbol\theta)^{-1}\mathbf u}$,
     the asymptotic conditional standard error of the ML estimate of the composite
     $\mathbf u^\top\boldsymbol\theta$, which reduces to $I(\theta)^{-1/2}$ for $D=1$. It is not
     $I_{\mathbf u}^{-1/2}$, which would be the standard error if the coordinates orthogonal to
     $\mathbf u$ were known; the two coincide only when $\mathbf u$ is an eigenvector of
     $\mathbf I$. Where $\mathbf I(\boldsymbol\theta)$ is numerically singular (its smallest
     eigenvalue is at most $D\varepsilon$ times its largest, $\varepsilon$ the double-precision
     machine epsilon; e.g. fewer linearly independent slope vectors than factors), `se` is
     $+\infty$, as for $I=0$ in one dimension. The item matrices are symmetrized,
     $(\mathbf M+\mathbf M^\top)/2$, so that they are exactly symmetric. **This differs from
     mirt**: mirt has no SE in `testinfo()`, but its `plot(type = "SE")` / `type = "infoSE"` for
     two- and three-factor models draws $1/\sqrt{I_{\mathbf u}}$ (`1/sqrt(testinfo(x, Theta,
     degrees = ))`), the conditional-on-the-other-coordinates value above. flexpyirt reports the
     composite SE instead, because it is the standard error of an estimable quantity and equals
     the unidimensional SE when $D=1$; $1/\sqrt{I_{\mathbf u}}$ is `1 / np.sqrt(information)`
     of the same DataFrame. The reference inverts the sum of mirt's item information matrices
     in R.
   - `information_matrix(res, theta, *, item=None, group=None)` returns a NumPy array of shape
     `(T, D, D)`: the test information matrix, or with `item=` (an item name) that item's matrix.
     It is an array rather than a DataFrame because a $D\times D$ matrix per point has no flat
     layout that is not awkward to index; rows and columns follow the fitted model's `Model.factor_names`.
     Unidimensional models give `(T, 1, 1)`. This is mirt's
     `iteminfo(..., multidim_matrix = TRUE)` (per point), summed over items for the test.
   - `expected_score(res, theta, *, group=None)`: $E_j(\boldsymbol\theta)=\sum_kk\,P_{jk}(\boldsymbol\theta)$
     and `total`, as in item 2 with the multidimensional `theta`; mirt's `expected.item()` /
     `expected.test()` with a $D$-column `Theta`.
   **Nothing new is stored** in `FitResult`; `schema_version` is unchanged.
   **Measured agreement** is recorded in VALIDATION.md (V43).

4. **Classification accuracy and consistency for cut scores (#107, V52)**:
   **What it computes.** A test with cut scores $c_1<\dots<c_C$ sorts respondents into $H=C+1$
   ordered categories $h=0,\dots,C$; a value at or above $c_k$ is above that cut, so
   the category of a value $v$ is $h(v)=\#\{k: v\ge c_k\}$. From one administration and the
   fitted model, **classification accuracy** is the probability that the observed category
   equals the true category, and **classification consistency** the probability that two
   independent parallel administrations give the same category (Lee, 2010; Rudner, 2001, 2005).
   **Conditional category probabilities.** At a true value $\theta$:
   - **Lee (2010)** (`method="lee"`, cuts on the summed score). The observed category comes from
     the summed score $S$, whose conditional distribution is the Lord–Wingersky likelihood
     $L(s\mid\theta)$ of item 1 (exact recursion, every item family):
     $P_h(\theta)=\sum_{s:\,h(s)=h}L(s\mid\theta)$. The true category is that of the expected
     summed score (true score) $\tau(\theta)=\sum_j\sum_kk\,P_{jk}(\theta)$, i.e.
     $t(\theta)=h(\tau(\theta))$. A non-integer cut $c$ puts the scores $s\ge\lceil c\rceil$ above
     it (as cacIRT does).
   - **Rudner (2001, 2005)** (`method="rudner"`, cuts on $\theta$). The estimate is
     $\hat\theta\sim N(\theta,\mathrm{SE}(\theta)^2)$ with $\mathrm{SE}(\theta)=I(\theta)^{-1/2}$
     from the test information of item 2:
     $P_h(\theta)=\Phi\big((c_{h+1}-\theta)/\mathrm{SE}\big)-\Phi\big((c_h-\theta)/\mathrm{SE}\big)$
     with $c_0=-\infty$, $c_H=\infty$; the true category is $t(\theta)=h(\theta)$.
   **Integration over the true values.** With points $\theta_n$ and weights $w_n$
   ($\sum_nw_n=1$):
   - **D method** (`data=None`; Lathrop & Cheng, 2013): the fitted quadrature nodes and the
     fitted latent weights of the group (the weights the EM integrates with, so every latent
     distribution works; the R reference covers `Normal`).
   - **P method** (`data` given): the respondents' estimates $\hat\theta_i$ of the group
     (`score(data, method=score_method)`, default `"EAP"`) as if they were true values, each with
     weight $1/N_g$. For Rudner the SE is $I(\hat\theta_i)^{-1/2}$ from the information, not the
     scoring SE (as Rudner, 2005, and cacIRT). Non-finite estimates (ML on extreme patterns)
     raise `SpecificationError`.
   **Indices.** All of them come from two $H\times H$ tables,
   $$A_{tk}=\sum_nw_n\,[t(\theta_n)=t]\,P_k(\theta_n),\qquad
   B_{hk}=\sum_nw_nP_h(\theta_n)P_k(\theta_n),$$
   the joint distribution of (true, observed) category and of the categories of two independent
   administrations. Accuracy $\gamma=\sum_hA_{hh}$, consistency $\phi=\sum_hB_{hh}$, and
   Cohen's (1960) kappa for consistency $\kappa=(\phi-\phi_c)/(1-\phi_c)$,
   $\phi_c=\sum_hp_h^2$, $p_h=\sum_kB_{hk}$ (Lee, 2010). Conditional indices at $\theta_n$ are
   $\gamma(\theta_n)=P_{t(\theta_n)}(\theta_n)$ and $\phi(\theta_n)=\sum_hP_h(\theta_n)^2$.
   Per category $h$: `true_proportion` $\sum_kA_{hk}$, `proportion` $p_h$, `accuracy`
   $A_{hh}/\sum_kA_{hk}$ (observed $=h$ given true $=h$), `consistency` $B_{hh}/p_h$ (second
   $=h$ given first $=h$); NaN where the denominator is 0. Per cut $c_k$ (the two-category
   split "below $c_k$" / "at or above $c_k$"): accuracy, consistency and kappa of that
   dichotomy (sums of the blocks of $A$ and $B$), `false_positive`
   $\sum_{t<k\le h}A_{th}$ (true below, observed above) and `false_negative`
   $\sum_{h<k\le t}A_{th}$. With one cut, the per-cut and overall indices coincide. These are
   cacIRT's "Marginal" (per cut) and "Simultaneous" (overall) indices; kappa, false
   positive/negative rates and the per-category indices are not in cacIRT and are checked by an
   independent recomputation in R.
   **Scales.** `scale=None` uses the method's own scale. `method="lee", scale="theta"` maps each
   cut through the test characteristic curve, $c^S_k=\tau(c_k)$; `method="rudner",
   scale="summed_score"` inverts it, $c^\theta_k=\tau^{-1}(c_k)$ by bisection on $[-40, 40]$
   (requires $\tau$ to be increasing, which holds for every family except possibly nominal
   items, and $\tau(-40)<c_k<\tau(40)$). In a multiple-group model the converted cuts differ by
   group and are reported in `by_cut` (`summed_score_cut` / `theta_cut`).
   **Discretization (D approach).** The D approach integrates over the quadrature grid, i.e.
   over the discrete latent distribution the EM works with, as cacIRT does with the quadrature it
   is given. Consistency is smooth in $\theta$ and agrees with the continuous-latent integral to
   about $10^{-10}$ with 61 nodes; accuracy jumps where $t(\theta)$ changes, so it differs from
   the continuous integral by up to the order of the node spacing (measured up to 0.008 for a
   5-item test with 61 nodes on $[-6,6]$, VALIDATION V52). The value is reported on the fitted
   grid, not on a dedicated finer grid, so that every latent distribution (EH included) is
   handled the same way; a finer `Quadrature` or the P approach is the remedy.
   **Validation of cuts.** Finite, strictly increasing; on the summed score
   $0<c_k\le S_{\max}$ and the ceilings $\lceil c_k\rceil$ distinct (otherwise an observed
   category is empty). Otherwise `SpecificationError`.
   **API (Decision, owner approved the proposal in #107).**
   `classification_accuracy(res, *, cut_scores, method="lee", scale=None, data=None, group=None,
   item_names=None, score_method="EAP") -> ClassificationResult`, unidimensional models only.
   `ClassificationResult` holds DataFrames `summary` (`accuracy`, `consistency`, `kappa`; one
   row per group), `by_category`, `by_cut`, `conditional` (per node or respondent: `theta`,
   `weight`, `true_category`, `accuracy`, `consistency`, plus `true_score` for Lee or `se` for
   Rudner), `accuracy_table` ($A$, rows true, columns observed) and `consistency_table` ($B$);
   a multiple-group result has a leading `group` column (a `(group, category)` index for the
   tables). The float properties `accuracy`, `consistency` and `kappa` are for single-group
   results and raise `SpecificationError` otherwise. Multiple-group fits give one set of
   indices per group (the group's item parameters and latent distribution); with `data`, a group
   without respondents is left out.
   **Kernel.** `_classification.py` holds pure JAX functions (category probabilities for both
   methods, the tables $A$ and $B$); the shell computes the derived indices from the small
   $H\times H$ tables in NumPy.
   **Left out.** Model-free observed-score methods (Livingston & Lewis, 1995; Hanson & Brennan,
   1990), composite scores, standard errors of the indices, multidimensional models.
   **Nothing new is stored** in `FitResult`; `schema_version` is unchanged.
   **Measured agreement** is recorded in VALIDATION.md (V52).

## 19. Multidimensional scale linking (M6)

1. **Linking of confirmatory multidimensional models (`link(..., transformation=)`, V40, #78)**:
   `link()` (§16.5) also accepts two fits with the same number of factors $D \in \{2, 3\}$ declared
   with `Model.with_factors` (product grids; bifactor and two-tier fits are rejected). Factors are
   matched by position: source factor $m$ is mapped onto target factor $m$.
   **Transformation.** The source scale is mapped onto the target scale by
   $\boldsymbol\theta^* = \mathbf A\boldsymbol\theta + \boldsymbol\beta$ with a nonsingular
   $D \times D$ matrix $\mathbf A$ (Oshima, Davey & Lee, 2000; Li & Lissitz, 2000). Substituting
   $\boldsymbol\theta = \mathbf A^{-1}(\boldsymbol\theta^* - \boldsymbol\beta)$ into the linear
   predictor $\mathbf a^\top\boldsymbol\theta + d$ gives, for every kernel,
   $$\mathbf a^* = \mathbf A^{-\top}\mathbf a, \qquad d_k^* = d_k - m_k\,\mathbf a^{*\top}\boldsymbol\beta,$$
   where $m_k$ is the multiplier of the linear predictor in category $k$: 1 (2PL/Rasch, 3PL/4PL,
   GRM), $k$ (GPCM/PCM) or $\mathrm{ak}_k$ (nominal, $\mathrm{ak}_{K-1} = K-1$). Asymptotes and
   nominal scoring coefficients are unchanged, and $P_{jk}(\boldsymbol\theta^*; \xi^*) =
   P_{jk}(\mathbf A^{-1}(\boldsymbol\theta^* - \boldsymbol\beta); \xi)$. For $D = 1$ this is §16.5.
   A latent distribution $N(\boldsymbol\mu, \boldsymbol\Sigma)$ becomes
   $N(\mathbf A\boldsymbol\mu + \boldsymbol\beta, \mathbf A\boldsymbol\Sigma\mathbf A^\top)$.
   **Three families of $\mathbf A$** (`transformation`, a static argument; $p$ free parameters $c$):
   - `"diagonal"` (default; $p = 2D$): $\mathbf A = \operatorname{diag}(A_1, \dots, A_D)$,
     $A_m > 0$, $c = (A_1..A_D, \beta_1..\beta_D)$. Each factor is rescaled and shifted separately.
     This is the transformation between two calibrations of the *same* confirmatory structure
     (means 0, variances 1 in each): zero slopes stay zero and factor correlations are unchanged,
     so it is the only family under which the transformed items keep the source model's
     structure.
   - `"orthogonal"` ($p = 1 + D(D-1)/2 + D$; Li & Lissitz, 2000): $\mathbf A = k\,\mathbf R$ with a
     central dilation $k > 0$ and a rotation $\mathbf R = \exp(\mathbf S(\boldsymbol\phi))$, where
     $\mathbf S$ is skew-symmetric with $S_{ij} = -\phi_{ij}$, $S_{ji} = \phi_{ij}$ ($i < j$ in
     lexicographic order), $c = (k, \boldsymbol\phi, \boldsymbol\beta)$. For $D = 2$,
     $\mathbf R = \begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}$. Only
     proper rotations ($\det\mathbf R = +1$) are reached: reflections are excluded.
   - `"oblique"` ($p = D^2 + D$; Oshima, Davey & Lee, 2000): any $\mathbf A$ with
     $\det\mathbf A > 0$, $c = (\operatorname{vec}(\mathbf A)$ row-major$, \boldsymbol\beta)$.
   Rotations and oblique transformations turn structural zero slopes of the source items into
   non-zero slopes; they are for forms whose factors were identified differently.
   **Criteria** on the target product grid $(\boldsymbol\theta_q^{(t)}, w_q^{(t)})$ (the fitted
   multivariate normal weights of §15.30, or user `nodes` of shape $(Q, D)$ and `weights`):
   $$H_{\mathrm{fwd}}(c) = \sum_q w_q^{(t)} \sum_{j\in\mathcal C}\sum_{k=0}^{K_j-1}\big[P_{jk}^{\mathrm{tgt}}(\boldsymbol\theta_q^{(t)}) - P_{jk}^{\mathrm{src}}(\mathbf A^{-1}(\boldsymbol\theta_q^{(t)} - \boldsymbol\beta))\big]^2,$$
   $$\mathrm{SL}_{\mathrm{fwd}}(c) = \sum_q w_q^{(t)} \big[\mathrm{TCC}^{\mathrm{tgt}}(\boldsymbol\theta_q^{(t)}) - \mathrm{TCC}^{\mathrm{src}}(\mathbf A^{-1}(\boldsymbol\theta_q^{(t)} - \boldsymbol\beta))\big]^2,$$
   the item and test characteristic function methods of Oshima, Davey & Lee (2000) in the
   category form of Kim & Lee (2006), with $\mathrm{TCC}(\boldsymbol\theta) = \sum_j\sum_k k\,P_{jk}(\boldsymbol\theta)$.
   `symmetric=True` adds the reverse term on the source grid (source curves at
   $\boldsymbol\theta_q^{(s)}$ against target curves at $\mathbf A\boldsymbol\theta_q^{(s)} + \boldsymbol\beta$).
   Stocking–Lord matches one scalar surface, so it identifies $\mathbf A$ only through the
   variety of slope directions among the common items; Haebara uses every item surface. A
   criterion Hessian that is not positive definite gives no standard errors.
   **Optimization.** `_linking.solve_multidim_linking` is the Newton iteration of §16.5 in $p$
   dimensions (exact step when the Hessian is positive definite, Levenberg–Marquardt otherwise,
   step norm capped at 1, 40 parallel step sizes, gradient polish). Parameters outside the valid
   region ($A_m \le 0$, $k \le 0$, $\det\mathbf A \le 0$) evaluate to NaN and are rejected by the
   step-size search, so the iterates stay in the connected component of the start. **Start**:
   with $X_s, X_t$ the $J_c \times D$ slope matrices of the common items, $\mathbf a_s =
   \mathbf A^\top\mathbf a_t$ gives the least-squares $\mathbf A_0 = (X_t^{+} X_s)$ (oblique; the
   "direct" estimate of Oshima, Davey & Lee, 2000); the orthogonal start is its polar factor
   ($\mathbf A_0 = U\Sigma V^\top \Rightarrow \mathbf R_0 = UV^\top$, $k_0 = \operatorname{mean}\Sigma$,
   $\boldsymbol\phi_0$ from the matrix logarithm; $\mathbf R_0 = I$ if $\det UV^\top < 0$); the
   diagonal start is $A_{m,0} = \sum_j |a^{\mathrm{src}}_{jm}| / \sum_j |a^{\mathrm{tgt}}_{jm}|$. An
   invalid $\mathbf A_0$ falls back to the diagonal start. $\boldsymbol\beta_0$ is the
   least-squares solution of $d^{\mathrm{tgt}}_{jk} - d^{\mathrm{src}}_{jk} = -m_k\,
   \mathbf a_j^{\mathrm{tgt}\top}\boldsymbol\beta$ over all intercepts of the common items.
   **Standard errors.** As in §16.5, $J_s = -H_{cc}^{-1}H_{c,\eta_s}$,
   $J_t = -H_{cc}^{-1}H_{c,\eta_t}$ and $\operatorname{Cov}(\hat c) = J_s\Sigma_sJ_s^\top +
   J_t\Sigma_tJ_t^\top (+ \text{cross terms for one joint fit})$. The reported constants
   $\kappa(c) = (\operatorname{vec}(\mathbf A)$ row-major$, \boldsymbol\beta)$ are a smooth
   function of $c$, so $\operatorname{Cov}(\hat\kappa) = G\operatorname{Cov}(\hat c)G^\top$ with
   $G = \partial\kappa/\partial c^\top$ by `jax.jacfwd`. This $(D^2 + D)$-square matrix has rank
   at most $p$: structural zeros of a diagonal $\mathbf A$ have zero variance, and the entries of
   an orthogonal $\mathbf A$ are linearly dependent to first order.
   **Result.** For $D > 1$, `LinkResult.A` is the $D \times D$ array $\mathbf A$, `B` the length-$D$
   array $\boldsymbol\beta$, `A_se` / `B_se` arrays of the same shapes, and `vcov` the
   $(D^2 + D)$-square covariance of $(\operatorname{vec}(\mathbf A), \boldsymbol\beta)$; for
   $D = 1$ they remain floats and a $2 \times 2$ matrix. `LinkResult.transformation` records the
   family (always `"diagonal"` for $D = 1$, where the three coincide). `items` holds the
   transformed slopes `a1..aD` and intercepts. `transform_theta` maps `theta_<factor>` columns,
   `pv<m>_<factor>` columns and arrays of shape $(\dots, D)$ by $\mathbf A\boldsymbol\theta +
   \boldsymbol\beta$; `se_<factor>` columns become $A_m\,\mathrm{se}_m$ under a diagonal
   $\mathbf A$ and NaN otherwise (a rotated standard error needs the posterior covariance, which
   the score table does not carry).

## 20. User-defined item models (#86)

1. **Public contract.** `flexpyirt.items.ItemModel` and `flexpyirt.items.ItemKernel` (§6) are the
   public extension point: an object that implements them can be put in a `Model` like a built-in
   item model, and every part of the library that only evaluates `log_prob` works with it.
   Instead of adding more built-in models, a model such as GGUM is written by the user.
   - **`ItemModel`** (Spec layer; all members required): the class attribute `name` (a non-empty
     string that is not a built-in name; shown in the `model` column and stored in JSON),
     `kernel` (the kernel with `n_factors == 1`), `default_free()`, `start_values(item, rest)`
     and `bounds()`. The last three describe the unidimensional parameter layout
     `kernel.param_names`; in a model with $D$ factors the slope entry is used for each of the $D$
     slopes. It should be a frozen dataclass (hashable, compared by value, like `Model`).
   - **`ItemKernel`** (Core layer), required: `n_factors`, `with_factors(n_factors)`,
     `param_names`, `n_categories` (≥ 2; categories are coded $0..K-1$), `slope_params`,
     `intercept_params` and `log_prob(theta, params)`. The kernel must be hashable and compare
     equal exactly when it describes the same function (a frozen dataclass does both): it is a
     static argument of jit, and items with equal kernels form one family (§4.3).
   - **Optional kernel members**: `irt_names`, `irt_dependency_indices` and
     `irt_thresholds(params)`. With them, `FitResult.items` adds display thresholds with
     delta-method SEs (§7, §15.18); without them it shows the internal parameters only. Internal
     code reads them through `items.irt_names(kernel)`, `items.irt_dependency_indices(kernel)` and
     `items.irt_thresholds(kernel, params)`, which return empty values for a kernel without them.
     `slope_params` and `intercept_params` stay required (they decide the parameter layout and
     the identification checks of §4.5; `intercept_params` may be empty).
   - **Parameter layout**: `param_names[:n_factors]` are the slopes and equal `slope_params`
     (every model has a slope: the multidimensional starting values, the scale of identification
     checks, the Heywood check and the ML/WLE extreme-pattern rule all read it). The other
     parameters follow in an order that does not depend on `n_factors`.
   - **`log_prob(theta, params)`** receives the nodes as a `(Q, D)` float64 array on every code
     path (the ML/WLE scoring helpers that passed a 1-D `theta` now reshape it to `(1, D)`) and
     `params` as a float64 vector in `param_names` order, and returns the `(Q, n_categories)`
     log-probabilities. It must be written with `jax.numpy` (traced by jit, differentiated
     twice by autodiff for the Newton M-step and the SEs) and must not branch in Python on
     parameter values. For parameters outside the model's domain it should return NaN, which the
     M-step line search rejects (§15.3).
2. **Checking the contract.** `Model.__post_init__` calls `items.check_item_model` for every item
   whose model is not built in: a non-empty, non-built-in `name`; a hashable kernel with
   `n_factors == 1`; unique `param_names`; exactly one slope, first; `intercept_params` among the
   non-slope parameters; `n_categories >= 2`; `default_free()` and both bounds of length
   `len(param_names)` with `lower < upper`. Errors are `SpecificationError` naming the item.
   `log_prob` itself is not checked (its shape and values are only known when traced).
3. **What works with a user-defined model.**

   | Feature | Status |
   |---|---|
   | `fit` (EM, MAP with priors, fixed parameters, equality constraints, groups, weights, latent distributions) | Yes |
   | Multidimensional, bifactor and two-tier models | Yes, through `kernel.with_factors` |
   | Standard errors (every `se=` method) | Yes |
   | `score`: EAP, MAP, PV, EAPsum | Yes |
   | `score`: ML, WLE | Yes, but the all-lowest / all-highest pattern rule (§15.24) assumes that category $0$ is the lowest and $K-1$ the highest response for a positive slope; it is not meaningful for a non-monotone model |
   | `item_fit`, `m2`, `local_dependence`, `jsi`, `dif`, information, expected score, reliability, summed scores | Yes (they evaluate `log_prob`; the item score is the category $0..K-1$) |
   | Display thresholds in `FitResult.items` | Only with the optional kernel members |
   | `Model.fix_from(parameterization="slope_intercept")` | Yes |
   | `Model.fix_from(parameterization="irt")` | No (`SpecificationError`): the conversion from $b$ is model-specific |
   | `link()` | No (`SpecificationError`): the scale transformation of the parameters (§16.5, §19) is model-specific |
   | Free latent variance by default (§4.5) | Only for built-in `Rasch` / `PCM`; for a fixed-slope custom model use `Model.with_latent(Normal(free_var=True))` |
   | `to_json` / `from_json` | After `register_item_model` (item 4) |

4. **JSON (`register_item_model(cls, *, to_dict=None, from_dict=None)`).** A user-defined item
   model is stored as `{"type": cls.name, **to_dict(model)}`, the same shape as the built-in
   entries (§15.28), and rebuilt with `from_dict(fields)` (the object without `"type"`). The
   defaults handle a dataclass whose `__init__` fields are `bool`, `int`, finite `float` or
   `str`. The registry is per Python session (never `eval` or `pickle`): `to_json` of an
   unregistered model and `from_json` of an unknown `"type"` raise `SpecificationError` naming
   `register_item_model`. Registering a built-in name, or another class under a name already
   taken (with a different `__qualname__`), is an error; a class with the same `__qualname__`
   replaces the earlier registration (a re-run notebook cell). `from_dict` must return an
   instance of the registered class with that `name`, and the rebuilt model is checked as in
   item 2. No stored name changes, so `schema_version` stays 7: a file with a user-defined model
   is read by a session that registered the class, and older flexpyirt versions report its
   `"type"` as unknown.
5. **Starting values in multidimensional models.** For a user-defined model, the non-slope
   starting values are `start_values(item, rest)[1:]` with the rest score on all items (the
   built-in effective-slope rule of §15.31 is model-specific). The slopes use the built-in rule
   (factor-specific item–rest correlations), or `start_values(...)[0]` on every loaded factor
   when the slope is fixed by default.
6. **Verification.** `tests/test_custom_items.py` re-implements the 2PL (without display members)
   and the GRM (with them) against the public protocol only. Fitted together in one mixed model,
   they reproduce the built-in fit to $10^{-12}$: log-likelihood, iteration count, estimates,
   Oakes SEs, display thresholds and their SEs, EAP/MAP/ML/WLE scores, S-X2, M2 and item
   information. The multidimensional case agrees to $10^{-5}$ (different starting values, item
   5). The tests also cover JSON registration, the contract checks and the unsupported features.
   A user-defined model is as correct as its `log_prob`: flexpyirt verifies the machinery, not the
   model. The guide's worked example is the 2PL re-implementation; a model not built in (such as
   GGUM, Roberts, Donoghue & Laughlin, 2000) would be presented as validated only with its own
   mirt comparison.

## 21. Plots and the numbers behind them (#103, V48)

1. **Where plotting lives (Decision, owner 2026-11-26: option A of #103).** A module
   `flexpyirt.plotting` with matplotlib as the optional extra `flexpyirt[plot]`. matplotlib is
   imported inside each plot function, never by `import flexpyirt` or
   `import flexpyirt.plotting`; without it a plot function raises
   `errors.MissingDependencyError` (a `FlexpyirtError` and an `ImportError`) naming the extra.
   The numbers behind every figure come from public functions returning DataFrames, defined in
   `flexpyirt.curves` and exported from `flexpyirt`; only those numbers are validated (V48). The
   plot functions are thin: they call those functions, draw the columns, return matplotlib
   objects and never call `plt.show()`. They are smoke-tested on the Agg back end (objects
   created, plotted data equal to the DataFrames); there is no image comparison.
   **Scope.** Unidimensional models, every item family, single and multiple groups, any latent
   distribution. Interactive back ends, surfaces of multidimensional models, styling themes and
   figures for flexpyctt / flexpyequate are left out. Nothing is stored in `FitResult`;
   `schema_version` is unchanged.

2. **Trace lines: `item_probabilities(res, theta, *, items=None, group=None)`.** The category
   probabilities $P_{jk}(\theta)=\Pr(Y_j=k\mid\theta)$, $k=0,\dots,K_j-1$, from the item
   kernels (`_information.category_probabilities`, a jitted kernel returning
   $(G, T, J, K_{\max})$ padded with zeros). Long format: `item`, `category`, `theta`,
   `probability` (one row per item, category and $\theta$, in that order). `group` is a group
   name, as in `item_information()`; it is required for a multiple-group model. mirt:
   `probtrace()`.

3. **Empirical item curves: `empirical_item_curve(res, data, *, items=None, n_bins=10,
   group=None, item_names=None)`.** For item $j$, the $n_j$ respondents who answered it are
   sorted by their EAP score $\hat\theta_i$ (`score(method="EAP")` on the fitted grid; ties in
   row order, a stable sort) and cut into $B$ consecutive bins: every bin gets
   $\lfloor n_j/B\rfloor$ respondents and, of the remainder $c=n_j\bmod B$,
   $\lfloor c/2\rfloor$ go to the first bin and $c-\lfloor c/2\rfloor$ to the last. This is
   mirt's `itemfit()` rule (`group.bins`), equal-count bins of EAP $\theta$, the rule proposed
   in #103. For bin $b$,
   $$\bar\theta_b=\frac1{n_b}\sum_{i\in\mathcal B_b}\hat\theta_i,\qquad
   O_{bk}=\frac1{n_b}\sum_{i\in\mathcal B_b}[y_{ij}=k],\qquad E_{bk}=P_{jk}(\bar\theta_b)$$
   (Bock, 1972; Yen, 1981): the columns `theta`, `observed` and `expected`, with `n` $=n_b$,
   `bin` $=1,\dots,B$ and `category`. $n_bO_{bk}$ and $n_bE_{bk}$ are the observed and expected
   tables of mirt's `itemfit(fit_stats = "X2", return.tables = TRUE)` before its cell collapsing
   (`mincell.X2`), and $(\bar\theta_b,O_{bk})$ the points of `itemfit(empirical.plot = )`.
   Missing responses: a respondent enters the bins of the items they answered (mirt's `pick`).
   A multiple-group model is binned within each group with that group's parameters (mirt's
   per-group `itemfit()`); `group=` are the labels of the rows of `data`, as in `score()`, and
   the result has a leading `group` column. An item (in a group) with fewer than `n_bins`
   responses raises `SpecificationError` (mirt would form empty bins). Differences from mirt
   left out on purpose: no cell collapsing and no $X^2$ (a fit statistic is a separate issue),
   no Clopper–Pearson interval of `empirical.CI`, and `method` is EAP only (mirt's `method=`).

4. **Item locations of the Wright map: `wright_map_table(res, data, *, group=None,
   item_names=None, method="EAP")`.** Returns `WrightMapTable(persons, items)`, a named tuple of
   two DataFrames. `persons` has `theta` from `score(method=method)` (`"EAP"`, `"MAP"` or
   `"WLE"`, all finite) with `data`'s index. `items` has one row per item threshold: `item`,
   `category` $k=1,\dots,K_j-1$ and `location`, the Thurstonian threshold
   $$\tau_{jk}:\quad P(Y_j\ge k\mid\tau_{jk})=\sum_{c\ge k}P_{jc}(\tau_{jk})=\tfrac12$$
   (Wright & Masters, 1982; Wu, Adams, Wilson & Haldane, 2007; TAM's `tam.threshold()`). It is
   $b_j=-d_j/a_j$ for 2PL / Rasch and $b_{jk}$ for GRM items; for 3PL / 4PL items it is where
   $P=1/2$ (not $b$, where $P=(g+u)/2$) and for GPCM / PCM items it differs from the step
   difficulties. One definition for every family is why the issue's "`b` / Thurstonian
   threshold" became the threshold alone. **Root finding**: $P(Y_j\ge k\mid\theta)$ is
   evaluated on 4001 equispaced points of $[-20,20]$; the first sign change of
   $P(Y_j\ge k\mid\theta)-1/2$ is refined by 64 bisection steps, all thresholds at once (one
   kernel call per step). The function is increasing for every ordered family with $a>0$; for
   nominal items (unordered categories) it is the first crossing in the data's category order;
   without a crossing (e.g. a 3PL item with $g\ge1/2$) the location is NaN. In a
   multiple-group model `items` holds every group's thresholds from that group's parameters
   (a `group` column in both tables).

5. **Plot functions** (`flexpyirt.plotting`; default $\theta$ grid: 161 points on $[-4,4]$,
   `theta=` to change it).
   - `plot_item_curves(res, items=None, *, data=None, group=None, n_bins=10, theta=None,
     item_names=None, ax=None) -> Figure`: one panel per item (three per row). Dichotomous
     items show $P_{j1}$, polytomous items every category. With `data` the points of
     `empirical_item_curve()`. A multiple-group fit draws one curve per group in each panel
     (colour by group, line style by category): the **DIF plot**. `ax=` draws one item into an
     existing Axes (the figure of `ax` is returned). `group=` are row labels used with `data`;
     the curves always cover every group.
   - `plot_information(res, items=None, *, standard_error=False, group=None, theta=None,
     ax=None) -> Axes`: `test_information()` (default) or `item_information()` of the given
     items; `standard_error=True` adds `se` on `ax.twinx()`. `group` is a group name (default:
     every group).
   - `plot_expected_score(res, *, group=None, theta=None, ax=None) -> Axes`: column `total` of
     `expected_score()`, one line per group by default.
   - `plot_wright_map(res, data, *, group=None, item_names=None, method="EAP", n_bins=30) ->
     Figure`: two axes sharing the $\theta$ axis, a horizontal histogram of `persons` and the
     `items` thresholds as markers per item (overlaid per group).
   **Functions, not `FitResult.plot_*()` methods**, as `item_information()` and `link()`: the
   plotting dependency stays out of the result class.
   **`group=` means** a group name in the functions of $\theta$ (`item_probabilities`,
   `plot_information`, `plot_expected_score`, like `item_information`) and the labels of the
   rows of `data` where `data` is taken (`empirical_item_curve`, `wright_map_table`,
   `plot_item_curves`, `plot_wright_map`, like `reliability` and `score`).

## 22. Common-item screening and item parameter drift (#105)

1. **`screen_common_items()` (V50, #105)**: screens the common items of a unidimensional linking
   (§16.5) for item parameter drift and repeats the linking without the items that moved. It is a
   separate function, not an argument of `link()`: `link()` takes the anchor set as given, and the
   screening calls `link()` once per purification step. Models: one-factor fits with the item
   families `link()` accepts (2PL, Rasch, 3PL, 4PL, GRM, GPCM, PCM, nominal, mixed); multiple-group
   fits through `source_group` / `target_group`. Multidimensional fits raise
   `SpecificationError`.
   **Per-item quantities** under the current linking $(A, B)$, with the source item transformed
   to the target scale ($\xi_j^* $ of §16.5):
   - slope ratio $\ell_j = \log|a_j^{*}| - \log|a_j^{\mathrm{tgt}}|$ (`log_ratio_a`), with
     $a_j^* = a_j^{\mathrm{src}}/A$; NaN when a slope is 0;
   - location difference $\delta_j = b_j^{*} - b_j^{\mathrm{tgt}}$ (`difference_b`, source minus
     target), where the **location** $b_j$ of an item is its display difficulty $-d/a$ for 2PL /
     Rasch / 3PL / 4PL, the mean of its display thresholds $\bar b_j = \frac{1}{K_j-1}\sum_k b_{jk}$
     for GRM and GPCM / PCM (the step difficulties for GPCM), and $-d_{K-1}/((K-1)\,a)$ for the
     nominal model (the crossing point of its first and last category curves, as in the Mean/Mean
     start of §16.5). Every one of these transforms as $b^* = A\,b + B$, so
     $\delta_j = A\,b_j^{\mathrm{src}} + B - b_j^{\mathrm{tgt}}$ (decision for #105 open question 4:
     the mean threshold keeps one location per item, so a polytomous item is not counted
     $K_j - 1$ times in the median and IQR);
   - displacement $|\delta_j|$ (`displacement`);
   - $\mathrm{RMSD}_j = \big\{\sum_q w_q^{(t)}\,[E_j^{*}(\theta_q) - E_j^{\mathrm{tgt}}(\theta_q)]^2\big\}^{1/2}$
     (`rmsd`), the root mean squared difference of the expected item score curves
     $E_j(\theta) = \sum_{k=0}^{K_j-1} k\,P_{jk}(\theta)$ (category index as score, as in the
     Stocking–Lord TCC of §16.5) on the target group's fitted grid $(\theta_q^{(t)}, w_q^{(t)})$ (the
     default grid of `link()`). $E_j^*(\theta) = E_j^{\mathrm{src}}((\theta - B)/A)$. It compares two
     model curves; it is not the observed-versus-model RMSD of item fit. It is on the item's score
     scale ($0$ to $K_j - 1$) and is not divided by $K_j - 1$.
   **Robust z** (Huynh & Meyer, 2010): for $x \in \{\ell, \delta\}$ over the common items $\mathcal C$,
   $$z_j = \frac{x_j - \operatorname{median}_{i\in\mathcal C} x_i}{0.74\,\mathrm{IQR}_{i\in\mathcal C}(x_i)},$$
   with the IQR from linearly interpolated quantiles (R `quantile(type = 7)`, NumPy's default
   `linear`); $0.74 \approx 1/1.349$ makes $0.74\,\mathrm{IQR}$ a consistent estimate of a normal
   standard deviation. The median and IQR are taken over **all** common items, removed ones
   included, and only the differences are recomputed after each relinking: both are robust to the
   drifted items by construction, whereas re-estimating them on the shrinking anchor set shrinks
   the IQR and flags further items at every step (in the V50a simulation, with three planted
   drifts among 12 common items, it removed 10 items before `min_items` stopped it). When the IQR is not positive (below $10^{-12}$, e.g. the slope ratios of Rasch items, which
   are all $-\log A$), that z is NaN and does not flag. Huynh & Meyer compute the statistic on the
   two calibrations' raw parameters; here it is computed after the linking, on $b^*$ and $a^*$. The
   slope statistic is the same either way ($\log A$ shifts every $\ell_j$ equally), and the location
   statistic is the same when $A = 1$ (Rasch); otherwise the transformed version keeps both
   locations on the target scale, which is also the scale of the displacement.
   **Statistic and flag.** `statistic="robust_z"`: $s_j = \max(|z_{a,j}|, |z_{b,j}|)$ (NaN entries
   ignored), default threshold 1.645 (Huynh & Meyer, 2010); `"displacement"`: $s_j = |\delta_j|$,
   default 0.3 logits; `"rmsd"`: $s_j = \mathrm{RMSD}_j$, default 0.1. An item is flagged when
   $s_j > $ `threshold`. The defaults of displacement and RMSD are conventions agreed in #105, not
   derived values; programmes set their own.
   **Purification** (`purify=True`; iterative linking in the sense of Candell & Drasgow, 1988):
   step 0 links on all common items. At each step the statistics are recomputed under the current
   linking and anchor set; if no kept item is flagged the loop stops. Otherwise
   `removal="largest"` removes the flagged item with the largest $s_j$ (ties: the first in item
   order) and `removal="all"` removes every flagged item, in decreasing $s_j$; the linking is
   repeated on the remaining items. Removals never take the anchor set below `min_items` (default
   2): when no item can be removed the loop stops with a `ScreeningWarning`, and so it does when
   `max_iter` removal steps are used while kept items are still flagged. `purify=False` computes the
   statistics once, under the linking on all common items, and removes nothing.
   **Result** (`CommonItemScreeningResult`): `table` (one row per common item under the final
   linking: `item`, `target_item`, `model`, `a_source` / `b_source` (transformed), `a_target` /
   `b_target`, `log_ratio_a`, `difference_b`, `displacement`, `robust_z_a`, `robust_z_b`, `rmsd`,
   `statistic`, `flagged`, `kept`, `iteration_removed` (nullable integer)), `link` (final
   `LinkResult`), `initial_link`, `kept_items`, `removed_items` (in removal order), `history` (one row
   per step: `iteration`, `n_items`, `A`, `B`, `criterion`, `converged`, `removed`).
   `stack_screening_tables({administration: result, ...})` stacks the tables of several
   administrations screened against one base form, with an `administration` column first.
   **Not included**: likelihood-ratio or Wald DIF tests inside one multiple-group fit
   (`FitResult.dif()`, #104), robust linking criteria that down-weight instead of removing items,
   linking error over chains, multidimensional screening.

## 23. IRT true-score and observed-score equating, raw-to-scale tables (#106)

1. **Scope and API (Decision, owner 2026-11-26: the API and scope proposed in #106)**:
   unidimensional models, every item family, single- and multiple-group fits.
   - `irt_equate(new, old, *, method="true_score", new_items=None, old_items=None, link=None,
     new_group=None, old_group=None, population="new", synthetic_weight=None)` →
     `IRTEquatingResult` converts every summed score $x=0,\dots,X_{\max}$ of the new form $X$ to
     the scale of summed scores on the old form $Y$. `new` and `old` are `FitResult`s whose
     $\theta$ scales coincide (FIPC, a concurrent calibration, or the same fit with
     `new_items` / `old_items` naming the items of each form), or `link=` is a `LinkResult` that
     maps the scale of `new` onto that of `old`, $\theta_Y = A\theta_X + B$ (§16.5). The common
     scale is always the old form's. `link` must be unidimensional and its source and target
     groups must be `new_group` and `old_group`.
   - `score_distribution(res, *, items=None, group=None)` → DataFrame `score`, `probability`:
     the model-implied summed-score distribution of a form in the fitted population of a group.
   - `scale_table(res, *, slope, intercept, method="true_score", items=None, group=None,
     lowest=None, highest=None, n_digits=0)` → DataFrame `score`, `theta`, `theta_se`,
     `scale_score`, `scale_score_se`, `scale_score_rounded`.
   **Deviation from the #106 sketch.** `synthetic_weight` defaults to `None` and is allowed (and
   required) only with `population="synthetic"`; `population="new"` / `"old"` are the weights 1
   and 0. One argument then never silently overrides the other.
   **Left out** (follow-ups): standard errors of equated scores (delta method through `vcov`),
   kernel smoothing of the model-implied distributions, nonlinear scale transformations,
   multidimensional models.

2. **Form curves on the common scale**: each form is a set of items with kernel parameters
   $\xi_j$ of one group of one fit. With a link, the new form's items are evaluated on their own
   scale at $\theta_X=(\theta-B)/A$, so $P^{X}_{jk}(\theta)=P_{jk}((\theta-B)/A;\xi_j)$; this is
   exactly the transformed item of §16.5 for every kernel, without transforming parameters.
   The test characteristic curve (TCC) is
   $$\tau(\theta)=\sum_j\sum_{k=0}^{K_j-1}k\,P_{jk}(\theta),$$
   categories scored $0,\dots,K_j-1$ as in `expected_score()`. Its asymptotes
   $\tau_{\mathrm{lo}}=\lim_{\theta\to-\infty}\tau$ and $\tau_{\mathrm{hi}}=\lim_{\theta\to\infty}\tau$
   (the sums of the lower and upper asymptotes $g_j$, $u_j$ of 3PL/4PL items, 0 and $K_j-1$ for
   the others) are evaluated at $\theta_X=\mp10^{8}$, where every kernel has reached its limit in
   double precision unless a slope is below about $10^{-6}$. True-score methods need a
   non-decreasing TCC: it is checked on 801 points of $[-20,20]$ (own scale) and a decrease of
   more than $10^{-10}(1+X_{\max})$ raises `SpecificationError` (negative slopes, nominal items
   whose expected score is not monotone). Far in the tails the graded kernel returns NaN for a
   category whose two cumulative probabilities round to the same double (`log1mexp(0)`); there
   the probability has underflowed, and the form's category probabilities set NaN to 0. Curves
   are evaluated on θ arrays padded to a power of 2 of at least 1024 points, so a form compiles
   each curve once whatever the number of scores.

3. **True-score equating (Lord, 1980; Kolen & Brennan, 2014, §6.5)**: for
   $\tau^X_{\mathrm{lo}}<x<\tau^X_{\mathrm{hi}}$, solve $\tau^X(\theta_x)=x$ and set
   $e_Y(x)=\tau^Y(\theta_x)$. **Solver**: safeguarded Newton (Newton–bisection) on all scores at
   once inside one `jit`-ed `lax.while_loop`; the bracket starts at $[-10,10]$ on the form's own
   scale and is doubled (in Python, up to $|\theta|=10^{6}$) until it holds every root;
   $\tau'$ comes from `jax.jvp`. A Newton step that leaves the bracket, or a non-positive
   derivative, is replaced by bisection. Stop when every $|\tau^X(\theta)-x|\le10^{-13}(1+X_{\max})$
   or the bracket is narrower than $10^{-14}(1+|\theta|)$ (at most 300 iterations; not
   converging raises `ConvergenceWarning`).
   **Scores outside the range of the TCC** (Kolen & Brennan, 2014, §6.5, eq. 6.27 and the
   procedure for the highest score): no $\theta$ solves the equation, and the scores are
   equated by linear interpolation between the points $(0,0)$,
   $(\tau^X_{\mathrm{lo}},\tau^Y_{\mathrm{lo}})$ and, at the top,
   $(\tau^X_{\mathrm{hi}},\tau^Y_{\mathrm{hi}})$, $(X_{\max},Y_{\max})$:
   $$e_Y(x)=x\,\frac{\tau^Y_{\mathrm{lo}}}{\tau^X_{\mathrm{lo}}}\ (0\le x\le\tau^X_{\mathrm{lo}}),\qquad
   e_Y(x)=\tau^Y_{\mathrm{hi}}+(x-\tau^X_{\mathrm{hi}})\frac{Y_{\max}-\tau^Y_{\mathrm{hi}}}{X_{\max}-\tau^X_{\mathrm{hi}}}\ (\tau^X_{\mathrm{hi}}\le x\le X_{\max}).$$
   When $\tau^X_{\mathrm{lo}}=0$ (no lower asymptotes) only $x=0$ is in the first range and maps to
   0; when $\tau^X_{\mathrm{hi}}=X_{\max}$ only $x=X_{\max}$ maps to $Y_{\max}$. The `theta`
   column is $-\infty$ for $x\le\tau^X_{\mathrm{lo}}$ and $+\infty$ for $x\ge\tau^X_{\mathrm{hi}}$
   (the limits of $\theta_x$; for $x$ strictly beyond an asymptote there is no solution).

4. **Observed-score equating (Lord & Wingersky, 1984; Kolen & Brennan, 2014, §6.6)**:
   **Populations.** The new population is the fitted latent distribution of `new_group` in `new`
   (nodes and weights as `link()` uses them: Normal, EH, Davidian or mixture, `_fitted_grid`),
   with nodes mapped to the common scale by $A\theta+B$ when a link is given; the old population
   is that of `old_group` in `old`. The synthetic population is
   $w_1\cdot\text{new}+(1-w_1)\cdot\text{old}$ with $w_1$ = 1, 0 or `synthetic_weight`
   (Braun & Holland, 1982).
   **Distributions.** For each form and population, $f(x)=\sum_qL(x\mid\theta_q)w_q$ with the
   Lord–Wingersky likelihood of §18.1 (the existing kernel `summed_score_distribution`,
   `method="recursion"`); in the synthetic population
   $f_s(x)=w_1f_1(x)+(1-w_1)f_2(x)$, the same for $g_s(y)$.
   **Equipercentile step (Decision for #106 owner question 2: option (b))**: flexpyirt keeps
   its own percentile-rank functions for integer scores, tested against flexpyequate
   (`flexpyequate.equate(method="equipercentile")`, a development dependency only). With
   $F$, $G$ the cumulative distributions, $P(x)=F(x-1)+f(x)/2$, and
   $$e_Y(x)=Q^{-1}(P(x)),\quad Q^{-1}_U(p)=\frac{p-G(y_U-1)}{g(y_U)}+y_U-\tfrac12,\ y_U=\min\{y:G(y)>p\},$$
   $Q^{-1}_L(p)=\frac{p-G(y_L)}{g(y_L+1)}+y_L+\tfrac12$, $y_L=\max\{y:G(y)<p\}$, and
   $Q^{-1}=(Q^{-1}_U+Q^{-1}_L)/2$ (Kolen & Brennan, 2014, ch. 2; the two coincide unless some
   $g(y)=0$). This is what flexpyequate, `equate::equate(type = "equipercentile")` and
   equateIRT's OSE compute. A dependency on flexpyequate (option (a)) was not taken because
   flexpyirt would then pull in a second distribution for about 30 lines of NumPy, and
   flexpyequate's design forbids the reverse direction anyway.

5. **Raw-to-scale tables**: `method="true_score"`: $\theta(x)$ as in point 3 on the fit's own
   scale ($\pm\infty$ outside the TCC's range), `theta_se` $=I(\theta)^{-1/2}$ with the test
   information of the form's items (§18.2; NaN where $\theta$ is infinite).
   `method="EAPsum"`: the summed-score EAP and posterior SD of §18.2 computed for the form's items
   under the group's fitted latent distribution (any latent family; equals
   `summed_score_table()` for a whole Normal form). Then
   `scale_score` $=\mathrm{clip}(\text{slope}\cdot\theta+\text{intercept},\text{lowest},\text{highest})$
   (slope $>0$; a missing bound is not applied, so infinite $\theta$ gives $\pm\infty$),
   `scale_score_se` $=\text{slope}\cdot$`theta_se` (not clipped), and `scale_score_rounded` is
   NumPy's `round(scale_score, n_digits)` (half to even; ties do not occur in practice).

6. **Layers**: `_equating.py` (Core, pure JAX: form category probabilities, TCC and its
   derivative, information, the Newton–bisection solver, summed-score distributions and EAPs on
   a grid; NumPy: the equipercentile functions) and `equating.py` (shell: validation, forms from
   fits, the K&B boundary rules, DataFrames, `IRTEquatingResult`). Validated in V51.

## 24. Model comparison and simulation of responses (#109)

1. **Model-comparison table (`compare_fits()`, V54)**: `compare_fits(*results, names=None)` takes
   two or more `FitResult`s of the same responses, ordered from the most restricted model to the
   least restricted, and returns one row per fit: `model`, `n_params`, `loglik`, `aic`, `sabic`,
   `hq`, `bic`, and against the previous row `x2`, `df`, `p`. With $\ell$ the marginal
   log-likelihood, $k$ = `n_params` (EH weights included, §4.8) and $N$ the number of respondents
   (the sum of the weights for weighted fits, as `bic`; §17.4):
   $$\mathrm{AIC}=-2\ell+2k,\quad\mathrm{SABIC}=-2\ell+k\log\tfrac{N+2}{24},\quad
   \mathrm{HQ}=-2\ell+2k\log\log N,\quad\mathrm{BIC}=-2\ell+k\log N$$
   (Akaike, 1974; Sclove, 1987; Hannan & Quinn, 1979; Schwarz, 1978), and
   $X^2_m=2(\ell_m-\ell_{m-1})$, $\mathit{df}_m=k_m-k_{m-1}$, $p_m=\Pr(\chi^2_{\mathit{df}_m}>X^2_m)$
   (Wilks, 1938). This is mirt's `anova(mod1, mod2, ...)`; the criteria share
   `_dif.information_criteria` with `FitResult.dif()` (§16.2), and the column order follows
   `dif()` (`aic, sabic, hq, bic`). **Checks.** `SpecificationError` for fewer than two fits,
   non-`FitResult` arguments, `names` that are not one distinct label per fit, or fits with
   different $N$ (the data themselves are not compared). `FlexpyirtWarning` (not an error, as
   mirt) when $\mathit{df}\le0$ (then $p$ is NaN: the fits are not ordered from the most
   restricted model), when $X^2<0$ while parameters are added (a non-nested pair or a local
   maximum; $p$ is computed at $X^2=0$, i.e. 1), and when a fit has active priors (as
   `FitResult.aic`). **Not checked.** Nesting cannot be verified from two fits, and no boundary
   correction is made (a variance fixed at 0, fewer mixture components, `DavidianCurve` against
   `Normal`, §16.4): both are the caller's responsibility and are stated in the docstring.
   `df` is a nullable integer column (`Int64`), NaN in the first row.
2. **Simulation of responses (`simulate_responses()`, V54)**: `simulate_responses(source, *,
   n_obs=None, key, parameters=None, theta=None, group=None)` returns `SimulatedData`
   (`responses`: a DataFrame with the item names and `int64` codes $0..K-1$, accepted by `fit()`;
   `theta`: $(N,)$ or $(N,D)$; `group`: labels or `None`). Random numbers come from an explicit JAX
   PRNG key (as `score(method="PV")`): the key is split into one key for $\boldsymbol\theta$
   (split again per group) and one for the $N\times J$ uniforms, so a draw is reproducible from
   `key`. **Responses.** With the category probabilities $P_{jk}(\boldsymbol\theta_i)$ of the
   fitted kernels (`_core._item_log_probs`, jitted, with the grid replaced by the drawn points and
   the bifactor / two-tier reduction switched off, as `FitResult._group_traces`), each response
   is drawn by inversion of one uniform number,
   $$X_{ij}=\min\{k:\ u_{ij}\le F_{jk}(\boldsymbol\theta_i)\}=\#\{k<K_j-1:\ u_{ij}>F_{jk}\},\qquad
   F_{jk}=\textstyle\sum_{c\le k}P_{jc}$$
   (Devroye, 1986, §III.2); the last cumulative probability is never compared, so round-off in
   $\sum_cP_{jc}$ cannot give a category beyond $K_j-1$. **Latent draws** per group, from the
   fitted (or specified) distribution:
   - `Normal`: $\boldsymbol\mu+\mathbf L\mathbf z$, $\mathbf{LL}^\top=\boldsymbol\Sigma$ built
     from `mean_m`, `var_m` and `cor_ij` (correlations absent from the table are 0: bifactor
     factors, two-tier specific factors);
   - `NormalMixture`: component $c\sim\mathrm{Cat}(\boldsymbol\pi)$, then $N(\mu_c,\sigma_c^2)$
     with the standardized components of `mixture_components` (§16.7), so the draws have mean
     $\mu$ and variance $\sigma^2$ exactly in expectation;
   - `DavidianCurve`: the continuous density $f_k$ of §16.4. $z$ is drawn from
     $h_k(z)=P_k(z)^2\varphi(z)$ by inverting its CDF (trapezoidal rule on 24 001 points of
     $[-12,12]$, linear interpolation; the truncated mass is below $10^{-16}$ for $k\le10$), then
     $\theta=\mu+\sigma(z-\mu_Z)/\sigma_Z$;
   - `EmpiricalHistogram`: the quadrature nodes with the fitted weights. The fitted distribution
     *is* this discrete distribution (§4.4), so the draws take the grid values only; jittering
     within a bin would simulate from a distribution that was not fitted.
   The likelihood uses grid-normalized versions of the continuous densities (§4.4); on the
   default grids the difference is far below Monte Carlo error. **Sources.** A `FitResult` uses
   its estimates (`parameters["estimate"]`, `_eta`, `_eh_weights`). A `Model` needs
   `parameters`, an item bank as for `Model.fix_from()` in the slope–intercept parameterization
   with every item parameter (any item parameter left free raises `SpecificationError`); the
   table is built by `build_table` on dummy responses that contain every category, and the latent
   values are the specification's (`Normal` mean / var, factor correlations 0, `NormalMixture`
   starting shape). `EmpiricalHistogram` and `DavidianCurve` have no shape before fitting and
   raise unless `theta` is given. **Sizes.** `n_obs` is an integer (per group for a
   multiple-group model) or a mapping group → size (groups left out get none); rows are ordered
   by group. With `theta` ($(N,)$ / $(N,1)$ for one factor, $(N,D)$ otherwise), `n_obs` is
   omitted and a multiple-group model needs `group=` labels, one per row. **Left out** (#109):
   missing-data mechanisms and test designs (the user masks `responses`), recovery-study
   drivers, parametric bootstrap of fit statistics and adaptive testing.
   `FitResult._simulate_responses` (unidimensional `Normal`, used by model-based parallel
   analysis, §16.11) is kept as it is so that its random stream and validated results do not
   change.

## 25. Person fit and residual item fit (#108)

1. **Scope and API (V53, #108)**: `FitResult.person_fit(data, *, method="EAP", group=None,
   item_names=None, statistics=("lz", "infit", "outfit"))` returns one row per respondent with
   `theta`, the requested statistics (`lz`, `lz_star`, `infit`, `z_infit`, `outfit`, `z_outfit`)
   and `n_items` (answered items). `FitResult.item_fit(data, *, statistics=("s_x2",),
   method="EAP", ...)` gains `"infit"` and `"outfit"`; the default stays S-X2 (§15.40), so
   existing calls do not change, and the residual columns are merged onto the S-X2 table when
   both are requested. Unidimensional models (one or several groups, every item family, any
   latent distribution `score()` supports for the chosen estimator); missing responses are
   skipped. `method` is the ability estimator of `score()` restricted to the point estimators
   EAP, MAP, ML and WLE (PV and EAPsum raise). Every statistic is evaluated at the respondent's
   estimate $\hat\theta$ with the respondent's group's item parameters. Owner decisions taken
   in #108: a method on `FitResult` (as `item_fit()`), `statistics=` on `item_fit()` rather than
   a separate method, infit / outfit allowed for every item family (as mirt; their reference
   distribution is derived for Rasch-family models, so for other models they are descriptive
   and the guide says so), and $l_z^*$ validated by an R recomputation (item 5).
   Out of scope: nonparametric person fit (U3, $H^T$, caution indices), CUSUM, resampling
   p-values, TAM's marginal infit / outfit, multidimensional models.
   **Layers.** `_personfit.pattern_statistics` (jit, `vmap` over the unique patterns of the
   compiled problem) returns per pattern $l_z$, $l_z^*$, the number of answered items and the
   per-item residual moments $r_{pj}^2$, $W_{pj}$, $C_{pj}$ (NaN for unanswered items);
   derivatives in $\theta$ come from JAX autodiff (`jacfwd` of each kernel's `log_prob`).
   `_personfit.mean_square` forms the mean squares and $z$ values in NumPy along the item axis
   (persons) or the person axis (items, within each group).
2. **$l_z$ (Drasgow, Levine & Williams, 1985)**: with $A$ the answered items and
   $P_{jk}=P_{jk}(\hat\theta)$,
   $$l_0=\sum_{j\in A}\log P_{jy_j},\qquad
   \mathrm E(l_0)=\sum_{j\in A}\sum_k P_{jk}\log P_{jk},\qquad
   \mathrm{Var}(l_0)=\sum_{j\in A}\Big[\sum_k P_{jk}(\log P_{jk})^2-\Big(\sum_k P_{jk}\log
   P_{jk}\Big)^2\Big],$$
   $l_z=(l_0-\mathrm E(l_0))/\sqrt{\mathrm{Var}(l_0)}$, the polytomous form of mirt's `Zh`
   (mirt writes the variance as $\sum_{k\ne k'}P_kP_{k'}\log P_k\log(P_k/P_{k'})$, the same
   quantity). Terms with $P_{jk}=0$ contribute 0.
3. **$l_z^*$ (Snijders, 2001; polytomous items: Sinharay, 2016)**: $l_z$ evaluated at
   $\hat\theta$ instead of $\theta$ has variance below 1. Write
   $l_0-\mathrm E(l_0)=\sum_j a_j$ with $a_j=\log P_{jy_j}-\sum_k P_{jk}\log P_{jk}$ (mean 0) and
   let the estimator solve $\sum_j b_j(\theta)+r_0(\theta)=0$ with $b_j=\partial\log
   P_{jy_j}/\partial\theta$. A first-order expansion of $\sum_j a_j(\hat\theta)$ around $\theta$,
   with $\mathrm E[\partial a_j/\partial\theta]=-\mathrm{Cov}(a_j,b_j)=-\sum_k P'_{jk}\log P_{jk}$
   and $\hat\theta-\theta\approx(\sum_j b_j+r_0)/\sum_j I_j$, gives
   $$l_z^*=\frac{l_0-\mathrm E(l_0)+c\,r_0(\hat\theta)}{\tau},\quad
   c=\frac{\sum_j\sum_k P'_{jk}\log P_{jk}}{\sum_j I_j},\quad
   \tau^2=\sum_j\Big[\mathrm{Var}(a_j)-2c\sum_k P'_{jk}\log P_{jk}+c^2I_j\Big],$$
   all at $\hat\theta$, with $I_j=\sum_k P'^2_{jk}/P_{jk}$. For dichotomous items this is
   Snijders' $l_z^*$ with $w_j=\log(P_j/Q_j)$ and $r_j=P'_j/(P_jQ_j)$. The estimator's term is
   $r_0=0$ for ML, $r_0=-(\theta-\mu_g)/\sigma_g^2$ for MAP (the group's normal prior, §15.22)
   and $r_0=I'(\theta)/(2I(\theta))$ for WLE — flexpyirt's WLE maximizes $\log L+\tfrac12\log I$
   (§15.24), so its $r_0$ uses $I'=\partial I/\partial\theta$, which equals Warm's
   $\sum_jP'_jP''_j/(P_jQ_j)$ for 1PL / 2PL items but not for 3PL / 4PL. EAP has no estimating
   equation of this form, so `"lz_star"` with `method="EAP"` raises.
4. **Residuals, infit and outfit (Wright & Masters, 1982)**: with category scores
   $0,\dots,K_j-1$, $E_j=\sum_k kP_{jk}$, $r_j=y_j-E_j$, $W_j=\sum_k(k-E_j)^2P_{jk}$ and
   $C_j=\sum_k(k-E_j)^4P_{jk}$, over the $n$ answered items of a respondent (or the $N_j$
   respondents who answered item $j$):
   $$\mathrm{outfit}=\frac1n\sum\frac{r^2}{W},\quad
   q_{\mathrm{out}}^2=\frac1{n^2}\sum\frac{C}{W^2}-\frac1n,\qquad
   \mathrm{infit}=\frac{\sum r^2}{\sum W},\quad
   q_{\mathrm{in}}^2=\frac{\sum(C-W^2)}{(\sum W)^2},$$
   $q$ capped at 1.4142 and $z=(\mathrm{MS}^{1/3}-1)(3/q)+q/3$ (Wilson–Hilferty). Category
   scores $0,\dots,K_j-1$ are used for every family, nominal included (as mirt). $W_j$ is
   floored at $\sqrt{10^{-5}}$ whenever $W_j^2<10^{-5}$ (mirt's guard at extreme $\hat\theta$).
   Item statistics are computed within each group in a multiple-group model (as mirt's
   `itemfit()` on `multipleGroup` fits); weighted fits raise (as S-X2).
5. **Differences from mirt and validation**: (a) person `z_outfit` uses $n^2$ (answered items)
   under the square root; mirt's `personfit()` uses $J^2$ (all items) and takes an absolute
   value, which differs only for rows with missing responses. Item `z_outfit` uses each item's
   respondent count $N_j$; mirt's `itemfit()` writes `colSums((C / W^2) / N^2)` with `N` a
   length-$J$ vector, which R recycles down the rows instead of across the columns, so its
   `z.outfit` is off when the items have different counts (missing responses) and right
   otherwise. (b) A non-finite $\hat\theta$ (ML on an all-lowest or
   all-highest pattern, all-missing rows) gives NaN statistics and is skipped in item
   statistics; mirt replaces $\pm\infty$ by the most extreme finite estimate $\pm0.1$.
   (c) mirt sets a picked probability below machine epsilon to 1 in $l_0$; flexpyirt keeps the
   log-probability. V53 compares $l_z$ and person infit / outfit with mirt's `personfit()`,
   person and item $z_{\mathrm{out}}$ with the formulas above on mirt's own residual moments
   (and with mirt's values on complete data), item infit / outfit with
   `itemfit(fit_stats = "infit")`, lz, infit and outfit for a Rasch model
   with `sirt::personfit.stat()`, and $l_z^*$ with an R recomputation from the formulas of
   item 3 (`probtrace()` with central differences, `iteminfo()`; compared for
   $|\hat\theta|\le10$, beyond which the differences lose precision) and with a 2PL closed form
   and a simulation of its null distribution in the unit tests: no installed R package
   implements $l_z^*$ (`PerFit::lzstar` could not be installed, and it covers dichotomous items
   only). The $l_z^*$ entry is listed in the user documentation only if the owner accepts this
   validation (#108).

## 26. Many groups: the group-blocked E-step (#79)

1. **Problem.** Before #79 the E-step evaluated every pattern under every group: the log joint was
   `einsum("pjk,gjqk->pgq")` followed by selecting each pattern's own group, and the expected
   counts were `einsum("pg,pjk,pq->gjkq")` with the group one-hot. Both cost
   $O(P\,G\,J\,K\,Q)$ time and $O(P\,G\,Q)$ memory, $G$ times the work that is needed, and
   Oakes' cross term (§15.20) unrolled a Python loop over the groups inside jit, so its compile
   time and the number of pattern-chunk passes grew linearly in $G$. Each pattern belongs to
   exactly one group; the latent parameters were already stacked as `(G, L)` and the latent
   M-step was already one `vmap` over groups (§4.4, §14).
2. **Decision: group blocks.** `compile_problem` splits each group's (real) pattern rows into
   blocks of a common size $B$ and records them in a `GroupBlocks` pytree:
   `rows (N_B, B)` (indices into the flat pattern arrays; padding slots hold the sentinel $P$),
   `group (N_B,)` (the block's group) and `last (N_B,)` (whether the block is the last of its
   group). Blocks are ordered group-major. The flat `(P, J)` pattern layout of
   `CompiledProblem` is unchanged, so every caller outside the E-step (scoring, item fit,
   person fit, M2, JSI, robust SEs) keeps working; the block index is derived data and is built
   only in `compile_problem` (no code replaces `y`'s rows, `group` or `counts` of a compiled
   problem). Padding slots read a zero-count, all-missing row, so they contribute nothing.
   - **Block size**: $B\in\{8,16,\dots,1024\}$ minimizing
     $\sum_g\lceil P_g/B\rceil B + 64\,N_B(B)$ (padded rows plus a per-block overhead of 64
     rows), ties to the larger $B$. The total padding is at most $G(B-1)$ rows. (Padding $N_B$
     to a multiple of 8 was tried and dropped: with two groups it tripled the E-step work.)
   - **When**: every non-reduced model with $G\ge2$. Single-group models ($G=1$) and bifactor /
     two-tier models (§17.1, whose E-step is factorized over block grids) keep the flat path;
     `CompiledProblem.blocks` is `None` there.
3. **Blocked E-step.** For block $b$ with group $g=g(b)$ and rows $p\in b$,
   $$\lambda_{pq}=\log w_{gq}+\sum_f\sum_{j,k}\mathbf 1[y_{pj}=k]\log P^{(g)}_{jkq},\qquad
   h_{pq}=\mathrm{softmax}_q\lambda_{pq},$$
   $$\bar n_{gq}=\sum_{b:g(b)=g}\sum_{p\in b}c_ph_{pq},\qquad
   \bar r_{gjkq}=\sum_{b:g(b)=g}\sum_{p\in b}c_p h_{pq}\mathbf 1[y_{pj}=k].$$
   The item tables $\log P^{(g)}_{jkq}$ `(G, J_f, Q, K_f)` and latent log weights `(G, Q)` are
   computed once per E-step (unchanged); each block gathers its own group's slice and does one
   `(B, J_f K_f) x (J_f K_f, Q)` product per family. The blocks are folded with `lax.scan`,
   accumulating $\bar n$ and $\bar r$ in place at the block's group (`.at[g].add`), so peak
   memory is $O(G J K Q + B J K Q)$ independent of $N_B$; the per-block posteriors
   $h$ `(N_B, B, Q)` are scattered back to the flat `(P, Q)` rows (padding rows of the flat
   layout, which are in no block, get $\lambda_{pq}=0$ and count 0). The marginal
   log-likelihood, EAP, plausible values, the Hessian SE (autodiff through the scan) and the
   E-step all go through this path. Cost: $O(N_B B\,J K Q)$ instead of $O(P G J K Q)$.
   The sums run in a different order than the flat path, so results agree to rounding
   ($\sim10^{-12}$ relative), not bit for bit (regression test in `tests/test_many_groups.py`).
4. **Oakes' cross term** (§15.20) becomes one `lax.scan` over the blocks: each block adds its
   category co-occurrences to the running group's $C_{q}$ `(Q, D, D)` and its
   $\bar s_p\bar s_p^\top$ term to $T_2$; on the last block of a group (`last`) a `lax.cond`
   adds $\sum_q L_{gq}^\top C_q L_{gq}$ to $T_1$ and resets $C$. The group's score tables are
   gathered with the dynamic block group once per block (sub-chunks of a block reuse them, and
   sub-chunks of padding rows only are skipped by `lax.cond`), so the traced program no longer
   depends on $G$ (compile time constant in $G$).
5. **Latent distributions.** The latent parameters stay stacked `(G, L)` with `free_index`
   `(G, L)`; the latent M-step, Oakes' own Hessian and the latent score tables are `vmap`ped
   over groups. Every group shares one latent family and one static shape $L$ (§4.4, #75 for
   Davidian / mixture per group, §27); no Python loop over groups runs inside jit.
6. **API.** `Model.with_groups()` also accepts one positional mapping
   (`with_groups({"ref": Normal.standard()} | {s: Normal.free() for s in schools})`;
   insertion order defines the groups and the first is the reference), so non-identifier group
   labels and hundreds of groups need no `**` unpacking; anchors / invariance are declared once
   with `equal_across_groups()` as before, and `FitResult.latent` is already one DataFrame row
   per group.
7. **Out of scope / limits (follow-ups).** (a) Group-specific item parameters for many groups:
   each item's local vector has up to $G\,P_f$ slots and its Newton Hessian is dense
   (§14.2), so fully non-invariant items scale as $O(G^2)$ per item; anchored designs with
   invariant items are unaffected. (b) `se="crossprod"` / `"sandwich"` differentiate each
   pattern with respect to every group's tables (§17.5), so their chunk size shrinks as $1/G$.
   (c) Bifactor / two-tier multiple-group models keep the $G$-fold flat E-step.
8. **Validation.** Regression: the blocked and flat paths agree on small $G$ (estimates,
   log-likelihood, Oakes and Hessian SEs, EAP) to $10^{-10}$. V55: a 2PL with $G=10$ groups and
   anchored invariant items against mirt `multipleGroup` (VALIDATION.md). Benchmarks for
   $G=10,50,200$ in `benchmarks/` (compile time, warm time, peak memory) against mirt.

## 27. Davidian curves and normal mixtures per group (#75)

1. **Scope.** `DavidianCurve` (§4.4, §16.4) and `NormalMixture` (§4.4, §16.7) become valid latent
   distributions of `with_groups()` models: each group has its own mean, variance and shape
   (Davidian angles $\boldsymbol\phi_g$; mixture raw means, raw variances and logits), with one set
   of item parameters linked by anchors (or fixed by `fix_from()` under FIPC). Unidimensional
   models on equispaced grids only (as for one group).
2. **One family per model (decision).** Every group uses the same class with the same `degree`
   (Davidian) or `n_components` (mixture); mixing families (e.g. a `Normal` reference group and
   `DavidianCurve` focal groups) or degrees raises `SpecificationError`. Reasons: the latent block
   is stacked `(G, L)` with one static kind (§26.5), so mixing would need per-group padding of
   $L$ and a `lax.switch` that evaluates every family in every group; and the main use is already
   covered without it, because a Davidian curve with $\boldsymbol\phi=\mathbf 0$ **is** the normal
   density: a group that should stay normal in a Davidian model gets its `phi_*` rows fixed at 0
   in the parameter table (`table=`), which gives exactly $N(\mu_g,\sigma^2_g)$ on the grid. A
   mixture has no such exact normal member (a component weight would have to be 0), so a normal
   group in a mixture model is not offered.
3. **Identification.** As for `Normal` (§4.5, §14.5): by default the reference group is
   standardized (`mean = 0`, `var = 1` fixed, shape free, as in one group), and each focal group
   frees `mean` / `var` through `DavidianCurve(k, free_mean=True, free_var=True)` /
   `NormalMixture(k, free_mean=True, free_var=True)`, checked by the same anchor graph (the latent
   rows are named `mean` and `var` for every family). Each group's shape is identified by its own
   data as in one group (the analytic standardization of §16.4 / §16.7 separates shape from
   location and scale). FIPC: `fix_from()` fixes the items and every group may free its mean and
   variance.
4. **Core.** Nothing new: `LatentBlock` already holds `(G, L)` parameters with a static
   `(davidian_degree, mixture_components)`; the log weights, the latent M-step (masked Newton on
   $\sum_q\bar n_{gq}\log w_{gq}$), Oakes' own Hessian and the latent score tables are `vmap`ped
   over groups, and the E-step is group-blocked for $G\ge2$ (§26). EM stays monotone and Oakes'
   identity exact per group.
5. **Outer layer.** `FitResult.latent` gives one row per group (mixtures with the standardized
   components per group); the degenerate-mixture warning (§16.7) runs per group and names it;
   EAP / ML / WLE / PV scoring use the respondent's group density (MAP stays unavailable, §16.4,
   §16.7); `item_fit()` (S-X2, Orlando & Thissen, 2000) uses each group's quadrature weights
   $w_{gq}$ in the Lord–Wingersky expected table, which only needs the weights, so it is enabled
   for Davidian and mixture models with one or several groups (the free latent parameters of the
   group, shape included, are subtracted from the degrees of freedom as for `Normal`); `link()`
   with default weights integrates over the chosen group's fitted density (unchanged);
   `simulate_responses()` draws from each group's density (unchanged); JSON stores each group's
   latent spec as for one group, so the format is unchanged (`schema_version` stays 8).
   SEs: `oakes` and `hessian`; `crossprod` / `sandwich` stay unavailable (§17.5). Still rejected
   for these families: `m2()`, `information()`, `EAPsum`.
6. **Validation (V56).** mirt `multipleGroup()` has no per-group Davidian with free mean and
   variance ("not currently supported") and its `dentype = "mixture-k"` is a latent-class model,
   not a per-group mixture density. As V19 and V22 do for one group, V56 fits the exact
   discretized model in mirt through `multipleGroup(customGroup = createGroup(...))` with the
   standardized density normalized on the grid (Davidian via `dcurver::ddc`; the mixture with each
   component normalized on the grid, then mixed), polishes mirt's solution by Newton steps on an
   independent R marginal log-likelihood, and records the R Richardson-Hessian SEs, mirt's Oakes
   SEs, EAPs computed in R from `probtrace()`, and S-X2 per group from `itemfit()` on single-group
   mirt objects with the same `customGroup` fixed at the group's estimates.

## References
- Ackerman, T. A. (1994). Using multidimensional item response theory to understand what items and tests are measuring. *Applied Measurement in Education*, 7(4), 255–278.
- Aitchison, J., & Shen, S. M. (1980). Logistic-normal distributions: Some properties and uses. *Biometrika*, 67(2), 261–272.
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.
- Asparouhov, T. (2005). Sampling weights in latent variable modeling. *Structural Equation Modeling*, 12(3), 411–434.
- Auerswald, M., & Moshagen, M. (2019). How to determine the number of factors to retain in exploratory factor analysis: A comparison of extraction methods under realistic conditions. *Psychological Methods*, 24(4), 468–491.
- Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series*, RR-81-20.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *JRSS B*, 57(1), 289–300.
- Berndt, E. R., Hall, B. H., Hall, R. E., & Hausman, J. A. (1974). Estimation and inference in nonlinear structural models. *Annals of Economic and Social Measurement*, 3(4), 653–665.
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM Journal on Control and Optimization*, 20(2), 221–246.
- Binder, D. A. (1983). On the variances of asymptotically normal estimators from complex surveys. *International Statistical Review*, 51(3), 279–292.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores* (pp. 397–479). Addison-Wesley.
- Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two or more nominal categories. *Psychometrika*, 37(1), 29–51.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431–444.
- Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
- Bolt, D. M., Cohen, A. S., & Wollack, J. A. (2001). A mixture item response model for multiple-choice data. *Journal of Educational and Behavioral Statistics*, 26(4), 381–409.
- Braeken, J., & van Assen, M. A. L. M. (2017). An empirical Kaiser criterion. *Psychological Methods*, 22(3), 450–466.
- Braun, H. I., & Holland, P. W. (1982). Observed-score test equating: A mathematical analysis of some ETS equating procedures. In P. W. Holland & D. B. Rubin (Eds.), *Test Equating* (pp. 9–49). Academic Press.
- Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika*, 75(4), 581–612.
- Candell, G. L., & Drasgow, F. (1988). An iterative procedure for linking metrics and assessing item bias in item response theory. *Applied Psychological Measurement*, 12(3), 253–260.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Chalmers, R. P. (2018). Model-based measures for detecting and quantifying response bias. *Psychometrika*, 83(3), 696–732.
- Chen, W.-H., & Thissen, D. (1997). Local dependence indexes for item pairs using item response theory. *Journal of Educational and Behavioral Statistics*, 22(3), 265–289.
- Christensen, K. B., Makransky, G., & Horton, M. (2017). Critical values for Yen's Q3: Identification of local dependence in the Rasch model using residual correlations. *Applied Psychological Measurement*, 41(3), 178–194.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- von Davier, M. (2005). *A general diagnostic model applied to language testing data* (ETS Research Report RR-05-16). Educational Testing Service.
- Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer.
- Drasgow, F., & Lissak, R. I. (1983). Modified parallel analysis: A procedure for examining the latent dimensionality of dichotomously scored item responses. *Journal of Applied Psychology*, 68(3), 363–373.
- Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness measurement with polychotomous item response models and standardized indices. *British Journal of Mathematical and Statistical Psychology*, 38(1), 67–86.
- Drezner, Z., & Wesolowsky, G. O. (1990). On the computation of the bivariate normal integral. *Journal of Statistical Computation and Simulation*, 35(1–2), 101–107.
- Edwards, M. C., Houts, C. R., & Cai, L. (2018). A diagnostic procedure to detect departures from local independence in item response theory models. *Psychological Methods*, 23(1), 138–149.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Froelich, A. G., & Habing, B. (2008). Conditional covariance-based subtest selection for DIMTEST. *Applied Psychological Measurement*, 32(2), 138–155.
- Garrido, L. E., Abad, F. J., & Ponsoda, V. (2013). A new look at Horn's parallel analysis with ordinal variables. *Psychological Methods*, 18(4), 454–474.
- Genz, A. (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities. *Statistics and Computing*, 14(3), 251–260.
- Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor analysis. *Psychometrika*, 57(3), 423–436.
- Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis methodology for selecting the correct number of factors to retain. *Educational and Psychological Measurement*, 55(3), 377–393.
- Golino, H. F., & Epskamp, S. (2017). Exploratory graph analysis: A new approach for estimating the number of dimensions in psychological research. *PLoS ONE*, 12(6), e0174035.
- Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules. *Mathematics of Computation*, 23(106), 221–230.
- González, J., Wiberg, M., & von Davier, A. A. (2016). A note on the Poisson's binomial distribution in item response theory. *Applied Psychological Measurement*, 40(4), 302–310.
- Goretzko, D., & Bühner, M. (2020). One model to rule them all? Using machine learning algorithms to determine the number of factors in exploratory factor analysis. *Psychological Methods*, 25(6), 776–786.
- Green, B. F., Bock, R. D., Humphreys, L. G., Linn, R. L., & Reckase, M. D. (1984). Technical guidelines for assessing computerized adaptive tests. *Journal of Educational Measurement*, 21(4), 347–360.
- Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research*, 22(3), 144–149.
- Hannan, E. J., & Quinn, B. G. (1979). The determination of the order of an autoregression. *JRSS B*, 41(2), 190–195.
- Hanson, B. A., & Brennan, R. L. (1990). An investigation of classification consistency indexes estimated under alternative strong true score models. *Journal of Educational Measurement*, 27(4), 345–359.
- Harman, H. H., & Jones, W. H. (1966). Factor analysis by minimizing residuals (minres). *Psychometrika*, 31(3), 351–368.
- Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel–Haenszel procedure. In H. Wainer & H. I. Braun (Eds.), *Test Validity* (pp. 129–145). Erlbaum.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Hong, Y. (2013). On computing the distribution function for the Poisson binomial distribution. *Computational Statistics & Data Analysis*, 59, 41–51.
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30(2), 179–185.
- Houts, C. R., & Cai, L. (n.d.). *flexMIRT user's manual: Flexible multilevel multidimensional item analysis and test scoring* (versions 3.5–3.7). Vector Psychometric Group.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 221–233). University of California Press.
- Humphreys, L. G., & Ilgen, D. R. (1969). Note on a criterion for the number of common factors. *Educational and Psychological Measurement*, 29(3), 571–578.
- Huynh, H., & Meyer, P. (2010). Use of robust z in detecting unstable items in item response theory models. *Practical Assessment, Research & Evaluation*, 15(2).
- Jang, E. E., & Roussos, L. (2007). An investigation into the dimensionality of TOEFL using conditional covariance-based nonparametric approach. *Journal of Educational Measurement*, 44(1), 1–21.
- Jodoin, M. G., & Gierl, M. J. (2001). Evaluating type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. *Applied Measurement in Education*, 14(4), 329–349.
- Jöreskog, K. G. (1967). Some contributions to maximum likelihood factor analysis. *Psychometrika*, 32(4), 443–482.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Kim, S., & Lee, W.-C. (2006). An extension of four IRT linking methods for mixed-format tests. *Journal of Educational Measurement*, 43(3), 193–213.
- Köhler, C., Robitzsch, A., & Hartig, J. (2020). A bias-corrected RMSD item fit statistic: An evaluation and comparison to alternatives. *Journal of Educational and Behavioral Statistics*, 45(3), 251–273.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking: Methods and Practices* (3rd ed.). Springer.
- Kopf, J., Zeileis, A., & Strobl, C. (2015). Anchor selection strategies for DIF analysis: Review, assessment, and new approaches. *Educational and Psychological Measurement*, 75(1), 22–56.
- Landis, J. R., Heyman, E. R., & Koch, G. G. (1978). Average partial association in three-way contingency tables: A review and discussion of alternative tests. *International Statistical Review*, 46(3), 237–254.
- Lathrop, Q. N., & Cheng, Y. (2013). Two approaches to estimation of classification accuracy rate under item response theory. *Applied Psychological Measurement*, 37(3), 226–241.
- Le Cam, L. (1960). An approximation theorem for the Poisson binomial distribution. *Pacific Journal of Mathematics*, 10(4), 1181–1197.
- Lee, W.-C. (2010). Classification consistency and accuracy for complex assessments using item response theory. *Journal of Educational Measurement*, 47(1), 1–17.
- Li, Y. H., & Lissitz, R. W. (2000). An evaluation of the accuracy of multidimensional IRT linking. *Applied Psychological Measurement*, 24(2), 115–138.
- Liu, I.-M., & Agresti, A. (1996). Mantel–Haenszel-type inference for cumulative odds ratios with a stratified ordinal response. *Biometrics*, 52(4), 1223–1234.
- Livingston, S. A., & Lewis, C. (1995). Estimating the consistency and accuracy of classifications based on test scores. *Journal of Educational Measurement*, 32(2), 179–197.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Erlbaum.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Lawrence Erlbaum.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings." *Applied Psychological Measurement*, 8(4), 453–461.
- Lorenzo-Seva, U., Timmerman, M. E., & Kiers, H. A. L. (2011). The Hull method for selecting the number of common factors. *Multivariate Behavioral Research*, 46(2), 340–364.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis and determination of sample size for covariance structure modeling. *Psychological Methods*, 1(2), 130–149.
- Mantel, N. (1963). Chi-square tests with one degree of freedom: Extensions of the Mantel–Haenszel procedure. *Journal of the American Statistical Association*, 58(303), 690–700.
- Mantel, N., & Haenszel, W. (1959). Statistical aspects of the analysis of data from retrospective studies of disease. *Journal of the National Cancer Institute*, 22(4), 719–748.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information estimation and goodness-of-fit testing in $2^n$ contingency tables: A unified framework. *Journal of the American Statistical Association*, 100(471), 1009–1020.
- Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit testing in multidimensional contingency tables. *Psychometrika*, 71(4), 713–732.
- Maydeu-Olivares, A., & Joe, H. (2014). Assessing approximate fit in categorical data analysis. *Multivariate Behavioral Research*, 49(4), 305–328.
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley.
- Meade, A. W. (2010). A taxonomy of effect size measures for the differential functioning of items and scales. *Journal of Applied Psychology*, 95(4), 728–743.
- Mislevy, R. J. (1984). Estimating latent distributions. *Psychometrika*, 49(3), 359–381.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56(2), 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Nandakumar, R., & Stout, W. (1993). Refinements of Stout's procedure for assessing latent trait unidimensionality. *Journal of Educational Statistics*, 18(1), 41–68.
- O'Connor, B. P. (2000). SPSS and SAS programs for determining the number of components using parallel analysis and Velicer's MAP test. *Behavior Research Methods, Instruments, & Computers*, 32(3), 396–402.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Oehlert, G. W. (1992). A note on the delta method. *The American Statistician*, 46(1), 27–29.
- Ogasawara, H. (2001). Standard errors of item response theory equating/linking by response function methods. *Applied Psychological Measurement*, 25(1), 53–67.
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation coefficient. *Psychometrika*, 44(4), 443–460.
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement*, 24(1), 50–64.
- Orlando, M., & Thissen, D. (2003). Further investigation of the performance of S-X2: An item fit index for use with dichotomous item response theory models. *Applied Psychological Measurement*, 27(4), 289–298.
- Oshima, T. C., Davey, T. C., & Lee, K. (2000). Multidimensional linking: Four practical approaches. *Journal of Educational Measurement*, 37(4), 357–373.
- Owen, D. B. (1956). Tables for computing bivariate normal probabilities. *The Annals of Mathematical Statistics*, 27(4), 1075–1090.
- Penfield, R. D. (2001). Assessing differential item functioning among multiple groups: A comparison of three Mantel–Haenszel procedures. *Applied Measurement in Education*, 14(3), 235–259.
- Penfield, R. D., & Algina, J. (2003). Applying the Liu–Agresti estimator of the cumulative common odds ratio to DIF detection in polytomous items. *Journal of Educational Measurement*, 40(4), 353–370.
- Phillips, A., & Holland, P. W. (1987). Estimators of the variance of the Mantel–Haenszel log-odds-ratio estimate. *Biometrics*, 43(2), 425–431.
- Plackett, R. L. (1954). A reduction formula for normal multivariate integrals. *Biometrika*, 41(3–4), 351–360.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.
- Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric item characteristic curve estimation. *Psychometrika*, 56(4), 611–630.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory*. Springer.
- Reckase, M. D., & McKinley, R. L. (1991). The discriminating power of items that measure more than one dimension. *Applied Psychological Measurement*, 15(4), 361–373.
- Revelle, W. (2024). *psych: Procedures for psychological, psychometric, and personality research* (R package). Northwestern University.
- Roberts, J. S., Donoghue, J. R., & Laughlin, J. E. (2000). A general item response theory model for unfolding unidimensional polytomous responses. *Applied Psychological Measurement*, 24(1), 3–32.
- Robins, J., Breslow, N., & Greenland, S. (1986). Estimators of the Mantel–Haenszel variance consistent in both sparse data and large-strata limiting models. *Biometrics*, 42(2), 311–323.
- Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods*, 21(2), 137–150.
- Rost, J. (1990). Rasch models in latent classes: An integration of two approaches to item analysis. *Applied Psychological Measurement*, 14(3), 271–282.
- Rudner, L. M. (2001). Computing the expected proportions of misclassified examinees. *Practical Assessment, Research, and Evaluation*, 7, Article 14.
- Rudner, L. M. (2005). Expected classification accuracy. *Practical Assessment, Research, and Evaluation*, 10, Article 13.
- Ruscio, J., & Kaczetow, W. (2008). Simulating multivariate nonnormal data using an iterative algorithm. *Multivariate Behavioral Research*, 43(3), 355–381.
- Ruscio, J., & Roche, B. (2012). Determining the number of factors to retain in an exploratory factor analysis using comparison data of known factorial structure. *Psychological Assessment*, 24(2), 282–292.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464.
- Sclove, S. L. (1987). Application of model-selection criteria to some problems in multivariate analysis. *Psychometrika*, 52(3), 333–343.
- Sinharay, S. (2016). Asymptotically correct standardization of person-fit statistics beyond dichotomous items. *Psychometrika*, 81(4), 992–1013.
- Skinner, C. J. (1989). Domain means, regression and multivariate analysis. In C. J. Skinner, D. Holt & T. M. F. Smith (Eds.), *Analysis of Complex Surveys* (pp. 59–87). Wiley.
- Snijders, T. A. B. (2001). Asymptotic null distribution of person fit statistics with estimated person parameter. *Psychometrika*, 66(3), 331–342.
- Somes, G. W. (1986). The generalized Mantel–Haenszel statistic. *The American Statistician*, 40(2), 106–108.
- Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement*, 7(2), 201–210.
- Stout, W. (1987). A nonparametric approach for assessing latent trait unidimensionality. *Psychometrika*, 52(4), 589–617.
- Stout, W. (1990). A new item response theory modeling approach with applications to unidimensionality assessment and ability estimation. *Psychometrika*, 55(2), 293–325.
- Stout, W., Froelich, A. G., & Gao, F. (2001). Using resampling methods to produce an improved DIMTEST procedure. In A. Boomsma, M. A. J. van Duijn, & T. A. B. Snijders (Eds.), *Essays on Item Response Theory* (pp. 357–375). Springer.
- Swaminathan, H., & Gifford, J. A. (1986). Bayesian estimation in the three-parameter logistic model. *Psychometrika*, 51, 589–601.
- Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning using logistic regression procedures. *Journal of Educational Measurement*, 27(4), 361–370.
- Thissen, D., & Wainer, H. (Eds.). (2001). *Test Scoring*. Lawrence Erlbaum Associates.
- Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of Polytomous Item Response Theory Models* (pp. 43–75). Routledge.
- Thissen, D., Pommerich, M., Billeaud, K., & Williams, V. S. L. (1995). Item response theory for scores on tests including polytomous items with ordered responses. *Applied Psychological Measurement*, 19(1), 39–49.
- Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of differential item functioning using the parameters of item response models. In P. W. Holland & H. Wainer (Eds.), *Differential Item Functioning* (pp. 67–113). Erlbaum.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Velicer, W. F. (1976). Determining the number of components from the matrix of partial correlations. *Psychometrika*, 41(3), 321–327.
- Velicer, W. F., Eaton, C. A., & Fava, J. L. (2000). Construct explication through factor or component analysis: A review and evaluation of alternative procedures for determining the number of factors or components. In R. D. Goffin & E. Helmes (Eds.), *Problems and Solutions in Human Assessment* (pp. 41–71). Kluwer.
- Volkova, A. Y. (1996). A refinement of the central limit theorem for sums of independent random indicators. *Theory of Probability and Its Applications*, 40(4), 791–794.
- Wald, A. (1943). Tests of statistical hypotheses concerning several parameters when the number of observations is large. *Transactions of the American Mathematical Society*, 54(3), 426–482.
- Wang, C. (2015). On latent trait estimation in multidimensional compensatory item response models. *Psychometrika*, 80(2), 428–449.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427–450.
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1–25.
- Williams, R. L. (2000). A note on robust variance estimation for cluster-correlated data. *Biometrics*, 56(2), 645–646.
- Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *The Annals of Mathematical Statistics*, 9(1), 60–62.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Wothke, W. (1993). Nonpositive definite matrices in structural modeling. In K. A. Bollen & J. S. Long (Eds.), *Testing Structural Equation Models* (pp. 256–293). Sage.
- Wright, B. D., & Masters, G. N. (1982). *Rating Scale Analysis*. MESA Press.
- Wu, M. L., Adams, R. J., Wilson, M. R., & Haldane, S. A. (2007). *ACER ConQuest Version 2: Generalised Item Response Modelling Software*. Australian Council for Educational Research.
- Yen, W. M. (1981). Using simulation results to choose a latent trait model. *Applied Psychological Measurement*, 5(2), 245–262.
- Yen, W. M. (1984). Effects of local item dependence on the fit and equating performance of the three-parameter logistic model. *Applied Psychological Measurement*, 8(2), 125–145.
- Zhang, D., & Davidian, M. (2001). Linear mixed models with flexible distributions of random effects for longitudinal data. *Biometrics*, 57, 795–802.
- Zhang, J. (2007). Conditional covariance theory and DETECT for polytomous items. *Psychometrika*, 72(1), 69–91.
- Zhang, J., & Stout, W. (1999). The theoretical DETECT index of dimensionality and its application to approximate simple structure. *Psychometrika*, 64(2), 213–249.
- Zieky, M. (1993). Practical questions in the use of DIF statistics in test development. In P. W. Holland & H. Wainer (Eds.), *Differential Item Functioning* (pp. 337–347). Erlbaum.
- Zumbo, B. D. (1999). *A Handbook on the Theory and Methods of Differential Item Functioning (DIF): Logistic Regression Modeling as a Unitary Framework for Binary and Likert-Type (Ordinal) Item Scores*. Directorate of Human Resources Research and Evaluation, Department of National Defense (Canada).
- Zumbo, B. D., & Thomas, D. R. (1997). *A Measure of Effect Size for a Model-Based Approach for Studying DIF*. Edgeworth Laboratory for Quantitative Behavioral Science, University of Northern British Columbia.
- Zwick, R. (2012). A review of ETS differential item functioning assessment procedures: Flagging rules, minimum sample size requirements, and criterion refinement. *ETS Research Report Series*, 2012(1), i–30.
