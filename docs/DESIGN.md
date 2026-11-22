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
- Later: normal mixtures. Can be added without touching the Core by following the `LatentDist`
  protocol (§6).

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
- Gauss–Hermite is available for Normal latent distributions (`Quadrature.gauss_hermite(n)`, §15.44).
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
| Nominal | $a, \mathrm{ak}_1..\mathrm{ak}_{K-2}, d_1..d_{K-1}$ ($\mathrm{ak}_0=0, \mathrm{ak}_{K-1}=K-1, d_0=0$; §16.6) | M5 |

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
    Multidimensional, EH and missing-data cases raise `SpecificationError` until they are
    validated (mirt imputes missing data and ignores EH weights for S-X2); multiple groups: §15.45.
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
   for dichotomous items this is exactly $M_2$) — for single-group, unidimensional, complete-data
   MLE fits with a `Normal` latent distribution (multidimensional, multiple-group,
   `EmpiricalHistogram`, missing-data, or MAP fits raise `SpecificationError`).
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
   averages.
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
6. **Nominal response model (`Nominal(n_cat)`, `NominalKernel(n_cat, n_factors=1)`, V23)**:
   Bock's (1972) nominal response model in Thissen, Cai & Bock's (2010) overall-slope ×
   scoring-function parameterization, matching R mirt's `itemtype = "nominal"` (Chalmers, 2012):
   $$P(y=k\mid\boldsymbol\theta)=\frac{\exp\big(\mathrm{ak}_k\,\mathbf a^\top\boldsymbol\theta+d_k\big)}{\sum_{c=0}^{K-1}\exp\big(\mathrm{ak}_c\,\mathbf a^\top\boldsymbol\theta+d_c\big)},\qquad k=0,\dots,K-1,\quad \mathrm{ak}_0=0,\ \mathrm{ak}_{K-1}=K-1,\ d_0=0,$$
   for categories `0..K-1` (`n_cat >= 2` validated) and $D\in\{1,2,3\}$ (evaluated with
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
     flexpyirt starts `Nominal(n_cat)` at the GPCM starting point: $a_j^{(0)}$ from the item–rest
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
     `irt_dep_indices = ()` (no `b` / `bk` columns in `FitResult.items`), and `Model.fix_from()`
     supports `Nominal` items with `parameterization="slope-intercept"` while raising
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

## References
- Barton, M. A., & Lord, F. M. (1981). An upper asymptote for the three-parameter logistic item-response model. *ETS Research Report Series*, RR-81-20.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *JRSS B*, 57(1), 289–300.
- Bertsekas, D. P. (1982). Projected Newton methods for optimization problems with simple constraints. *SIAM Journal on Control and Optimization*, 20(2), 221–246.
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical Theories of Mental Test Scores* (pp. 397–479). Addison-Wesley.
- Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two or more nominal categories. *Psychometrika*, 37(1), 29–51.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika*, 46, 443–459.
- Bock, R. D., & Lieberman, M. (1970). Fitting a response model for n dichotomously scored items. *Psychometrika*, 35, 179–197.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement*, 6(4), 431–444.
- Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
- Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software*, 48(6).
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27–38.
- Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules. *Mathematics of Computation*, 23(106), 221–230.
- Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research*, 22(3), 144–149.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Kim, S. (2006). A comparative study of IRT fixed parameter calibration methods. *Journal of Educational Measurement*, 43, 355–381.
- Kim, S., & Lee, W.-C. (2006). An extension of four IRT linking methods for mixed-format tests. *Journal of Educational Measurement*, 43(3), 193–213.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking: Methods and Practices* (3rd ed.). Springer.
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings." *Applied Psychological Measurement*, 8(4), 453–461.
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *JRSS B*, 44, 226–233.
- MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis and determination of sample size for covariance structure modeling. *Psychological Methods*, 1(2), 130–149.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika*, 47, 149–174.
- Maydeu-Olivares, A., & Joe, H. (2005). Limited- and full-information estimation and goodness-of-fit testing in $2^n$ contingency tables: A unified framework. *Journal of the American Statistical Association*, 100(471), 1009–1020.
- Maydeu-Olivares, A., & Joe, H. (2006). Limited information goodness-of-fit testing in multidimensional contingency tables. *Psychometrika*, 71(4), 713–732.
- Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*, 51, 177–195.
- Mislevy, R. J. (1991). Randomization-based inference about latent variables from complex samples. *Psychometrika*, 56(2), 177–196.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16, 159–176.
- Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm. *JRSS B*, 61, 479–482.
- Oehlert, G. W. (1992). A note on the delta method. *The American Statistician*, 46(1), 27–29.
- Ogasawara, H. (2001). Standard errors of item response theory equating/linking by response function methods. *Applied Psychological Measurement*, 25(1), 53–67.
- Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for dichotomous item response theory models. *Applied Psychological Measurement*, 24(1), 50–64.
- Orlando, M., & Thissen, D. (2003). Further investigation of the performance of S-X2: An item fit index for use with dichotomous item response theory models. *Applied Psychological Measurement*, 27(4), 289–298.
- Reckase, M. D. (2009). *Multidimensional Item Response Theory*. Springer.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph*, 17.
- Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement*, 7(2), 201–210.
- Swaminathan, H., & Gifford, J. A. (1986). Bayesian estimation in the three-parameter logistic model. *Psychometrika*, 51, 589–601.
- Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of Polytomous Item Response Theory Models* (pp. 43–75). Routledge.
- Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of differential item functioning using the parameters of item response models. In P. W. Holland & H. Wainer (Eds.), *Differential Item Functioning* (pp. 67–113). Erlbaum.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for accelerating the convergence of any EM algorithm. *Scand. J. Statist.*, 35, 335–353.
- Wang, C. (2015). On latent trait estimation in multidimensional compensatory item response models. *Psychometrika*, 80(2), 428–449.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika*, 54(3), 427–450.
- Woods, C. M. (2007). Empirical histograms in item response theory with ordinal data. *Educational and Psychological Measurement*, 67, 73–87.
- Woods, C. M., & Lin, N. (2009). Item response theory with estimation of the latent density using Davidian curves. *Applied Psychological Measurement*, 33, 102–117.
- Zhang, D., & Davidian, M. (2001). Linear mixed models with flexible distributions of random effects for longitudinal data. *Biometrics*, 57, 795–802.
