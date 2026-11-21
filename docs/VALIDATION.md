# Validation plan (numerical agreement with R mirt)

English | [日本語](VALIDATION.ja.md)

## Approach
- Reference values come from R `mirt` and are committed as JSON in `tests/reference/`. CI does not
  run R.
- Generator scripts are `validation/r/<case>.R`. Each writes the equivalent of `sessionInfo()`
  (R and mirt versions) into the JSON.
- Comparisons use **the same conditions**: the same quadrature (e.g. 61 equally spaced points on
  $[-6,6]$), the same convergence criterion (a strict `TOL` on the mirt side, e.g. 1e-8), no
  acceleration, the same priors.
- Data are both simulated (fixed seeds) and public (e.g. mirt's bundled `LSAT7`, `Science`,
  `SAT12`).

## Tolerances (initial values, revised as needed)
| Quantity | Tolerance (max. absolute difference) |
|---|---|
| Log-likelihood | 1e-6 (relative) |
| Item parameters (internal scale $a, d$) | 1e-4 |
| Latent-distribution parameters (mean, SD) | 1e-4 |
| Standard errors | 1e-3 (relative) |
| EAP | 1e-5 |
| MAP / ML / WLE ($\theta$) | 1e-4 |
| MAP / ML / WLE (SE) | 1e-3 (relative) |

## Cases (by milestone)
| ID | Contents | mirt call | MS |
|---|---|---|---|
| V1 | 2PL, single group (LSAT7) | `mirt(LSAT7, 1, "2PL", SE=TRUE, SE.type="Oakes")` | M1 |
| V2 | Rasch, single group | `mirt(..., "Rasch")` | M1 |
| V3 | GRM, single group (Science) | `mirt(Science, 1, "graded")` | M2 |
| V4 | 2PL, two groups, partial anchors, focal group free | `multipleGroup(..., invariance=c(anchors, "free_means","free_var"))` | M2 |
| V5 | FIPC (fixed items + latent distribution estimated) | `pars='values'` with `est=FALSE` | M2 |
| V6 | EH under FIPC (single group, two groups) | `mirt(..., dentype = "EH", pars = <bank fixed>)` / `multipleGroup(...)` | M3 |
| V7 | 3PL + priors (MAP) | equivalent of `PRIOR = (1-30, g, expbeta, ...)` in `mirt.model` | M3 |
| V8 | GPCM and PCM (Science + simulated) | `mirt(..., "gpcm")` / `mirt(..., "Rasch")` (polytomous) | M3 |
| V9 | Missing data (MAR) | data with NA | M2 |
| V10 | SE (Oakes / crossprod / sandwich) | `SE.type=` | M1–M3 |
| V11 | 3PL and 4PL, single group (simulated) | `mirt(..., "3PL")` / `mirt(..., "4PL")` | M3 |
| V12 | Display IRT thresholds and delta-method SEs (2PL, Rasch, GRM, GPCM, PCM, 2-group 2PL) | `coef(m, IRTpars = TRUE, printSE = TRUE)` | M4 |
| V13 | Ability scoring: MAP, ML, WLE (2PL, GRM, GPCM, 3PL, 2-group, missing) | `fscores(m, method = "MAP"/"ML"/"WLE", response.pattern = ..., full.scores.SE = TRUE)` | M4 |
| V15 | S-X2 item fit (GRM, GPCM, PCM with free variance, 3PL, FIPC 2PL; `mincell` 1 and 3) | `itemfit(m, fit_stats = "S_X2")` | M5 |
| V16 | Equality constraints across items (LSAT7 common slope; 2PL + graded with cross-model constraints; two groups with anchors and a common slope) | `mirt(..., mirt.model("... CONSTRAIN = (1-5, a1)"))` / `multipleGroup(...)` | M5 |
| V17 | DIF likelihood-ratio tests on the V4 data (`add` for `a`+`d` and for `d`; `drop` for `a`+`d` from the fully invariant model) | `DIF(mg, which.par = ..., scheme = ...)` | M5 |
| V14 | Confirmatory 2–3D MIRT (2PL simple, 2PL cross-loading, GRM, 3D 2PL) | `mirt(..., mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes")` | M4 |
| V23 | Nominal response model (Science; simulated with unordered/negative scoring coefficients; nominal + 2PL mixed; mostly non-monotone items; two-factor confirmatory) | `mirt(..., "nominal")` with `pars` starting at the GPCM point (or the best alternative start); `itemfit(fit_stats = "S_X2")` | M5 |

## mirt settings and known differences (found in M1)
- **M-step optimizer**: references are generated with `optimizer = "NR"`. mirt's default M-step
  optimizers (`nlminb` for Rasch, BFGS for 2PL) shrink their iteration cap as the log-likelihood
  change gets small; the M-step then barely moves and the `TOL` test passes before EM has converged
  (the LSAT7 Rasch variance was off by about 2e-5, regardless of `TOL`).
- **Latent M-step**: mirt updates the variance (and mean) with closed-form moments. That does not
  maximize the discretized marginal likelihood (the log-likelihood mirt itself reports), so the
  Rasch variance differs slightly from flexpyirt's exact M-step (DESIGN §4.4): a variance difference
  of 7.9e-5 in a simulation with N=2000, where flexpyirt's log-likelihood is 5.6e-7 higher. A
  closed-form update reproduces mirt's value to 1e-10. This is within tolerance (1e-4), so the
  comparison is run as usual, plus a check that "flexpyirt's log-likelihood ≥ mirt's".
- **How latent-distribution parameters are compared (Decision)**: flexpyirt has only the exact
  M-step (DESIGN §4.4). The difference from mirt comes from the grid cut-off, so where the cut-off is
  not negligible (e.g. V4's focal group: a mean away from 0 or a variance above 2), **both mirt and
  flexpyirt use a wider grid (e.g. `theta_lim = c(-8, 8)` with proportionally more points)** so the
  cut-off difference is well below tolerance. The default grid (61 points on $[-6,6]$) is used only
  where the cut-off difference is below 1e-5. In every case, check that "flexpyirt's
  log-likelihood ≥ mirt's".
- **Passing `theta_lim` (found in M2)**: for `mirt()` / `multipleGroup()` it goes inside
  `technical = list(theta_lim = ...)`. As a top-level argument it is silently ignored and the default
  $[-6,6]$ is used (`fscores()` does take it at the top level). The reference JSON records the grid
  actually used in `meta.grid_check`.
- **Comparing log-likelihoods**: the log-likelihood mirt reports comes from the last iteration's
  E-step and need not equal the value at the final estimates. The "flexpyirt ≥ mirt" check compares
  with the log-likelihood flexpyirt evaluates with every parameter fixed at mirt's estimates (from
  M2 on).
- **M2 results**: V3 (GRM), V4 (two groups, 5 anchor items, focal group free, 81 points on
  $[-8,8]$), V5 (FIPC, same grid) and V9 (15% MCAR) all agree with mirt to ~1e-8 in parameters,
  log-likelihood and SEs. This also confirms that with the same grid on both sides, the difference
  between the exact M-step and mirt's closed-form update (from the cut-off) is negligible.
  Tests: `tests/test_mirt_m2.py`. Reference generator: `validation/r/m2_reference.R`.
- **3PL / 4PL asymptote scale and SE conversion (V11, found in M3)**: mirt optimizes $g$ and $u$
  on the logit scale, and `coef(m, printSE = TRUE)` reports `logit(g)` / `logit(u)` with
  logit-scale SEs (even with `rawug`). flexpyirt works on the natural scale (DESIGN §15.2), so
  `validation/r/m3_reference.R` converts mirt's estimates with `plogis()` and its SEs with the
  delta method, $SE_g = g(1-g)\,SE_{\mathrm{logit}(g)}$ (and likewise for $u$), while keeping the
  raw logit-scale values under `raw_logit`. At an interior ML solution the observed information
  transforms identically with the Jacobian.
- **Empirical histogram (V6, M3)**: mirt's `dentype = "EH"` re-estimates the histogram every
  cycle without standardizing it, so with all items free the scale is unidentified and EM drifts
  (a free 2PL calibration did not converge in 50,000 cycles and its histogram variance drifted to
  1.5). flexpyirt standardizes the reference group instead (DESIGN §15.34), which mirt does not
  offer, so V6 uses FIPC, where fixed items set the scale and neither program standardizes:
  V6a ($J=20$, items 1–10 fixed, $N=3000$, standardized $\chi^2_4$ trait) and V6b (two groups of
  3000, the same bank fixed in both, items 11–20 equal across groups, focal trait
  $0.5-1.2\,\chi^2_4$), 61 points on $[-6,6]$. On a fine grid the EH likelihood is nearly flat in
  many histogram directions and its maximum is spiky, so EM creeps (mirt needed 69,683 and 62,632
  cycles at `TOL = 1e-9`), and mirt's stopping rule ignores the histogram. Final weights are
  therefore not compared node by node. At mirt's solution: our log-likelihood equals mirt's to
  1.5e-15; one flexpyirt EM step moves items by 1e-9 (mirt's `TOL`) and the histogram by
  4e-7 (mirt reports it one E-step behind); Hessian and Oakes SEs (histogram held fixed) match
  mirt's Richardson SEs to 1.1e-7. A full flexpyirt fit (SQUAREM, `tol = 1e-6`) converges in
  7,162 / 2,837 iterations, ends above mirt's log-likelihood (by 1.6e-3 / 7.3e-4), with items
  within 5e-5 of mirt's. mirt 1.46.1's Oakes SEs for EH are 0.1–1.3% below its own Richardson
  SEs (cause not confirmed), and its Richardson SEs fail for the two-group EH model, so V6b checks
  Oakes == Hessian (4e-16). Tests: `tests/test_mirt_v6_eh.py`. Reference generator:
  `validation/r/m3_eh_reference.R`.
- **V11 simulation design and mirt `NR` starting-value trap (M3)**: unregularized ML for 3PL/4PL
  easily produces finite-sample boundary solutions ($g\to 0$ or $u\to 1$, with a singular
  information matrix) when items are too easy/hard or slopes are too low for the asymptote tails to
  be observed in $N(0,1)$. In addition, on 4PL data, mirt's default starting slope (`a = 0.851`)
  causes `optimizer = "NR"` on the logit scale to overshoot `logit(g) ≈ -15` on early iterations
  and stay trapped at a strictly lower log-likelihood (because the logit-scale gradient
  $g(1-g)\,\partial Q/\partial g$ vanishes near 0). To validate an interior, converged global ML
  solution without loosening tolerances, V11a uses $J=15, N=5000$ with
  $a\sim\mathrm{lognormal}(0.4,0.2)$, $d\sim U(-1.5,0.5)$, $g\sim U(0.12,0.25)$; V11b uses
  $J=12, N=20000$ (at most $2^{12}=4096$ patterns) with $a\sim\mathrm{lognormal}(0.6,0.15)$,
  $d\sim U(-0.5,0.5)$, $g\sim U(0.10,0.22)$, $u\sim U(0.78,0.90)$, starting mirt at $a=2.0$ while
  flexpyirt runs from its own default starting values. Both cases converge cleanly with no
  `HeywoodWarning`: parameters agree to ~1e-8 (3PL: 5.4e-9, 4PL: 7.2e-8), log-likelihood to
  ~1e-15, Oakes SEs to ~1e-7 (3PL: 1.1e-8, 4PL: 1.9e-7), Hessian vs mirt Richardson SEs to
  ~6e-4 (3PL: 6.8e-5, 4PL: 5.6e-4, reflecting Richardson's finite-difference error while flexpyirt's
  Oakes and Hessian agree to ~1e-11), and EAP to ~1e-15.
  Tests: `tests/test_mirt_m3.py`. Reference generator: `validation/r/m3_reference.R`.
- **GPCM and PCM parameterization mapping and results (V8, M3)**: mirt's `itemtype = "gpcm"` (and
  `itemtype = "Rasch"` on polytomous data, which is the PCM with `a1 = 1` fixed and `COV_11` free)
  reports columns `a1`, `ak0..ak{K-1}` (fixed at `0..K-1`), `d0` (fixed at `0`), and `d1..d{K-1}`
  (free). `validation/r/m3_gpcm_reference.R` verifies those fixed constants and maps `a1` → `a` and
  `d1..d{K-1}` → `d1..d{K-1}`, omitting `ak*` and `d0`. Three cases are checked: V8a (`Science`
  recoded `0..3`, GPCM, 61 points on $[-6,6]$), V8b (simulated GPCM, $J=10, K=4, N=3000$, including
  an item with reversed step order, 61 points on $[-6,6]$), and V8c (simulated PCM with generating
  latent variance $1.5$, mirt `itemtype = "Rasch"`, 81 points on $[-8,8]$ so grid truncation is
  negligible as in V4/V5). All three agree with mirt well inside tolerance: item parameters to
  ~1.4e-8 (latent variance in V8c to 6.1e-10), log-likelihood to ~3.7e-15, Oakes SEs to ~1.3e-8,
  Hessian vs mirt Richardson SEs to ~1.1e-6 (flexpyirt Oakes vs Hessian to ~1.1e-11), and EAP to
  ~8.0e-15.
  Tests: `tests/test_mirt_m3_gpcm.py`. Reference generator: `validation/r/m3_gpcm_reference.R`.
- **SQUAREM agreement (M3)**: every mirt agreement suite (`tests/test_mirt_m1.py`,
  `tests/test_mirt_m2.py`, `tests/test_mirt_m3.py`, `tests/test_mirt_m3_gpcm.py`,
  `tests/test_mirt_m3_priors.py`) is parametrized over `BAEM(squarem=False)` and
  `BAEM(squarem=True)` against the same unaccelerated references. All 16 cases converge to the same
  fixed point within the documented tolerances (max item difference $\le 2.8\times 10^{-8}$ across
  2PL/GRM/GPCM/PCM/3PL/4PL/FIPC/2-group/missing/MAP, and $7.9\times 10^{-5}$ on V2's simulated
  Rasch latent variance from the exact vs. closed-form grid cut-off, matching plain EM) with
  monotone log-likelihoods (log posteriors under MAP) and 1.3×–5.0× fewer EM map evaluations (e.g.
  `v2_lsat7_rasch`: 121 → 24, `v4_twogroup_2pl`: 302 → 113, `v8_science_gpcm`: 251 → 96,
  `v11_4pl`: 531 → 272, `v7_2pl_map`: 80 → 35, `v7_3pl_map`: 64 → 37). Unit, Hypothesis
  monotonicity, safeguard, and reduction tests: `tests/test_m3_squarem.py`.
- **Priors and MAP verification against R mirt 1.46 (V7, M3)**: inspecting `mirt:::LL.Priors`,
  `mirt:::DerivativePriors`, `mirt:::BL.LL`, and running numerical stationarity/Hessian checks in R
  established both the exact objective and three implementation defects in mirt 1.46:
  1. **Objective (`mirt:::LL.Priors`) — matches DESIGN §4.7 exactly**: `norm(mu, sd)` evaluates
     `dnorm(x, mu, sd, log = TRUE)`, `lnorm(mu, sd)` evaluates `dlnorm(x, mu, sd, log = TRUE)`, and
     `expbeta(a, b)` evaluates `dbeta(plogis(logit_g), a, b, log = TRUE)` on the **natural scale
     with no logit Jacobian**. Fixing all parameters at an exact MAP estimate in mirt confirms
     `mirt@Fit$logLik` and `mirt@Fit$logPrior` agree with flexpyirt to $\le 10^{-14}$.
  2. **`expbeta` derivative bug in mirt 1.46**: `mirt:::DerivativePriors` (and C++ `computeDPars`)
     evaluates natural-scale Beta derivatives $(a-1)/g - (b-1)/(1-g)$ and
     $-(a-1)/g^2 - (b-1)/(1-g)^2$ and adds them directly to the **logit-scale** gradient and
     Hessian without the chain-rule factor $g(1-g)$. Consequently, `optimizer = "NR"` fails to
     converge with `expbeta`, gradient-based optimizers (`BFGS`, `L-BFGS-B`, `nlminb`) stop at a
     non-stationary point with strictly lower `logLik + logPrior` (deficit `4.89` and `max |Δg| = 0.095`
     in V7c), and mirt's `SE.type = "Oakes"` underestimates `g_se` by ~10×.
  3. **`lnorm` Hessian sign bug in mirt 1.46**: the `lnorm` gradient is exact (so `NR` finds the
     exact MAP estimate to ~1e-8), but `DerivativePriors` / `computeDPars` has a minus sign on the
     third term of the second derivative (`-(log(a)-mu)/(a^2*s^2)` instead of `+(log(a)-mu)/(a^2*s^2)`),
     making mirt's `Oakes` SEs differ from the true posterior Hessian SEs by ~5% (confirmed to
     `8.3e-10` against the sign-flipped formula). For `norm` priors, both gradient and Hessian in
     mirt are exact and `Oakes` SEs agree with the true posterior Hessian SEs to `9.6e-10`.
  4. **`SE.type = "Richardson"` omits priors**: `mirt:::BL.LL` computes only the marginal
     log-likelihood without calling `LL.Priors`, so mirt's Richardson SEs equal the
     **likelihood-only** information $-\nabla^2\ell(\hat\eta_{\text{MAP}})$ (confirmed to ~1e-8).

  Three V7 cases ($J=15, N=1000$, 61 points on $[-6,6]$) are validated in
  `tests/test_mirt_m3_priors.py` (generated by `validation/r/m3_priors_reference.R`):
  - **V7a (`v7_2pl_map`)**: 2PL with $a\sim\mathrm{NormalPrior}(1.2, 0.5)$, $d\sim\mathrm{NormalPrior}(0, 2)$,
    compared directly against mirt (`NR`, `SE.type = "Oakes"`): `loglik` rel `3.0e-13`,
    `log_posterior` rel `5.1e-15`, parameters `8.0e-9`, Oakes SE vs mirt Oakes `3.4e-9`, Oakes vs
    Hessian `1.2e-12`, EAP `6.4e-15` (and mirt Richardson matches flexpyirt's likelihood-only
    Hessian at $\hat\eta_{\text{MAP}}$).
  - **V7b (`v7_2pl_lnorm_map`)**: 2PL with $a\sim\mathrm{LogNormal}(0.2, 0.3)$, $d\sim\mathrm{NormalPrior}(0, 2)$,
    compared directly against mirt (`NR`) for `loglik` (`4.0e-13`), `log_posterior` (`1.4e-14`),
    parameters (`2.6e-9`), and EAP (`7.1e-15`), and against the exact R posterior Hessian for Oakes
    and Hessian SEs (`1.8e-9`; Oakes vs Hessian `2.3e-12`).
  - **V7c (`v7_3pl_map`)**: 3PL with $a\sim\mathrm{LogNormal}(0.2, 0.3)$, $d\sim\mathrm{NormalPrior}(0, 2)$,
    $g\sim\mathrm{Beta}(5, 17)$ (`expbeta` in mirt), maximizing mirt's exact `logLik + logPrior`
    objective in R (`max |grad| = 5.1e-13`, cross-checked against `mirt(..., pars = sv_fixed)` to
    `8.9e-15` with EAP from mirt `fscores`): `loglik` rel `2.5e-13`, `log_posterior` rel `8.1e-16`,
    parameters `2.9e-9`, Oakes and Hessian SEs `5.7e-9` (Oakes vs Hessian `5.7e-12`), EAP `6.0e-15`.
  Unit tests (scipy `log_pdf`, autodiff guards, fixed ignores prior, shared `free_id` counted once,
  edited table priors, AIC/BIC warning): `tests/test_m3_priors.py`.
- **Display IRT parameterization and delta-method SEs (V12, M4)**: `coef(m, IRTpars = TRUE, printSE = TRUE)`
  in mirt 1.46.1 transforms slope–intercept parameters to $b = -d/a$ (2PL/Rasch), $b_k = -d_k/a$
  (GRM / `graded`), and step difficulties $b_k = -(d_k - d_{k-1})/a$ with $d_0 = 0$ (GPCM / PCM),
  matching flexpyirt's `FitResult.items` definitions identically. However, inspecting
  `mirt:::mirt2traditional` for `cls == "gpcm"` uncovered an off-by-one indexing bug in mirt 1.46.1
  affecting **only `b1_se` of GPCM**:
  - `delta_index[[nfact + 1L]]` is overwritten with `c(which.a, ncat + nfact + 2L)` (indices of
    `a1` and `d1` in `opar`, which includes `d0`), while inside `fns` the returned vector `ret`
    omits `d0` so `b1` sits at position `ncat + nfact + 1L` and `b2` at `ncat + nfact + 2L`.
  - Consequently `ret[index[2]]` selects `b2 = -(d2 - d1)/a` instead of `b1 = -d1/a` and
    differentiates it with respect to `(a, d1)` only (holding `d2` fixed), giving gradient
    $((d_2 - d_1)/a^2,\, 1/a)$ instead of $(d_1/a^2,\, -1/a)$ (confirmed in R to `< 1e-9` against
    mirt's raw `b1_se`, which is off by up to 268% on `Science`).
  - Point estimates `b1..b{K-1}` and delta-method SEs `b2_se..b{K-1}_se` in mirt's `gpcm` output
    are exact (verified against `vcov(m)` to `< 1e-9`). For PCM (`a = 1` fixed), the `a` component
    is zero and $|\partial b_2/\partial d_1| = 1 = |\partial b_1/\partial d_1|$, so the bug cancels
    and mirt's `b1_se..b3_se` are all exact.

  Six cases are validated in `tests/test_mirt_m4_irtpars.py` (with both plain EM and SQUAREM;
  reference generator `validation/r/m4_irtpars_reference.R` → `tests/reference/v12_irtpars.json`,
  `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`, `SE.type = "Oakes"`):
  - **V12a (`lsat7_2pl`, 2PL, 61 pts $[-6,6]$)**: `loglik` rel `3.4e-16`, `max |Δb| = 1.4e-8`,
    `b_se` rel `1.5e-8`.
  - **V12b (`lsat7_rasch`, Rasch, 61 pts $[-6,6]$)**: `loglik` rel `3.1e-15`, `max |Δb| = 1.1e-7`,
    `b_se` (`= d_se`) rel `6.9e-8`.
  - **V12c (`science_grm`, GRM 0..3, 61 pts $[-6,6]$)**: `loglik` rel `3.7e-15`,
    `max |Δbk| = 2.7e-8`, `bk_se` rel `1.1e-8`.
  - **V12d (`science_gpcm`, GPCM 0..3, 61 pts $[-6,6]$)**: `loglik` rel `3.1e-15`,
    `max |Δbk| = 1.5e-9`, `b2_se` / `b3_se` vs mirt `IRTpars` rel `9.7e-10`, `b1_se` vs exact delta
    method from `vcov(m)` in R rel `9.3e-10`.
  - **V12e (`science_pcm`, PCM 0..3, 81 pts $[-8,8]$)**: `loglik` rel `8.4e-16`,
    `max |Δbk| = 2.3e-9`, `bk_se` (all three directly vs mirt `IRTpars`) rel `3.4e-10`.
  - **V12f (`mg_2pl`, 2-group 2PL, 4 anchors, focal free, 81 pts $[-8,8]$)**: `loglik` rel
    `1.1e-15`, `max |Δb| = 7.3e-9`, `b_se` rel `1.5e-8`.
  Unit tests (analytic checks, fixed parameters, `se=None`, non-PD `vcov`, partial equality,
  mixed models, `summary()`, `_repr_html_()`): `tests/test_m4_result.py`.
- **Comparing EAP**: so that differences in the estimates do not propagate into EAP, every
  parameter is fixed at mirt's estimates (`free=False` in the ParameterTable) and only the scoring
  is compared. Tolerance 1e-5.
- **Ability scoring: MAP, ML, WLE (V13, M4)**: inspected `mirt:::fscores.internal`, `mirt:::MAP`,
  `mirt:::ML`, `mirt:::WLE`, `mirt:::MAP.mirt`, and `mirt:::WLE.mirt` in mirt 1.46.1:
  - **Bounds**: `theta_lim` sets the EAP quadrature grid (and the EAP starting values for
    MAP/ML/WLE); MAP, ML, and WLE optimize over `[-max_theta, max_theta]` with default
    `max_theta = 20`.
  - **Objectives and SEs (default `expected.info = FALSE`)**: MAP maximizes
    $\log L(\theta\mid y) + \log\phi(\theta;\mu_g,\sigma_g^2)$ with the continuous normal latent
    prior, ML maximizes $\log L(\theta\mid y)$, and WLE maximizes
    $\log L(\theta\mid y) + \tfrac{1}{2}\log I(\theta)$ (where $I(\theta)$ is the observed-item
    Fisher test information, matching Warm's equation $\partial\log L/\partial\theta + I'(\theta)/(2I(\theta)) = 0$
    and Firth's Jeffreys prior). For all three, mirt's default SE is the inverse square root of the
    `nlm()` Hessian of the negative objective at the optimum (observed curvature; passing
    `expected.info = TRUE` without `response.pattern` switches to $I(\hat\theta)^{-1/2}$, which
    differs on short tests and extreme patterns; note mirt 1.46.1 does not forward `expected.info`
    through `response.pattern`).
  - **Extreme patterns and mirt 1.46.1 quirks**: for ML, all-lowest / all-highest observed
    categories return `-Inf` / `+Inf` with SE `NA`, and all-missing returns `NA` / `NA`. Two
    edge-case behaviours were seen in mirt 1.46.1: (a) on diverging 3PL low-score tails where
    $\log L(\theta)$ has its supremum at $-\infty$, when `nlm`'s finite-difference Hessian turns
    $\le 0$ (`SE = NA`), `mirt:::ML` skips its `SEest > 30` check and returns a finite
    `F1 ≈ -17` with `SE = NA` instead of `-Inf` (31 of 3,790 patterns in `v11_3pl`; confirmed in
    `mirt:::ML`, whose NaN-SE branch skips the `SEest > 30` → `±Inf` conversion); (b) with missing
    data, some all-extreme patterns that are missing the last item run into `max_theta = 20`
    (`F1 ≈ 20`, `SE = NA`) instead of `±Inf` (observed; the cause was attributed to how
    `fscores.internal` counts missing values, but the code could not be located in mirt's R source,
    so treat the explanation as unconfirmed). Neither case is used in the V13 comparisons. flexpyirt
    detects all of these cleanly and returns `±inf` with SE `NaN`. Separately, on bimodal 3PL
    patterns where a strictly concave interior local mode $\hat\theta$ (with $\mathrm{SE} \le 30$)
    has strictly smaller log-likelihood than the lower-asymptote plateau
    $\lim_{\theta\to-\infty}\log L(\theta)$, `mirt:::ML`'s local `nlm()` search starting from EAP
    stops at the interior local mode and returns finite `(F1, SE_F1)`, whereas flexpyirt compares
    $f(\hat\theta)$ against $f(\pm 20)$ and intentionally returns the true global supremum
    `(-inf, NaN)` (the actual maximum likelihood). Across all 3,790 unique patterns of `v11_3pl`,
    this occurs on **25 patterns** (none appeared in the seeded V13d subsample; pattern index 1285
    is verified directly in `tests/test_mirt_m4_scoring.py`). For all-missing rows, mirt strips
    them to `NA` for all methods, whereas flexpyirt's Bayesian methods (EAP, MAP, PV) return the
    prior and likelihood-only methods (ML, WLE) return `(NaN, NaN)`.

  Six cases are validated in `tests/test_mirt_m4_scoring.py` with all parameters fixed at mirt's
  estimates (reference generator `validation/r/m4_scoring_reference.R` →
  `tests/reference/v13_fscores.json`, `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`;
  tolerances `1e-4` abs for $\theta$, `1e-3` rel for SE, exact agreement on `-inf`/`+inf`/`NaN`;
  remaining ~`1e-6`–`2e-4` differences reflect `nlm`'s finite-difference gradient/Hessian):
  - **V13a (`lsat7_2pl`, 2PL, 32 patterns, 61 pts $[-6,6]$)**: `loglik` rel `5.1e-16`;
    MAP $\theta$ `2.2e-6` / SE rel `1.8e-5`; ML $\theta$ `1.5e-6` / SE rel `4.1e-5` (2 `±inf`);
    WLE $\theta$ `2.1e-6` / SE rel `5.1e-5`.
  - **V13b (`science_grm`, GRM 0..3, 90 patterns, 61 pts $[-6,6]$)**: `loglik` rel `3.7e-15`;
    MAP $\theta$ `3.0e-6` / SE rel `3.6e-5`; ML $\theta$ `2.5e-6` / SE rel `8.7e-5` (2 `±inf`);
    WLE $\theta$ `2.8e-6` / SE rel `1.5e-4`.
  - **V13c (`science_gpcm`, GPCM 0..3, 90 patterns, 61 pts $[-6,6]$)**: `loglik` rel `2.8e-15`;
    MAP $\theta$ `2.2e-6` / SE rel `4.1e-5`; ML $\theta$ `2.0e-6` / SE rel `8.8e-5` (2 `±inf`);
    WLE $\theta$ `2.5e-6` / SE rel `1.2e-4`.
  - **V13d (`v11_3pl`, 3PL, 112 patterns, 61 pts $[-6,6]$)**: `loglik` rel `1.5e-15`;
    MAP $\theta$ `3.8e-6` / SE rel `7.4e-5`; ML $\theta$ `5.8e-6` / SE rel `2.1e-4` (20 `±inf`,
    including diverging low-score patterns); WLE $\theta$ `3.2e-6` / SE rel `1.9e-4`.
  - **V13e (`twogroup_2pl`, 2-group 2PL, 104 patterns across ref/focal, 81 pts $[-8,8]$)**:
    `loglik` rel `1.9e-15`; MAP $\theta$ `2.9e-6` / SE rel `5.1e-5`; ML $\theta$ `2.6e-6` / SE rel
    `8.7e-5` (4 `±inf`); WLE $\theta$ `3.3e-6` / SE rel `1.6e-4`.
  - **V13f (`missing_2pl`, 2PL with 15% MCAR missing, 88 patterns, 61 pts $[-6,6]$)**:
    `loglik` rel `7.4e-16`; MAP $\theta$ `2.0e-6` / SE rel `4.6e-5`; ML $\theta$ `1.4e-6` / SE rel
    `1.1e-4` (8 `±inf`); WLE $\theta$ `2.0e-6` / SE rel `1.5e-4`.
  Plausible values (`method="PV"`; for $D > 1$ see `tests/test_mirt_m4_mirt.py`, DESIGN §15.42) are
  validated statistically in `tests/test_m4_scoring.py`
  (reproducibility given `key`, independence across respondents sharing a pattern, and sample
  mean/SD matching EAP and posterior SD within Monte-Carlo error), alongside unit tests for
  flat-prior MAP $\approx$ ML, WLE Warm-equation stationarity across all seven item models, extreme
  patterns, multiple groups, missing data, and input validation.
- **S-X2 item fit (V15, M5)**: six cases reuse existing references (V3 GRM and V8 GPCM on
  Science, V8 PCM with a free latent variance, V11 3PL, V5 FIPC 2PL with eight fixed items, and
  V3 again with `mincell = 3`). mirt evaluates `itemfit(m, fit_stats = "S_X2")` at its reference
  estimates with every parameter's `est` flag kept (so free parameters count against the df) and
  `TOL = NaN`, on the same grid as the fit (mirt 1.48, `validation/r/m5_itemfit_reference.R` →
  `tests/reference/v15_sx2.json`). flexpyirt refits the data: degrees of freedom are identical in
  every case, S-X2 agrees to `2.2e-8` (relative) and p-values to `1.3e-8`, the level at which the
  estimates themselves agree. Unit tests check the Lord–Wingersky recursion against brute-force
  enumeration and that expected rows are conditional probabilities. Tests:
  `tests/test_mirt_m5_itemfit.py`.
- **Confirmatory 2–3 dimensional MIRT (V14, M4)**: four simulated cases are validated in
  `tests/test_mirt_m4_mirt.py` (with both plain EM and SQUAREM; reference generator
  `validation/r/m4_mirt_reference.R` → `tests/reference/v14_mirt2d.json`, `accelerate = "none"`,
  `TOL = 1e-10`, `SE.type = "Oakes"`, mirt's default product grids on $[-6, 6]$: $31^2 = 961$ nodes
  for 2D and $15^3 = 3{,}375$ nodes for 3D). The three 2D cases use `optimizer = "NR"`; the 3D case
  uses `optimizer = "BFGS"` with `control = list(reltol = 1e-13, maxit = 500L)` because mirt's
  `NR` falls back to slow numerical Hessians on 3D `GroupPars` with 3,375 nodes. In all four cases
  `flexpyirt.loglik >= fixed_at_mirt.loglik` holds, EM and SQUAREM are strictly monotone (with
  2.2×–2.8× fewer EM map evaluations under SQUAREM), and Oakes agrees with direct Hessian to
  $\le 1.2\times 10^{-14}$:
  - **V14a (`mirt2d_2pl_simple`, 2D 2PL, $J=12, N=2000$, `F1=1-6`, `F2=7-12`, $\rho=0.5$, $31^2$ pts)**:
    `loglik` rel `2.1e-15`, item params `3.3e-9`, `cor_12` `1.0e-8`, item Oakes SE rel `3.7e-9`,
    `cor_12_se` rel `1.1e-8`, Oakes vs Hessian `7.5e-15`, EAP $\theta$ `7.3e-15` / SD `8.0e-15`.
  - **V14b (`mirt2d_2pl_cross`, 2D 2PL with cross-loading `q07`, $J=12, N=2500$, $\rho=0.45$, $31^2$ pts)**:
    `loglik` rel `1.8e-15`, item params `4.0e-9`, `cor_12` `7.3e-9`, item Oakes SE rel `4.9e-9`,
    `cor_12_se` rel `2.9e-9`, Oakes vs Hessian `5.6e-15`, EAP $\theta$ `6.9e-15` / SD `7.0e-15`.
  - **V14c (`mirt2d_grm`, 2D GRM $K=4$, $J=10, N=2000$, `F1=1-5`, `F2=6-10`, $\rho=0.55$, $31^2$ pts)**:
    `loglik` rel `1.7e-15`, item params `2.0e-9`, `cor_12` `4.6e-9`, item Oakes SE rel `2.2e-9`,
    `cor_12_se` rel `3.0e-9`, Oakes vs Hessian `4.4e-15`, EAP $\theta$ `8.0e-15` / SD `1.2e-14`.
  - **V14d (`mirt3d_2pl`, 3D 2PL, $J=9, N=2000$, `F1=1-3`, `F2=4-6`, `F3=7-9`, $\rho=(0.45, 0.25, 0.50)$, $15^3$ pts)**:
    `loglik` rel `4.7e-15`, item params `1.9e-8`, correlations `4.9e-8`, item Oakes SE rel
    `9.0e-7`, correlation Oakes SE rel `7.3e-5` (mirt's 3D finite-difference Oakes step; flexpyirt
    Oakes vs Hessian `1.2e-14`), EAP $\theta$ `1.1e-14` / SD `1.1e-14`.
  - **MAP (all four cases)**: with every parameter fixed at mirt's V14 estimates,
    `fscores(method = "MAP")` on the EAP patterns plus 20 copies with 3 responses set to missing
    (220 patterns per case; mirt 1.48, `validation/r/m4_mirt_map_reference.R` →
    `tests/reference/v14_mirt_map.json`): $\theta$ within `4.3e-6` and SEs within `3.8e-5`
    (relative) of mirt, at the level of mirt's `nlm` tolerance (tolerances `1e-4` / `1e-3`).
  - **ML and WLE (all four cases)**: same setup as MAP plus the all-lowest, all-highest and a
    one-factor-extreme pattern (223 patterns per case; `validation/r/m4_mirt_mlwle_reference.R` →
    `tests/reference/v14_mirt_mlwle.json`). WLE agrees on all 892 rows ($\theta$ `5.0e-6`, SE
    `1.7e-4` relative). ML agrees on the 644 rows that are interior in both programs ($\theta$
    `6.5e-6`, SE `1.2e-4`); on the 242 rows where mirt returns `±Inf` on some dimension,
    flexpyirt returns the same signed infinity, and on 22 rows where mirt stops a diverging
    dimension near ±20 with SE 0, flexpyirt returns the signed infinity (DESIGN §15.41).
  Unit tests (kernels, MVN weights & PD guard, product grid shapes, 2D Rasch/mixed/priors/fix,
  Oakes == Hessian == reference Oakes, identification & specification errors):
  `tests/test_m4_mirt.py`.
- Tests: `tests/test_mirt_m1.py` (V1, V2, V10). Reference generator: `validation/r/m1_reference.R`.
- **Equality constraints across items (V16, M5)**: mirt's `CONSTRAIN` with `optimizer = "NR"`,
  `TOL = 1e-10`. V16a: LSAT7 2PL with one common slope ($N(0,1)$, 61 points on $[-6,6]$). V16b:
  simulated $N=2500$, items 1–6 2PL and 7–10 graded with 3 categories, with a slope shared by
  2PL and graded items (1, 2, 7), an intercept shared by items 3 and 4, a slope shared by 8–10 and
  `d1` shared by 9 and 10. V16c: two groups of 1500, anchors q01–q05 equal across groups, the
  slopes of q06–q10 one common slope within each group, focal mean and variance free (81 points on
  $[-8,8]$). All agree with mirt (plain EM and SQUAREM): parameters to $2.1\times10^{-8}$,
  log-likelihood to $4\times10^{-11}$, Oakes SEs to $1.3\times10^{-8}$ (relative), Hessian vs
  mirt Richardson SEs to $2.9\times10^{-8}$, flexpyirt Oakes vs Hessian to $5\times10^{-15}$, and
  the number of free parameters equals mirt's `nest`. Tests: `tests/test_mirt_m5_equal_items.py`.
  Reference generator: `validation/r/m5_equal_items_reference.R`.
- **DIF likelihood-ratio tests (V17, M5)**: mirt's `DIF()` (`optimizer = "NR"`, `TOL = 1e-10`, 81
  points on $[-8,8]$ for the baseline and every refit) on the V4 data: `scheme = "add"` from the
  anchored model (q01–q05 equal, focal mean and variance free) for `a1`+`d` and for `d` on
  q06–q15, and `scheme = "drop"` for `a1`+`d` on all items from the fully invariant model. mirt
  reports information criteria as the model with the item free minus the constrained one (in both
  schemes) and adjusts p-values with Holm by default (`p.adjust = "holm"`); its `p` is
  `1 - pchisq(...)`, which is exactly 0 for very large $X^2$ (ours uses the survival function).
  Agreement (plain EM and SQUAREM): $X^2$ and the AIC/SABIC/HQ/BIC differences to $2.2\times10^{-11}$,
  `p` and `adj_p` to $3\times10^{-11}$. Tests: `tests/test_mirt_m5_dif.py`. Reference generator:
  `validation/r/m5_dif_reference.R`.
- **Nominal response model (V23, M5)**: mirt's `itemtype = "nominal"` (`optimizer = "NR"`,
  `TOL = 1e-10`, 61 points on $[-6,6]$; 31 per dimension for V23f). V23a: `Science` recoded to
  `0..3`. V23b: simulated $J=10, K=4, N=3000$ with swapped interior scoring coefficients (`q09`:
  $\mathrm{ak}_1=2.2 > \mathrm{ak}_2=0.7$) and a negative one (`q10`: $\mathrm{ak}_1=-0.4$). V23c:
  five nominal items ($K=4$) and five 2PL items, $N=2500$. V23e: simulated $J=8, K=4, N=3000$ with
  six non-monotone items, including $\mathrm{ak}_k<0$ and $\mathrm{ak}_k>K-1$. V23f: two-factor
  confirmatory nominal model ($J=8$, four items per factor, $K=4$, $N=3000$, correlation free).
  **Starting values**: mirt's default nominal starting values (`ak1 = ... = 0.5`, `d = 0`) make
  undamped NR stop at a saddle on `Science` (log-likelihood $-1631.79$ with Comfort's `a1`
  $\approx 0$, versus $-1608.452$), so the script starts mirt's NR at the GPCM point
  ($\mathrm{ak}_k=k$, $d_k=\log(p_k/p_0)$). It also fits five alternative mirt starts per
  unidimensional case (default-start NR and BFGS, three `GenRandomPars` NR starts) and stores them
  in `alt_starts`; when one of them is higher, NR is restarted from it (`meta.reference_start`).
  This happens on V23e, where the GPCM-point NR stops at a local maximum ($-29561.14$) and the
  default-start BFGS reaches $-26107.48$. No stored start exceeds any reference, and flexpyirt,
  always fitting from its own default starting values, reaches every reference (including V23e).
  **Agreement** (plain EM and SQUAREM; maxima over V23a–c, e): log-likelihood to
  $3.6\times10^{-11}$ absolute ($1.6\times10^{-15}$ relative), parameters to $7.1\times10^{-8}$,
  Oakes SEs to $3.8\times10^{-8}$ (relative), Hessian vs mirt Richardson SEs to
  $5.6\times10^{-7}$, flexpyirt Oakes vs Hessian to $1.7\times10^{-13}$, EAP to $8\times10^{-15}$,
  MAP / ML / WLE on 40–44 patterns per case to $3.7\times10^{-6}$ in $\theta$ and
  $1.9\times10^{-4}$ (relative) in SE, and S-X2 (`itemfit(fit_stats = "S_X2")`) to
  $5.8\times10^{-9}$ (relative) with identical degrees of freedom (Science: $8.235, 10.223, 6.246,
  12.483$ on 6, 9, 6, 9 df). V23f: log-likelihood to $4.0\times10^{-11}$, parameters to
  $7.0\times10^{-9}$, Oakes SEs to $2.5\times10^{-9}$, correlation to $5.4\times10^{-12}$ (SE
  $9.3\times10^{-11}$ relative), EAP within $10^{-5}$. **Known ML difference**: mirt's
  `fscores(method = "ML")` returns $-\infty$ / $+\infty$ for every all-category-0 /
  all-top-category pattern regardless of the scoring coefficients, and leaves one true $+\infty$
  V23e pattern (not all-top-category) at its `max_theta = 20` bound ($19.99989$, SE NA).
  flexpyirt classifies extreme patterns by the category slopes $a\,\mathrm{ak}_c$; the tests
  check that classification against an independent R scan of the pattern log-likelihood
  (`fscores.ml_scan`: 0.01 grid on $[-20,20]$ refined with `optimize()`) and compare the interior
  ML estimates (one in V23b and V23c, two in V23e) with it. Multidimensional nominal models are
  validated for two factors (V23f); three factors are unit-tested only. Unit tests (probabilities,
  gradients, reduction to GPCM / 2PL, starting values, identification, `fix_from`, JSON, S-X2,
  scoring, priors, equality constraints, multiple groups and DIF, EH, 2D, the degenerate-item
  `HeywoodWarning`): `tests/test_m5_nominal.py`. Tests: `tests/test_mirt_m5_nominal.py`.
  Reference generator: `validation/r/m5_nominal_reference.R`.

## Validation beyond agreement
- **Parameter-recovery simulations**: bias / RMSE over conditions (N, J, number of groups, number
  of anchors) × 100 replications, with results and figures in `validation/recovery/`.
- **EM monotonicity**: in every validation case, the log-likelihood (posterior with MAP) is
  non-decreasing across iterations.
- **Internal SE consistency**: Oakes SEs agree with SEs from the direct `jax.hessian` of the
  marginal log-likelihood at the converged point (small cases).
- **Invariance**: results do not change with item reordering, group relabelling, or with or
  without pattern compression.
