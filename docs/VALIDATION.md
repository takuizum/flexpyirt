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

## Cases (by milestone)
| ID | Contents | mirt call | MS |
|---|---|---|---|
| V1 | 2PL, single group (LSAT7) | `mirt(LSAT7, 1, "2PL", SE=TRUE, SE.type="Oakes")` | M1 |
| V2 | Rasch, single group | `mirt(..., "Rasch")` | M1 |
| V3 | GRM, single group (Science) | `mirt(Science, 1, "graded")` | M2 |
| V4 | 2PL, two groups, partial anchors, focal group free | `multipleGroup(..., invariance=c(anchors, "free_means","free_var"))` | M2 |
| V5 | FIPC (fixed items + latent distribution estimated) | `pars='values'` with `est=FALSE` | M2 |
| V6 | EH (single group, two groups) | `dentype="EH"` | M3 |
| V7 | 3PL + priors (MAP) | equivalent of `PRIOR = (1-30, g, expbeta, ...)` in `mirt.model` | M3 |
| V8 | GPCM | `"gpcm"` | M3 |
| V9 | Missing data (MAR) | data with NA | M2 |
| V10 | SE (Oakes / crossprod / sandwich) | `SE.type=` | M1–M3 |
| V11 | 3PL and 4PL, single group (simulated) | `mirt(..., "3PL")` / `mirt(..., "4PL")` | M3 |

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
- **Comparing EAP**: so that differences in the estimates do not propagate into EAP, every
  parameter is fixed at mirt's estimates (`free=False` in the ParameterTable) and only the scoring
  is compared. Tolerance 1e-5.
- Tests: `tests/test_mirt_m1.py` (V1, V2, V10). Reference generator: `validation/r/m1_reference.R`.

## Validation beyond agreement
- **Parameter-recovery simulations**: bias / RMSE over conditions (N, J, number of groups, number
  of anchors) × 100 replications, with results and figures in `validation/recovery/`.
- **EM monotonicity**: in every validation case, the log-likelihood (posterior with MAP) is
  non-decreasing across iterations.
- **Internal SE consistency**: Oakes SEs agree with SEs from the direct `jax.hessian` of the
  marginal log-likelihood at the converged point (small cases).
- **Invariance**: results do not change with item reordering, group relabelling, or with or
  without pattern compression.
