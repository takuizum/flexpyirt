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
| Scale linking $(A, B)$ / criterion, on identical item estimates | 1e-9 / 1e-9 (relative) |
| Scale linking delta-method SE / `vcov`, on identical estimates and covariance | 1e-5 (relative) |
| Scale linking $(A, B)$ / criterion / SE, end-to-end (calibration + linking) | 1e-6 / 1e-5 (relative) / 1e-3 (relative) |
| Polychoric correlation vs lavaan / psych; thresholds; smoothing vs `cor.smooth` (both implementations) | 1e-6 / 1e-3; 1e-12; 1e-12 |
| flexpyctt (NumPy) vs flexpyirt (JAX) polychoric correlation on the V24 cases; $\Phi_2$ by Owen's $T$ vs Plackett's integral | 1e-9, equal flags; 1e-14 |
| EKC, MAP, observed eigenvalues on the same matrix; normal reference mean / 95th percentile | 1e-11; $\lvert z\rvert\le4.5$ / 6e-3 |
| LD $X^2$ / $G^2$; signed $V$, Q3 (refitted models); model-based PA observed / simulated mean / 95th percentile | 1e-5 (relative); 1e-6; 2e-4 / $\lvert z\rvert\le4.5$ / 0.02 |
| DETECT conditional covariances, indices (both halves), Ward heights | 1e-12 |
| EFA implied $\Lambda\Lambda^\top$ / communalities; CAF, CFI, RMSEA; $\chi^2$; Hull fit / $st$ | 1e-4; 2e-4; 1e-4 (relative); 1e-5 / 1e-2 (relative) |
| CD: PAF $\Lambda\Lambda^\top$; one-factor population from the same draws; rank-sum $p$; mean RMSE (retained $k$ / the rejected extra $k$) | 1e-12; exact; 1e-10 (relative); 15% / 60% (relative) |
| DIMTEST (simulation study; small version in the tests): type I error at .05 / power | $\le 0.15$ of 20 / $\ge 8$ of 10 |
| Observed-score DIF: MH / GMH statistics, $\hat\alpha_{MH}$, its log variance, Mantel statistic, cumulative odds ratio and its SE; logistic LRT, $\Delta R^2$, coefficients / SEs and Wald; flags and purification paths | 1e-10 (relative); 1e-8 / 1e-6 (relative); equal |
| Cross-product / sandwich covariance at mirt's estimates (entries relative to $\sqrt{v_{ii}v_{jj}}$); parameters / SEs end-to-end | 1e-9 / 1e-6; 1e-6 / 1e-5 (relative) |
| Classical item analysis (alpha, standardized alpha, if-deleted values, Feldt interval, item–total / item–rest, point-biserial, two-step polyserial / biserial, keyed scores, option statistics) vs psych / ltm / polycor / base R; omega total and its loadings vs psych | 1e-12 (scores exact); 1e-6 / 1e-5 |
| Person / item fit at mirt's $\theta$ ($l_z$, infit, outfit, $z$); $l_z^*$ vs the R recomputation; end-to-end with flexpyirt's own estimates | 1e-9 (relative); 1e-7; 1e-4 |

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
| V10 | SE (Oakes; crossprod / sandwich are V35) | `SE.type=` | M1–M3 |
| V11 | 3PL and 4PL, single group (simulated) | `mirt(..., "3PL")` / `mirt(..., "4PL")` | M3 |
| V12 | Display IRT thresholds and delta-method SEs (2PL, Rasch, GRM, GPCM, PCM, 2-group 2PL) | `coef(m, IRTpars = TRUE, printSE = TRUE)` | M4 |
| V13 | Ability scoring: MAP, ML, WLE (2PL, GRM, GPCM, 3PL, 2-group, missing) | `fscores(m, method = "MAP"/"ML"/"WLE", response.pattern = ..., full.scores.SE = TRUE)` | M4 |
| V15 | S-X2 item fit (GRM, GPCM, PCM with free variance, 3PL, FIPC 2PL; `mincell` 1 and 3; two-group 2PL and GRM, per group) | `itemfit(m, fit_stats = "S_X2")` | M5 |
| V16 | Equality constraints across items (LSAT7 common slope; 2PL + graded with cross-model constraints; two groups with anchors and a common slope) | `mirt(..., mirt.model("... CONSTRAIN = (1-5, a1)"))` / `multipleGroup(...)` | M5 |
| V17 | DIF likelihood-ratio tests on the V4 data (`add` for `a`+`d` and for `d`; `drop` for `a`+`d` from the fully invariant model) | `DIF(mg, which.par = ..., scheme = ...)` | M5 |
| V18 | M2 / M2* limited-information goodness of fit (LSAT7 2PL, simulated 2PL, 3PL, Rasch with free variance, GRM $K=3$, GPCM $K=4$) | `M2(m, quadpts = 61, theta_lim = c(-6, 6))` | M5 |
| V14 | Confirmatory 2–3D MIRT (2PL simple, 2PL cross-loading, GRM, 3D 2PL) | `mirt(..., mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes")` | M4 |
| V19 | Davidian-curve latent distributions (2PL $k=3$, GRM $k=2$, FIPC 2PL $k=3$ with free mean/var) | `mirt(..., dentype = "Davidian-k")` & `mirt(..., customGroup = ..., SE.type = "Oakes")` | M5 |
| V20 | Scale linking: Haebara & Stocking–Lord (non-symmetric & symmetric) + Ogasawara SEs (2PL, GRM, GPCM, 3PL, non-standard source grid, mixed 2PL + GPCM) | separate `mirt()` calibrations + base-R criteria via `probtrace()` / `expected.item()` | M5 |
| V21 | Multiple-group confirmatory 2D MIRT (simple structure, cross-loading; anchors on both factors, focal means/variances free) | `multipleGroup(..., invariance = c(anchors, "free_means", "free_var"))` with `COV = F1*F2` | M4 |
| V22 | Normal-mixture latent distributions (2PL $k=2$ bimodal, GRM $k=2$ with a small component, GPCM $k=3$) | `multipleGroup(..., dentype = "mixture-k", invariance = c(<all items>, "free_means", "free_var"))` on the image grid, plus an independent R implementation of the standardized likelihood (Newton polish, Richardson-Hessian SEs, EAP) | M5 |
| V23 | Nominal response model (Science; simulated with unordered/negative scoring coefficients; nominal + 2PL mixed; mostly non-monotone items; two-factor confirmatory) | `mirt(..., "nominal")` with `pars` starting at the GPCM point (or the best alternative start); `itemfit(fit_stats = "S_X2")` | M5 |
| V24 | Polychoric / tetrachoric correlations (simulated tetrachoric, mixed 2–5 categories, 10% MCAR, one-empty-cell 2 x 2 tables, LSAT7, Science) and smoothing | lavaan `lavCor(..., ordered = , missing = )`; psych `polychoric(..., smooth = FALSE, correct = 0)`, `cor.smooth()` | M5 |
| V25 | Number of dimensions: EKC, MAP (squared and fourth power), PCA / SMC eigenvalues on the V24 matrices; normal-data parallel-analysis references (3 sizes) | EFAtools `EKC()`, `PARALLEL()`; psych `vss()$map`; EFA.dimensions `MAP()`; independent R fourth-power MAP | M5 |
| V26 | Local dependence (LSAT7 2PL, Science GRM, simulated 2PL with a dependent pair and 5% MCAR) and model-based parallel analysis (LSAT7) | `residuals(m, type = "LD" / "LDG2" / "Q3")`; ltm `unidimTest(data = , thetas = cbind(d, a1), IRT = FALSE)` | M5 |
| V26 (M7) | Local dependence for two groups (2PL with 4% MCAR, GRM; two factors 2PL), `EmpiricalHistogram`, `DavidianCurve(2)`, `NormalMixture(2)` (V22a's estimates), two factors (2PL with a cross-loading and 3% MCAR, GRM) and three factors (2PL) | `residuals()` on `multipleGroup()`, `dentype = "EH"` / `"Davidian-2"` (`use_dentype_estimate = TRUE`; `Theta = fscores(m, use_dentype_estimate = TRUE)` for Q3) and `mirt.model()` fits; for the mixture a single-group mirt object with the mixture weights as its prior | M7 |
| V27 | DETECT: confirmatory (simulated 2D with true / wrong partitions, 1D, Science) and exploratory (simulated 3 clusters, given split) | sirt `conf.detect(use_sum_score = TRUE, smooth = FALSE)`, `expl.detect(use_sum_score = TRUE, estsample = )` | M5 |
| V28 | EFA (ULS, ML; every $k$ up to Hull's bound) and the Hull method (CAF, CFI, RMSEA) on the V24 matrices with $\ge 6$ variables and two simulated matrices | EFAtools `EFA(method = , rotation = "none")`, `HULL(method = , gof = )`; psych `fa(fm = "minres" / "ml", rotate = "none")` | M5 |
| V29 | Comparison data: PAF, the one-factor population generator from exported draws, rank-sum tests, `CD()` on 3-factor continuous, 2-factor ordinal and 1-factor binary data | EFAtools `.paf_iter`, `.simulate_cfm_empirical`, `CD()`; `wilcox.test(x, y, "less")` | M5 |
| V30 | DIMTEST: type I error and power (2PL; $N$ 1000–2000, $J$ 20–30; unidimensional and two dimensions correlated 0.7–0.8; 200 replications each) | none (no implementation outside DIMPACK): `packages/flexpyctt/validation/simulation/dimtest_study.py` | M5 |
| V31 | Jackknife slope index (LSAT7 2PL, Science GRM, simulated 2PL with two dependent pairs and 5% MCAR, simulated GPCM with a dependent pair); the Newton and one-step approximations against the refit | `residuals(m, type = "JSI", fold = FALSE, TOL = 1e-10)` | M5 |
| V32 | Bifactor models with dimension reduction (2PL with three specific factors and three general-only items; GRM $K=4$ with three testlets) | `bfactor(data, model, itemtype = , SE = TRUE, SE.type = "Oakes", quadpts = 31)`; `fscores(method = "EAP", quadpts = 15)` | M7 |
| V33 | Two-tier models with dimension reduction (2PL with two correlated primary factors and three specific factors, one spanning both; GRM $K=4$ with a cross-loading item and two testlets) | `bfactor(data, model, model2 = mirt.model("F1 = ...\nF2 = ...\nCOV = F1*F2"), SE = TRUE, SE.type = "Oakes", quadpts = 15)`; `fscores(method = "EAP")` | M7 |
| V34 | Sampling weights: 2PL with non-integer weights (sum $\ne N$, five zero weights), Science GRM with weights normalized to $N$, two groups with anchors, free focal latent and 5% missing, FIPC with a free latent, a bifactor 2PL with dimension reduction; estimates, log-likelihood, AIC, BIC, Oakes SEs | `mirt(..., survey.weights = w)`, `multipleGroup(..., survey.weights = w)`, `bfactor(..., survey.weights = w)`, `SE.type = "Oakes"` | M7 |
| V35 | Robust standard errors: cross-product and sandwich covariance matrices (2PL under misspecification with 5% MCAR; 3PL + GRM + GPCM + nominal; MAP with normal priors; two groups with a free focal mean and variance; FIPC with an equality across items; two and three correlated factors; EH under FIPC) | `mirt(..., SE.type = "crossprod")` / `SE.type = "sandwich"` (`"Richardson"` for the EH bread) | M7 |
| V36 | Summed-score distribution kernel (Poisson-binomial: five probability vectors, $J$ 10–300; polytomous: GRM $K=4$ and mixed 2PL + GRM) | `poibin::ppoibin(method = "RF" / "DFT-CF" / "RNA" / "NA")`, `poibin::dpoibin()`, `stats::dpois()`; mirt `fscores(method = "EAPsum", full.scores = FALSE)` | M7 |
| V37 | Information, expected score, reliability and summed-score EAP at mirt's estimates (2PL $J=12$; the same with 8% missing; 3PL $J=12$; GRM and GPCM $K=4$, $J=8$; nominal $K=4$, $J=6$; mixed 2PL + 3PL + GRM + GPCM + nominal, $J=14$; two groups, 2PL + GRM, focal mean and variance free) | `iteminfo()`, `testinfo()`, `expected.item()`, `expected.test()`, `marginal_rxx()`, `empirical_rxx()`, `fscores(method = "EAPsum", full.scores = FALSE / TRUE)` | M7 |
| V38 | Local dependence for bifactor and two-tier models at mirt's estimates (bifactor 2PL, $J=11$, three specific factors, two general-only items, 3% MCAR; bifactor GRM $K=3$, $J=8$; two-tier 2PL, $J=10$, two correlated primary factors with a cross-loading item and two specific factors) | `residuals(bfactor(...), type = "LD" / "LDG2" / "Q3")`, `bfactor(model2 = )` for two-tier; Q3 with `Theta = fscores(m, quadpts = <the fit's>)` | M7 |
| V39 | $M_2$ / $M_2^*$ and S-X2 for confirmatory multidimensional models (the four V14 models: 2D 2PL simple structure, 2D 2PL with a cross-loading, 2D GRM $K=4$, 3D 2PL; correlated factors; `mincell` 1 and 3 for the GRM) | `M2(m, quadpts =, theta_lim =)`, `itemfit(m, fit_stats = "S_X2", quadpts =, theta_lim =)` at mirt's V14 estimates, plus an exact R $M_2^*$ with the correlation columns in $\Delta_2$ | M6 |
| V40 | Scale linking for 4PL, nominal and mixed 2PL + nominal common items; multidimensional linking $\boldsymbol\theta^*=\mathbf A\boldsymbol\theta+\boldsymbol\beta$ (diagonal, orthogonal, oblique $\mathbf A$) for two-factor 2PL and GRM and three-factor 2PL; Haebara & Stocking–Lord (non-symmetric & symmetric) + Ogasawara covariances | separate `mirt()` calibrations + base-R criteria via `probtrace()` / `expected.item()` minimized with `optim()` | M6 |
| V42 | $M_2^*$ for multiple groups, missing data and MAP fits (two-group 2PL with unequal groups and anchors, complete and 15% MCAR; two-group GRM $K=3$; two-group two-factor 2PL; single-group 2PL with 20% MCAR; 2PL MAP) | `M2(m, quadpts =, theta_lim =)` on `multipleGroup()` / `mirt()` fits at mirt's estimates, plus an independent R computation of DESIGN §16.20 and R code with mirt's conventions | M6 |
| V43 | Information matrix, directional information and expected scores of multidimensional models at chosen parameters (two correlated factors with 2PL, 3PL, 4PL, GRM, GPCM and nominal items and four cross-loadings; three factors with 2PL, 3PL, GRM and GPCM and two cross-loadings; bifactor 2PL + GRM with three specific factors; two-tier 2PL + GRM with two correlated primary factors and two specific factors) | `iteminfo(item, Theta, multidim_matrix = TRUE)`, `iteminfo(item, Theta, degrees = )`, `testinfo(m, Theta, degrees = )`, `expected.item()`, `expected.test()` on `mirt()` / `bfactor()` objects built with `pars =` and `TOL = NaN` | M7 |
| V45 | Optional unconstrained M-step scale (`BAEM(m_step_scale="unconstrained")`) on the V3, V4, V5, V7, V9, V11 and V14 cases; `LogitNormalPrior` (`"natural"`: no mirt equivalent, independent numerical check; `mode_scale="logit"`: mirt `norm` on `g`) | the existing references of those cases; SciPy (`stats.norm`, `integrate.quad`) and a stationarity condition written from the formula; `v45_3pl_logit_norm_map.json` (mirt) and the flexMIRT logit-metric mode computed in the test | #15 |
| V46 | Classical item analysis (flexpyctt): keyed scoring, alpha (pairwise and listwise), item statistics, point-biserial / biserial / polyserial correlations, omega total (Pearson and polychoric), option analysis; simulated 10-item multiple-choice (complete, 5% MCAR) and 6-item 0–4 data (complete, 6% MCAR) | none (not mirt): psych `alpha()`, `score.multiple.choice()`, `responseFrequency()`, `polyserial()`, `biserial()`, `fa()`, `omega(nfactors = 1)`; ltm `cronbach.alpha()`, `biserial.cor()`; polycor `polyserial(ML = FALSE)`; lavaan `lavCor()`; base R for option statistics | flexpyctt |
| V47 | Observed-score DIF (flexpyctt, #102): Mantel–Haenszel with and without continuity correction, Holm / BH, purification, user anchors, an external matching variable (`difR::verbal`, simulated 2PL with uniform and non-uniform DIF); generalized MH for three groups and for nominal categories (simulated graded responses); Mantel's test and the Liu–Agresti odds ratio; logistic regression (both / uniform / non-uniform, LRT / Wald, purification, anchors, matching) | difR 6.1.0 `difMH()`, `difGMH()`, `difLogistic()`, `difMantel.poly()`, `liu_agresti_ccor()`; `stats::mantelhaen.test()`; direct R implementations: `packages/flexpyctt/validation/r/ctt_dif_reference.R`; the worked example of Liu & Agresti (1996, Table 1) | — |
| V48 | The numbers behind the plots at mirt's estimates (2PL $J=10$, $N=1003$; 2PL with 10% missing; GRM $K=4$, $J=6$; mixed 2PL + 3PL + GPCM + nominal, $J=10$; two groups, 2PL + GRM, focal mean and variance free; TAM's GPCM $K=4$, $J=6$): empirical item curves, trace lines, Thurstonian thresholds | `itemfit(fit_stats = "X2", return.tables = TRUE, group.bins = 10, mincell.X2 = 0, Theta = fscores(method = "EAP"))` (per group via `extract.group()`), `probtrace()`, `uniroot()` on `probtrace()`, `coef(IRTpars = TRUE)`; TAM `tam.threshold()` | #103 |
| V49 | DIF screening with anchor purification (all others as anchors, sequential freeing, sequential constraining; two groups, 8 2PL + 4 GRM $K=3$; three groups 2PL), Wald DIF tests, signed / unsigned expected-score differences, DTF, ESSD / ETSSD and RMSD at mirt's estimates | `DIF(scheme = "drop" / "drop_sequential" / "add_sequential", seq_stat = 0.05, p.adjust = "BH", max_run = )`, `DIF(Wald = TRUE)`, `mirt:::calc_DRFs(best_fitting = TRUE)` (behind `DRF()`), `empirical_ES()`, `RMSD_DIF(probfun = FALSE)` | M5 |
| V50 | Common-item screening for linking (#105): robust z (Huynh & Meyer, 2010), displacement and RMSD of expected item score curves; removal order, linking constants at every purification step and the final item table (2PL; mixed 2PL + GRM + GPCM) | separate `mirt()` calibrations + base-R recomputation (`median()`, `quantile(type = 7)`, `expected.item()`, V20 linking); `equateIRT::dif.test` as a cross-check of the drifted items | field |
| V51 | IRT true-score and observed-score equating and raw-to-scale tables (dichotomous: 20 + 20 items, 3PL + 2PL, with a link; polytomous: 4 GRM $K=4$ + 6 2PL vs 3 GPCM $K=3$ + 2 GRM $K=4$ + 5 3PL + 3 2PL, with a link; populations new / old / synthetic $w_1=0.5$; scale table true score and EAPsum) | `equateIRT:::score_compute(method = "TSE" / "OSE")`, `SNSequate::irt.eq(method = "TS")`; mirt `expected.test()`, `probtrace()`, `testinfo()`, `fscores(method = "EAPsum", full.scores = FALSE)` with `uniroot()`, and `equate::equate(type = "equipercentile")` | #106 |
| V52 | Classification accuracy and consistency for cut scores (#107): Lee (summed-score cuts) and Rudner ($\theta$ cuts), D and P approaches, one to three cuts, non-integer cuts, both cut-scale conversions (3PL $J=20$; GRM $K=4$, $J=10$; two-group 2PL $J=15$, focal mean and variance free) | cacIRT 1.4 `class.Lee()`, `Lee.poly.D()` / `Lee.poly.P()`, `Rud.D()` / `Rud.P()` with mirt's `testinfo()` SE; independent recomputation from mirt `probtrace()` / `expected.test()` / `testinfo()` (kappa, false positive / negative rates, per-category indices, tables) | field |
| V53 | Person fit ($l_z$, $l_z^*$, person infit / outfit with $z$) and item infit / outfit at the EAP, MAP, ML and WLE estimates, on V37's data and estimates (2PL; 2PL with 8% missing; 3PL; GRM and GPCM $K=4$; nominal $K=4$; mixed 2PL + 3PL + GRM + GPCM + nominal; two groups 2PL + GRM) and a Rasch refit | `personfit(m, Theta = )`, `itemfit(m, fit_stats = "infit", Theta = )`; `sirt::personfit.stat()`; $l_z^*$ recomputed in R from the formulas | — |
| V54 | Model-comparison table (Rasch vs 2PL vs 3PL on the V11a data; PCM vs GPCM on the V8b data; a fully invariant two-group 2PL vs the anchored model and vs the model with the last item free, V4 data) and simulation of responses (deterministic inversion for 2PL, Rasch, 3PL, 4PL, GRM, GPCM and nominal items, unidimensional and two-factor; distributional and recovery checks per latent distribution) | `anova(mod1, mod2, ...)`; `probtrace()` with the same $\theta$ and uniform numbers (random streams cannot match `simdata()`) | #109 |
| V55 | Many groups with the group-blocked E-step: 2PL, 10 groups of unequal size (200–470), 15 items, anchors q01–q06 invariant, q07–q15 free per group, focal means and variances free, 5% MCAR missing; regression of the blocked against the flat E-step | `multipleGroup(invariance = c(<anchors>, "free_means", "free_var"), SE.type = "Oakes")`, `fscores(extract.group(m, g), method = "EAP")`; the flat path for the regression | #79 |

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
  - **V7b (`v7_2pl_lnorm_map`)**: 2PL with $a\sim\mathrm{LogNormalPrior}(0.2, 0.3)$, $d\sim\mathrm{NormalPrior}(0, 2)$,
    compared directly against mirt (`NR`) for `loglik` (`4.0e-13`), `log_posterior` (`1.4e-14`),
    parameters (`2.6e-9`), and EAP (`7.1e-15`), and against the exact R posterior Hessian for Oakes
    and Hessian SEs (`1.8e-9`; Oakes vs Hessian `2.3e-12`).
  - **V7c (`v7_3pl_map`)**: 3PL with $a\sim\mathrm{LogNormalPrior}(0.2, 0.3)$, $d\sim\mathrm{NormalPrior}(0, 2)$,
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
- **Multiple-group 2D MIRT (V21, M4)**: two simulated cases (2 × 1,500 respondents; simple
  structure with anchors q01–q03 / q07–q09, and a cross-loading q07 with anchors q01–q03 /
  q08–q10), focal means and variances free, correlations free in both groups, 81 points per
  dimension on $[-6, 6]$ (`validation/r/m4_mirt_multigroup_reference.R` →
  `tests/reference/v21_mirt_multigroup.json`, mirt 1.48, `NR`, `TOL = 1e-10`). mirt stops short of
  the maximum along one factor's ridge (that factor's focal mean/variance traded against its
  slopes): flexpyirt reproduces mirt's log-likelihood at mirt's estimates exactly, yet its own
  solution is 3.2e-5 higher, and the score at mirt's estimates is ~0.035 / ~0.018 for that
  factor's focal variance / mean (< 1e-3 elsewhere) while flexpyirt's is < 1e-6. The gap did not
  shrink from 31 to 81 points per dimension (issue #20, item 13). Hence parameters and SEs
  outside that factor's block (its items' slopes and intercepts, its focal mean/variance) are
  compared at the usual tolerances (`1e-4`, SE `1e-3` relative), and the block is checked by
  likelihood (flexpyirt higher) and stationarity. Oakes equals the direct Hessian, EM and SQUAREM
  are monotone, scoring and JSON round-trip work. Tests: `tests/test_mirt_m4_multigroup.py`.
- **S-X2 item fit (V15, M5)**: six cases reuse existing references (V3 GRM and V8 GPCM on
  Science, V8 PCM with a free latent variance, V11 3PL, V5 FIPC 2PL with eight fixed items, and
  V3 again with `mincell = 3`). mirt evaluates `itemfit(m, fit_stats = "S_X2")` at its reference
  estimates with every parameter's `est` flag kept (so free parameters count against the df) and
  `TOL = NaN`, on the same grid as the fit (mirt 1.48, `validation/r/m5_itemfit_reference.R` →
  `tests/reference/v15_sx2.json`). flexpyirt refits the data: degrees of freedom are identical in
  every case, S-X2 agrees to `2.2e-8` (relative) and p-values to `1.3e-8`, the level at which the
  estimates themselves agree. Unit tests check the Lord–Wingersky recursion against brute-force
  enumeration and that expected rows are conditional probabilities. Tests:
  `tests/test_mirt_m5_itemfit.py`. Multiple groups (DESIGN §15.45): mirt's per-group `itemfit()`
  on V4's two-group 2PL at its estimates (rebuilt with every parameter fixed, then each `est`
  flag restored, because `multipleGroup(TOL = NaN)` still takes one EM step), and on Science (GRM)
  split into two groups by row parity with all items invariant and the focal mean/variance free
  (`validation/r/m5_itemfit_multigroup_reference.R` → `tests/reference/v15_sx2_multigroup.json`).
  mirt stops ~2e-4 short in Science's item parameters (flexpyirt's log-likelihood is higher), so
  flexpyirt is evaluated at mirt's estimates there. Degrees of freedom are identical and S-X2
  agrees within `1e-6` (relative) in both groups of both cases. Tests:
  `tests/test_mirt_m5_itemfit_multigroup.py`.
- **$M_2$ / $M_2^*$ and S-X2 for confirmatory multidimensional models (V39, M6; DESIGN §16.19)**:
  the four V14 models (2D 2PL simple structure, 2D 2PL with the cross-loading `q07`, 2D GRM
  $K=4$, 3D 2PL; one free correlation in 2D, three in 3D) are rebuilt in mirt 1.48 at the V14
  estimates with every `est` flag kept and `TOL = NaN`, and `M2()` and
  `itemfit(fit_stats = "S_X2")` are evaluated on the product grid of the fit ($31^2$ and $15^3$
  nodes on $[-6,6]$, passed explicitly because neither function reads the grid of the fit)
  (`validation/r/m6_fit_multidim_reference.R` → `tests/reference/v39_fit_multidim.json`). This is
  a new ID rather than an extension of V15 / V18 because the cases, the R script and the
  reference file are separate. Tests: `tests/test_mirt_m6_fit_multidim.py` (slow tier, except the
  S-X2 comparison of the 2D simple-structure case and the rejected model structures, which run in
  the fast tier) and the unit tests
  `tests/test_m6_fit_multidim.py` (expected S-X2 tables, $\pi_2$ and $\Xi_2$ on a product grid
  against enumeration of every response pattern; the correlation column of $\Delta_2$ against a
  central difference).
  - **S-X2**: degrees of freedom identical for every item in all four models (and for the GRM
    with `mincell = 3`); at mirt's estimates S-X2 agrees to `2.7e-14` (relative) and p-values to
    `1.2e-14`; refitted from default starting values, S-X2 to `1.2e-7` and p-values to `5.0e-8`
    (the level at which the estimates agree, largest in the 3D case). mirt's `collapseCells`
    port is used unchanged; no difference in collapsing or in the quadrature was found.
  - **$M_2$, mirt's convention**: mirt's `M2()` sets the `GroupPars` `est` flags to `FALSE` when
    it builds $\Delta_2$, so the free correlations have no column there although $df$ subtracts
    them (as for `rasch_free_var` in V18). With `_latent_in_delta=False`, flexpyirt reproduces
    mirt's raw $M_2$ to `7.4e-13` (relative) at mirt's estimates and `1.9e-8` refitted; $df$ is
    identical, SRMSR agrees to `1.5e-14` / `3.4e-8`, and TLI / CFI (three dichotomous cases) to
    `4.5e-9`. The R script also recomputes mirt's value from its own pieces (`dP`,
    `buildXi2els`) and asserts agreement to `1e-10`.
  - **$M_2$, flexpyirt's default** (every free parameter in $\Delta_2$): compared with an exact R
    computation that adds one analytic column per free correlation (checked in R against a
    central difference of the normalized prior): `2.5e-13` at mirt's estimates, `5.3e-9`
    refitted. The default is smaller than mirt's raw value by `4.7e-5` (2D simple), `2.8e-7` (2D
    cross-loading), `2.5e-4` (2D GRM) and `6.9e-4` (3D), relative.
  - **A numerical note**: for the GRM, $\Delta_2$ under item-score margins is ill-conditioned;
    central-difference item derivatives (step $10^{-6}$, error $\sim10^{-10}$) shift $M_2$ by
    `4e-6` relative, so the R reference uses mirt's analytic `dP` for the item columns (correct
    for 2PL and graded items) and flexpyirt uses autodiff.
  - **Not validated**: S-X2 for multiple-group multidimensional models (the V21 configurations;
    it raises `SpecificationError`, as do both statistics for bifactor and two-tier models;
    $M_2$ for multiple groups is V42), 3PL / GPCM / nominal items
    in a multidimensional model (the item kernels are those validated in V18 / V15 and the unit
    tests cover a mixed 2PL + GRM model, but there is no mirt comparison), and grids other than
    mirt's defaults.
- **$M_2^*$ for multiple groups, missing data and MAP fits (V42, M6, #76; DESIGN §16.23)**: six
  simulated cases fitted by mirt 1.46.1 (`multipleGroup()` / `mirt()`, `TOL = 1e-10`, $[-6,6]$
  with 61 points, 31 per dimension for two factors; `validation/r/m6_m2_groups_missing_map_reference.R`
  → `tests/reference/v42_m2_groups_missing_map.json`): `twogroup_2pl` (2PL, $J=6$, 700 / 300,
  items 1–3 equal across groups, focal mean and variance free), `twogroup_2pl_missing` (the same
  with 15% MCAR), `twogroup_grm` (GRM $K=3$, $J=5$, 500 / 500, all items equal),
  `twogroup_2d` (two-factor 2PL, $J=8$, 600 / 400, all items equal, focal means, variances and
  correlation free), `missing_2pl` (2PL, $J=7$, $N=800$, 20% MCAR) and `map_2pl` (2PL, $J=6$,
  $N=250$, `lnorm(0, 0.5)` on $a$ and `norm(0, 2)` on $d$). mirt's `M2()` is the raw reference.
  The R script also computes `exact`, the formulas of DESIGN §16.23 written independently in R
  (central-difference $\Delta$ over every free parameter with the group rows scaled by
  $\sqrt{N_g/N}$, mirt's `buildXi2els` per group rescaled by $N_gN_{ab}/(N_aN_b)$, the
  multiple-group independence model), and the same code with mirt's conventions, which must
  reproduce mirt's raw $M_2$ (asserted in R). Tests: `tests/test_mirt_m6_m2_groups.py` (slow tier)
  and unit tests in `tests/test_m5_m2.py`.
  - **At mirt's estimates**, flexpyirt's default agrees with `exact` to `5.5e-10` (relative) in
    every dichotomous case and `3.8e-7` for the GRM (the central-difference error of the R
    reference, as in V39), with identical $df$, TLI / CFI to `1.1e-10`, and SRMSR per group equal
    to mirt's to `7.8e-15`. With `_mirt_convention=True` and `_latent_in_delta=False` flexpyirt
    reproduces mirt's raw $M_2$ to `1.8e-14`, and its TLI / CFI to `9.4e-8` (mirt fits the null
    model by EM).
  - **Refitted** by flexpyirt, the log-likelihood (log-posterior for MAP) agrees with mirt's to
    `6.1e-10` and $M_2$ with `exact` to `3.0e-5` (the estimates agree to about `1e-5`).
  - **mirt versus the exact statistic**: mirt is `0.45%` and `0.37%` larger for `twogroup_2pl`
    and `twogroup_2d` (group rows unscaled, latent columns dropped), `2.6%` for `twogroup_grm`
    (latent columns), `5.8%` and `18.3%` for `missing_2pl` and `twogroup_2pl_missing` (the
    `N.ratio` weighting), and identical for `map_2pl` (mirt ignores the prior, as flexpyirt does).
  - **Calibration by simulation** (not a mirt comparison; scratch scripts, summarized here and in
    the slow unit test `test_m2_with_mcar_missing_follows_chi2_at_true_parameters`): 2PL, $J=6$,
    $N=600$, 1000 replications at the true parameters ($df=21$): with 30% MCAR on every item the
    exact statistic has mean `20.86 ± 0.21`, variance `42.9` (42) and 5.3% rejections at .05
    (KS $p=0.84$), mirt's weighting mean `19.12` (KS $p=10^{-12}$); with uneven MCAR rates
    (0, 0, 10, 40, 60, 70%) `20.86 ± 0.21`, 5.6% (KS $p=0.12$) against mirt's `19.90` with
    variance `59.6` (KS $p=10^{-14}$). With estimated parameters ($df=9$, 400 replications unless noted):
    complete data `9.22 ± 0.22` (6.0% rejections, KS $p=0.63$); 30% MCAR `9.04 ± 0.20`, 4.8%
    (KS $p=0.61$), mirt's weighting `8.94`; uneven rates `9.19 ± 0.22`, 6.0% (KS $p=0.64$) against
    mirt's `7.17`, 1.5% (KS $p=5\times10^{-17}$); uneven rates with $N=2000$ (300 replications)
    `8.92 ± 0.24`, 4.7% (KS $p=0.82$) against mirt's `6.81` with no rejection (KS $p=4\times10^{-14}$).
    Two groups of 900 / 150 with all items equal and the focal mean and variance free (400
    replications, $df=28$): mean `28.17`, 5.5% rejections (KS $p=0.29$); mirt's convention
    `26.72` (KS $p=0.004$).
  - **Not validated**: multiple-group `DavidianCurve` models (raise), polytomous null models (TLI /
    CFI stay `NaN`, §16.3), and MAR mechanisms that are not MCAR (the statistic has no reference
    distribution there; documented).
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
- **M2 / M2* limited-information goodness of fit (V18, M5)**: six cases on 61 points over $[-6,6]$
  (`validation/r/m5_m2_reference.R` → `tests/reference/v18_m2.json`, mirt 1.46.1): `lsat7_2pl` (V1),
  `sim_2pl` (V1), `threepl` (V11 3PL), `rasch_free_var` (V2 Rasch with free latent variance),
  `grm_sim` (GRM $K=3, J=10, N=2000$), and `gpcm_sim` (GPCM $K=4, J=8, N=2000$). Comparing against
  mirt 1.46.1's `M2()` uncovered four discrepancies in mirt 1.46.1 (all verified in R):
  1. **Free latent parameters omitted from $\Delta_2$**: mirt forces `GroupPars` entries to
     `est = FALSE` inside `M2()` while still subtracting them from `df`. flexpyirt includes all
     free parameters (item and latent) in $\Delta_2$ by default (Maydeu-Olivares & Joe, 2005,
     2006); internal `_m2.m2_statistic(..., _latent_in_delta=False)` reproduces mirt's
     `rasch_free_var` value (`M2` rel `9.7e-16`, `SRMSR` `4.9e-17`, `TLI` `1.6e-9` at mirt's
     estimates), while the default (`True`) matches R's exact-with-latent reference (`M2` rel
     `1.0e-12` at mirt's estimates, `9.5e-6` on refit due to the documented $7.9\times10^{-5}$
     latent variance difference on $[-6,6]$) and differs from mirt's `M2` by `0.035`.
  2. **`dP("dich")` bug for 3PL/4PL**: `mirt:::dP` for `dich` uses the 2PL derivative
     $P_0 P_1$ for `(a1, d)` instead of $(u-g)\sigma(1-\sigma)$ and places raw category
     probabilities $(P_0, P_1)$ in columns 3–4 (`cbind(Theta * PQ, PQ, P)`) instead of
     $((1-\sigma)g(1-g),\, \sigma u(1-u))$, shifting `threepl`'s `M2` by 4.8% (`64.82` vs exact
     `68.11`).
  3. **`dP("gpcm")` slope-derivative bug**: `mirt:::dP` for `gpcm` omits division by
     `den = rowSums(num)` in the `a1` derivative (`rowSums(Theta[, i] * aknum[[i]])`), shifting
     `gpcm_sim`'s `M2` by 10.8% (`5.27` vs exact `5.90`).
  4. **Polytomous null model rank deficiency**: for $K_j > 2$, item $j$'s $K_j - 1$ intercept
     columns in the null $\Delta_2$ are strictly collinear ($\operatorname{rank}=J < \sum(K_j-1)$);
     mirt slices `qr.Q(qr(delta), complete = TRUE)` after `ncol(delta)` without checking rank,
     making its polytomous null $M_2^*$ depend on item ordering. flexpyirt returns `NaN` for
     `tli` and `cfi` when any item is polytomous (DESIGN §16.3).

  Agreement (both plain EM and SQUAREM, exact `df` in every case):
  - **Direct vs mirt (`lsat7_2pl`, `sim_2pl`, `grm_sim`)**: `M2` rel $\le 3.8\times10^{-11}$,
    `p` $\le 7.8\times10^{-11}$, `RMSEA` $\le 4.4\times10^{-13}$, CI bounds $\le 1.4\times10^{-8}$,
    `SRMSR` $\le 1.4\times10^{-12}$; dichotomous `TLI` $\le 4.8\times10^{-10}$, `CFI` $\le 2.4\times10^{-10}$.
  - **Exact R $M_2^*$ (`threepl`, `gpcm_sim`, using central-difference derivatives of
    `expected.item()` + mirt's `buildXi2els`)**: `M2` rel $\le 1.6\times10^{-9}$, `p` $\le 7.1\times10^{-10}$,
    `RMSEA` $\le 3.7\times10^{-11}$, CI bounds $\le 6.0\times10^{-9}$, `threepl` `TLI` `1.8e-10` /
    `CFI` `0`, and `SRMSR` directly vs mirt $\le 8.5\times10^{-13}$.
  Tests: `tests/test_mirt_m5_m2.py` (mirt agreement), `tests/test_m5_m2.py` (unit tests).
- **Davidian-curve latent distributions (V19, M5)**: inspecting `mirt` 1.46.1 (`Theta_DC_den`,
  `Mstep.DC.optim`, `EM.group`, `LoadGroupPars`, `updatePrior`) and `dcurver` 0.9.3 established
  four key differences from exact discretized MMLE (DESIGN §16.4):
  1. Built-in `mirt(..., dentype = "Davidian-k")` applies Woods' linear-interpolation
     `standardizeQuadrature` to expected counts at every EM cycle before running `nlminb` on
     unnormalized `dcurver::ddc`, which breaks EM monotonicity (mirt disables its monotonicity
     check for `"Davidian"`) and ends with a strictly lower marginal log-likelihood (by `21.47`
     points on V19a's 2PL $k=3$, and by `139.32` points on V19b's GRM $k=2$). On V19a mirt's end
     point is not stationary even for its own discretized likelihood ($\sum\log\sum_q L_q\,
     \mathrm{ddc}(\theta_q)/\sum_q\mathrm{ddc}(\theta_q)$): the score in `PHI_1..3` is
     $(-0.54, -3.40, 56.7)$, and maximizing over `PHI` alone with mirt's item parameters gains
     `4.67`. On V19b mirt stalls at the stationary point $N(0,1)$ (`PHI_1 = 1.5707963`, score
     $(3.6\times10^{-4}, 0)$, no gain from re-optimizing `PHI`): the score in `PHI_2` vanishes
     because of the sine-first chart's polar singularity at $\phi_1=\pi/2$, and the score in
     `PHI_1` vanishes because its first-order direction is a pure location/scale change, which the
     free items absorb. (The second reason applies to any chart; flexpyirt leaves
     $\boldsymbol\phi=\mathbf 0$ through its perturbed default starts, DESIGN §16.4.)
  2. Built-in `dentype = "Davidian-k"` forbids `SE = TRUE` (`"No standard error method currently
     supported for Davidian curves"`) and forbids freeing `MEAN_1` / `COV_11` (`"Estimating
     Davidian mean-variance hyper-parameters is not currently supported"`), both in
     `mirt:::EM.group`.
  3. In `fscores()`, mirt defaults to an $N(0,1)$ prior unless `use_dentype_estimate = TRUE` is
     passed (which supports `EAP` and plausible values, and rejects `MAP`).
  4. When validating the exact standardized Davidian MMLE inside `mirt` via `createGroup()`
     (calling `dcurver::ddc` normalized on the grid, `TOL = 1e-10`, `accelerate = "none"`,
     `SE.type = "Oakes"`), `createGroup`'s default `derivType = "Richardson"` uses a coarse
     second-derivative step (`mirt:::numerical_deriv` uses `delta * 1000 = 0.01`, and
     `2*delta = 0.02` on the diagonal) that corrupts `k = 3` Oakes SEs by up to 25.6%, so
     `derivType = "central"` (`delta = 1e-5`) is used; and `mirt:::BL.LL` reads `pars` instead of
     `pars2` for `dentype == "custom"` (the latent density is not perturbed), so Richardson
     Hessian SEs are computed directly on the exact R marginal log-likelihood (`h = 1e-4`).

  **What the V19 references validate.** The `exact` block is the standardized Davidian MMLE
  of DESIGN §16.4, re-coded in R: the chart, the closed-form moments $(\mu_Z,\sigma_Z^2)$ and the
  standardization $z_q=\mu_Z+\sigma_Z(\theta_q-\mu)/\sigma$ are the same definitions as
  flexpyirt's (written separately in R), while the density values come from `dcurver::ddc`, the
  E-step, EM, Oakes SEs and EAP from `mirt` (`createGroup()`), and the Hessian SEs from a
  separately written R marginal log-likelihood. It therefore validates flexpyirt's estimation
  machinery (E-step, latent M-step, convergence, Oakes/Hessian SEs, scoring) for that model, and
  the density against `dcurver`. The closed-form moments are checked independently: the R script
  integrates $z\,\mathrm{ddc}(z)$ and $z^2\,\mathrm{ddc}(z)$ with `stats::integrate` at every
  fitted and built-in solution (max abs error `1.2e-9`, stored as `moments_check`; it stops if
  the error exceeds `1e-8`), and `tests/test_m5_davidian.py` checks them by trapezoidal
  integration. The `mirt_builtin` block checks the density, `Prior`, `logLik` and EAP against
  mirt's own Davidian implementation at mirt's parameters.
  **Convergence of the reference.** mirt's EM (BFGS M-step, `TOL = 1e-10`) stops with a gradient
  of the exact marginal log-likelihood of up to `1.7e-3` (V19c; `3.8e-5` and `5.8e-5` for V19a/b,
  stored as `max_grad_r_mirt_em`). The R script therefore polishes mirt's solution by Newton
  steps on the exact R marginal log-likelihood (central-difference gradient and Hessian) until
  max|grad| < `2e-5` (one step in each case, final `1.8e-6`–`3.6e-6`, `max_grad_r`) and re-runs
  `mirt` at the polished values with `TOL = NaN` (no EM cycles) to obtain Oakes SEs, `logLik` and
  EAP there. Before this polish the V19c item and latent differences were `8.5e-6` and `7.3e-6`,
  with flexpyirt's log-likelihood `1.8e-8` above the reference, i.e. the residual was in the
  reference, not in flexpyirt.

  Three simulated cases with skewed standardized $\chi^2_4$ traits ($N = 3000$) are validated
  in `tests/test_mirt_m5_davidian.py` (plain EM and SQUAREM, `validation/r/m5_davidian_reference.R`
  → `tests/reference/v19_davidian_*.json`; maxima over plain EM and SQUAREM):
  - **V19a (`v19_davidian_2pl_k3`, 2PL, $J=20, k=3$, 61 pts $[-6,6]$)**: exact MMLE vs R
    `mirt(customGroup)` / R Hessian: `loglik` rel `2.3e-16`, item params `5.7e-8`, `phi_1..3`
    `1.3e-8`, Oakes SE vs mirt Oakes `4.6e-6` (items `6.3e-7`), Hessian SE vs R Richardson
    `2.8e-6` (latent `3.8e-7`), flexpyirt Oakes vs Hessian `9.7e-16`, EAP `6.2e-15` (SQUAREM
    reduces EM evaluations `113 → 39`). At built-in `mirt(..., dentype="Davidian-3")` parameters
    (with `mean = mu_z`, `var = var_z`), flexpyirt reproduces mirt's `Prior`, `logLik` (`1.4e-15`
    rel), and `fscores(..., use_dentype_estimate=TRUE)` (`8.2e-15`), while exact MMLE gains
    `+21.47` log-likelihood points.
  - **V19b (`v19_davidian_grm_k2`, GRM $K=4$, $J=10, k=2$, 61 pts $[-6,6]$)**: exact MMLE:
    `loglik` rel `1.9e-16`, item params `4.9e-8`, `phi_1..2` `9.1e-9`, Oakes SE vs mirt Oakes
    `3.6e-7`, Hessian SE vs R Richardson `1.9e-6`, flexpyirt Oakes vs Hessian `1.1e-15`, EAP
    `6.0e-15` (SQUAREM `111 → 33`). At built-in `mirt(..., dentype="Davidian-2")` parameters,
    `logLik` (`1.9e-16` rel) and EAP (`6.7e-15`) match identically, while exact MMLE gains
    `+139.32` log-likelihood points.
  - **V19c (`v19_davidian_fipc_k3`, FIPC 2PL, `q01–q10` fixed, $J=20, k=3$, free `mean`/`var`,
    shifted/scaled trait $0.4 + 1.2\cdot\text{skewed}$, 81 pts $[-8,8]$)**: exact MMLE: `loglik`
    rel `1.6e-15`, item params `7.5e-8`, latent params (`mean, var, phi_1..3`) `1.4e-7`, Oakes SE
    vs mirt Oakes `6.0e-6`, Hessian SE vs R Richardson `1.8e-6`, flexpyirt Oakes vs Hessian
    `2.9e-15`, EAP `5.3e-15` (SQUAREM `107 → 33`).
  The remaining `~1e-6` SE differences come from mirt's and R's finite-difference derivatives
  (`delta = 1e-5`, `h = 1e-4`); flexpyirt's SEs use exact autodiff. Tolerances are those of the
  table above (`1e-4` parameters, `1e-3` relative SEs).
  Unit tests (unit sphere, moments, zero-angle $N(0,1)$ equivalence, polynomial-root autodiff
  guard and fits with a root on a quadrature node, start-angle signs, fits with `degree` 1 and
  10, Rasch/FIPC, identification/spec errors, scoring, JSON round trip and malformed
  `DavidianCurve` JSON): `tests/test_m5_davidian.py`.
- **Differentiable IRT scale linking (V20, M5)**: `mirt` does not implement characteristic-curve
  linking, so `validation/r/m5_linking_reference.R` (→ `tests/reference/v20_linking.json`)
  calibrates two separate forms with common items in `mirt` (`optimizer = "NR"`,
  `accelerate = "none"`, `TOL = 1e-10`, $N(0, 1)$, 61 equispaced nodes on $[-6, 6]$,
  `SE.type = "Oakes"`; target $\theta\sim N(0, 1)$, source $\theta\sim N(0.5, 1.2^2)$, true
  $A = 1.2, B = 0.5$; the V20e source form is calibrated under a fixed $N(0.3, 1.5)$, so the
  symmetric reverse term uses a non-standard source grid) and computes reference $(A, B)$,
  criterion values, and Ogasawara (2001)
  delta-method SEs in base R directly from the formulas (Haebara, 1980; Stocking & Lord, 1983;
  Kim & Lee, 2006) using `mirt::probtrace()` / `mirt::expected.item()` and `optim(BFGS)` + `nlm`
  with central-FD Newton polish (`max |grad| <= 1.9e-11`). Haebara sums over all categories,
  including both categories of a dichotomous item (Kim & Lee, 2006). Six cases are checked across
  all four variants (Stocking–Lord and Haebara, non-symmetric and symmetric) in three ways:
  (i) linking on identical mirt item estimates (separating linking error from calibration error);
  (ii) delta-method SEs from mirt's own common-item estimates and Oakes covariance blocks
  (separating the SE formula from calibration and covariance differences); and (iii) end-to-end
  (fitting both forms in flexpyirt and linking). The Newton solver is also run from four far-off
  starts ($(A, B) = (0.25, -3), (4, 3), (0.5, 2.5), (3, -2)$) on every case and variant and must
  reach the reference optimum within 1e-9. No external R linking package (e.g. `plink`,
  `equateIRT`) is used: the criteria and SEs are checked against an independent implementation
  of the same published formulas, not against third-party software.
  - **V20a (`v20a_2pl`, 2PL, $J=20$/form, 10 common, $N=2000$/group)**: direct `max |Δ(A, B)| = 9.6e-13`,
    criterion rel `1.6e-13`, SE rel on mirt's covariance `2.2e-8`; end-to-end
    `max |Δ(A, B)| = 1.9e-10`, criterion rel `2.6e-9`, Ogasawara SE rel `2.2e-8`.
  - **V20b (`v20b_grm`, GRM $K=4$, $J=16$/form, 8 common, $N=2000$/group)**: direct `max |Δ(A, B)| = 9.1e-13`,
    criterion rel `1.0e-13`, SE rel on mirt's covariance `8.3e-9`; end-to-end
    `max |Δ(A, B)| = 9.5e-10`, criterion rel `2.2e-9`, Ogasawara SE rel `8.1e-9`.
  - **V20c (`v20c_gpcm`, GPCM $K=4$, $J=16$/form, 8 common, $N=2000$/group)**: direct `max |Δ(A, B)| = 2.5e-12`,
    criterion rel `9.5e-14`, SE rel on mirt's covariance `1.6e-8`; end-to-end
    `max |Δ(A, B)| = 1.4e-9`, criterion rel `7.1e-10`, Ogasawara SE rel `1.6e-8`.
  - **V20d (`v20d_3pl`, 3PL, $J=16$/form, 8 common, $N=4000$/group, interior $g$)**: direct
    `max |Δ(A, B)| = 3.6e-12`, criterion rel `1.2e-13`, SE rel on mirt's covariance `9.4e-8`;
    end-to-end `max |Δ(A, B)| = 3.0e-10`, criterion rel `5.2e-9`, Ogasawara SE rel `9.4e-8`.
  - **V20e (`v20e_2pl_srcgrid`, V20a data, source calibrated under fixed $N(0.3, 1.5)$)**: direct
    `max |Δ(A, B)| = 9.1e-13`, criterion rel `1.4e-13`, SE rel on mirt's covariance `3.4e-9`;
    end-to-end `max |Δ(A, B)| = 4.2e-10`, criterion rel `7.0e-10`, Ogasawara SE rel `3.4e-9`.
  - **V20f (`v20f_mixed`, mixed 2PL + GPCM $K=4$, $J=16$/form, 5 + 3 common, $N=2000$/group)**:
    direct `max |Δ(A, B)| = 2.0e-11`, criterion rel `6.2e-14`, SE rel on mirt's covariance
    `2.1e-8`; end-to-end `max |Δ(A, B)| = 1.3e-10`, criterion rel `3.6e-10`, Ogasawara SE rel
    `2.1e-8`.
  Validated item models for linking in V20 are therefore 2PL, 3PL, GRM, GPCM and mixed 2PL + GPCM
  (4PL and nominal: V40 below); Rasch and PCM share these kernels and are covered by
  exact-recovery unit tests only.
  Tests: `tests/test_mirt_m5_linking.py`. Unit tests (all 7 unidimensional models, ICC/threshold
  invariance, exact identity & recovery, symmetric reversibility, custom grids, EH/multiple-group,
  joint covariance when one multiple-group fit is both source and target, no SEs without
  convergence, Rasch/PCM slope $1/A$, `transform_theta`, finite-difference gradient & Ogasawara SE
  checks vs finite differences and a Monte Carlo linearization check that re-solves the linking
  for parameter draws from $N(\hat\eta, \Sigma)$ with fixed weights — not a bootstrap of the
  calibrations): `tests/test_m5_linking.py`.
- **Linking for 4PL, nominal and multidimensional models (V40, M6, #78)**: an extension of V20
  with the same method. `validation/r/m5_linking_extensions_reference.R` calibrates two forms with
  common items in `mirt` (`accelerate = "none"`, `TOL = 1e-10`, `SE.type = "Oakes"`, latent means
  0 and variances 1 in both calibrations) and computes the linking in base R, independently of
  flexpyirt: the criteria from `mirt::probtrace()` / `mirt::expected.item()` on the calibration
  grid, minimized with `optim(BFGS)` + `nlm` and a central-finite-difference Newton polish
  (`max |grad| <= 4.2e-11`, Hessian checked positive definite), and the Ogasawara (2001)
  covariances from category probabilities written out in R with nested central finite
  differences (step 1e-5). Every case is run for Stocking–Lord and Haebara, non-symmetric and
  symmetric, and checked (i) on identical mirt item estimates with mirt's Oakes covariance
  blocks — at the core level, and through `link()` on `FitResult`s fixed at mirt's estimates
  with the library's own starting values — and (ii) end-to-end (both forms calibrated in
  flexpyirt). Each reference optimum is also compared with the simulated transformation, which
  it recovers within 4 delta-method standard errors.
  **Unidimensional** (`tests/reference/v40_linking_unidim.json`, `tests/test_mirt_v40_linking.py`;
  61 nodes on $[-6, 6]$, `optimizer = "NR"`; target $\theta\sim N(0,1)$, source
  $\theta\sim N(0.5, 1.2^2)$, true $A = 1.2$, $B = 0.5$). The solver is also started from
  $(1, 0)$, $(0.4, -2)$ and $(3, 2.5)$.
  - **V40a (`v40a_4pl`, 4PL, $J=12$/form, 8 common, $N=20000$/group, interior $g$ and $u$; mirt
    started at $a = 2$ as in V11b)**: direct `max |Δ(A, B)| = 5.5e-12`, criterion rel `2.1e-14`,
    covariance rel `4.1e-7`; end-to-end `max |Δ(A, B)| = 1.2e-9`, criterion rel `6.8e-8`,
    covariance rel `4.2e-7` (item parameters agree with mirt to `2.5e-8`). The covariance blocks
    include $g$ and $u$ (natural scale).
  - **V40b (`v40b_nominal`, nominal $K=4$, $J=12$/form, 6 common, $N=3000$/group; estimated
    scoring coefficients include negative values)**: direct `max |Δ(A, B)| = 1.2e-12`, criterion
    rel `5.5e-14`, covariance rel `4.2e-7`; end-to-end `max |Δ(A, B)| = 3.3e-9`, criterion rel
    `1.3e-8`, covariance rel `4.4e-7`. Stocking–Lord scores categories $0..K-1$ as
    `expected.item()` does.
  - **V40c (`v40c_mixed`, 2PL + nominal $K=3$, $J=15$/form, 6 + 3 common, $N=3000$/group)**:
    direct `max |Δ(A, B)| = 1.9e-12`, criterion rel `2.6e-14`, covariance rel `1.6e-7`;
    end-to-end `max |Δ(A, B)| = 2.2e-10`, criterion rel `5.7e-10`, covariance rel `1.6e-7`.
  **Multidimensional** (`tests/reference/v40_linking_multidim.json`,
  `tests/test_mirt_v40_linking_multidim.py`; DESIGN §19). Confirmatory models with free factor
  correlations; the target population is standard and the source population has factor means
  $\boldsymbol\mu$ and standard deviations $\boldsymbol\sigma$, so the true transformation is
  $\mathbf A = \operatorname{diag}(\boldsymbol\sigma)$, $\boldsymbol\beta = \boldsymbol\mu$. Each
  case is solved for the three families of $\mathbf A$ (12 variants per case). The R reference
  parameterizes the rotation in closed form (cosine/sine for two factors, Rodrigues' formula for
  three), independently of flexpyirt's matrix exponential. The core check starts from the
  identity transformation. Differences are the largest over the 4 criteria; covariance
  differences are relative to the largest element of the $(D^2+D)$-square reference matrix.
  The simulated transformation is recovered within 4 standard errors by the diagonal and
  oblique families (the orthogonal family, with one common dilation, does not contain it).

  | Case | Family | Direct `max \|Δ(A, β)\|` | Direct covariance | End-to-end `max \|Δ(A, β)\|` | End-to-end covariance |
  |---|---|---|---|---|---|
  | V40d (`v40d_2pl_2d`): 2PL, 2 factors, $J=16$/form, 10 common (2 cross-loading), $N=3000$/group, 31 nodes per dimension, $\boldsymbol\mu=(0.4,-0.3)$, $\boldsymbol\sigma=(1.2,0.8)$ | diagonal | `1.4e-11` | `2.9e-7` | `4.0e-10` | `2.9e-7` |
  | | orthogonal | `3.2e-11` | `8.2e-7` | `4.1e-10` | `8.2e-7` |
  | | oblique | `3.2e-11` | `4.8e-7` | `3.6e-10` | `4.8e-7` |
  | V40e (`v40e_grm_2d`): GRM $K=3$, 2 factors, $J=12$/form, 8 common (2 cross-loading), $N=3000$/group, 31 nodes per dimension, $\boldsymbol\mu=(-0.3,0.5)$, $\boldsymbol\sigma=(0.85,1.25)$ | diagonal | `4.3e-12` | `1.7e-7` | `3.8e-10` | `1.7e-7` |
  | | orthogonal | `1.5e-11` | `8.6e-7` | `4.1e-10` | `8.6e-7` |
  | | oblique | `2.9e-11` | `1.2e-7` | `4.1e-10` | `1.2e-7` |
  | V40f (`v40f_2pl_3d`): 2PL, 3 factors, $J=15$/form, 12 common (3 cross-loading), $N=3000$/group, 15 nodes per dimension, mirt `optimizer = "BFGS"` (`reltol = 1e-13`), $\boldsymbol\mu=(0.3,-0.2,0.4)$, $\boldsymbol\sigma=(1.2,0.9,1.1)$ | diagonal | `2.8e-11` | `6.1e-7` | `1.5e-8` | `6.2e-7` |
  | | orthogonal | `2.2e-10` | `1.6e-6` | `7.0e-9` | `1.5e-6` |
  | | oblique | `1.5e-10` | `1.4e-6` | `2.3e-8` | `1.3e-6` |

  Criterion values agree to `7.9e-14` (direct) and `1.3e-7` (end-to-end) in relative terms;
  `link()` on the fixed mirt estimates gives the same constants as the core check (at most
  `2.2e-10` from R) in 3–6 Newton iterations from its least-squares start. Item parameters of the
  flexpyirt calibrations agree with mirt to `5.1e-10` (two factors) and `1.3e-7` (three factors,
  where mirt uses BFGS).
  **Not validated**: multidimensional 3PL/4PL, GPCM and nominal common items (their
  transformation and the solver are covered by exact-recovery unit tests only); multiple-group
  multidimensional fits as `source` or `target`; reflections ($\det\mathbf A < 0$), which are
  not searched; bifactor and two-tier models, which are rejected. No external R linking package
  is used. Unit tests (curve invariance for every kernel and family, the three families'
  closed forms, the one-factor criterion equal to §16.5's, finite-difference gradients, exact
  recovery of known transformations for two and three factors, the delta-method Jacobian
  against re-solving the linking, `transform_theta`, errors): `tests/test_m6_linking_multidim.py`.
- **Common-item screening (V50, #105)**: no R package implements the robust z, the displacement
  or the expected-score RMSD together with characteristic-curve linking and purification, so
  `validation/r/v50_screening_reference.R` (→ `tests/reference/v50_screening.json`, mirt 1.48,
  equateIRT 2.5.2) recomputes the rules of DESIGN §22 in base R: each form is calibrated with
  `mirt` (`optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`, 61 nodes on $[-6, 6]$, latent
  $N(0, 1)$); the linking on each anchor set is the V20 base-R Stocking–Lord / Haebara (asserted
  `max |grad| < 1e-9`); locations, `log_ratio_a`, robust z (`median()`,
  `quantile(type = 7)`, `IQR()`), displacement and the RMSD of `expected.item()` curves on the
  target grid follow the formulas, and the purification loop is written out with the same removal
  rule. `tests/test_r_v50_screening.py` runs `screen_common_items()` on `FitResult`s fixed at
  mirt's estimates and compares the removal order, $(A, B)$ and the criterion at every step, the
  final item table, the flags and the `ScreeningWarning`. Source populations $N(0.4, 1.2^2)$ on
  the target scale; drift planted in the source form.
  - **V50a (`v50a_2pl`, 2PL, 12 common + 8 unique items per form, $N=2000$/form; drift: `c01`
    $b + 0.8$, `c02` $a \times 1.8$, `c03` $b - 0.6$)**: seven variants (robust z with
    `removal="largest"` and `"all"`, displacement, RMSD, displacement with symmetric Haebara,
    robust z with `max_iter=1`, RMSD 0.02 with `min_items=11`). Removal orders agree exactly
    (robust z removes `c02, c01, c03, c10`; displacement and RMSD `c01, c03`); over all steps
    `max |Δ(A, B)| = 2.3e-11`, per-item quantities `5.0e-12`, robust z `2.7e-10`.
    `equateIRT::dif.test` (Lord's Wald test after Stocking–Lord linking, with purification,
    $\alpha = 0.05$) flags `c01, c02, c03, c12`: the three planted drifts, which robust z also
    removes first.
  - **V50b (`v50b_mixed`, 2PL + GRM $K=4$ + GPCM $K=4$, 6 + 3 + 3 common, $N=2500$/form; drift:
    `d01` $b + 0.7$, `g01` thresholds $+0.6$, `p02` $a \times 0.5$)**: four variants (robust z,
    displacement, RMSD, RMSD with Haebara and `removal="all"`). Removal orders agree exactly
    (robust z `p02, d01, g01`; RMSD `g01, p02, d01`; displacement `d01, g01`);
    `max |Δ(A, B)| = 1.0e-11`, per-item quantities `4.8e-12`, robust z `3.4e-10`.
  Tolerances: $(A, B)$ `1e-9`, criterion rel `1e-7`, per-item quantities `1e-9`, robust z
  `1e-8`. **Checked only independently** (unit tests in `tests/test_screening.py`): the
  location of 4PL and nominal items ($b^* = A\,b + B$ for every kernel), the RMSD against a direct
  evaluation of the moved source curve, exact recovery of $(A, B)$ on drift-free items, the removal of
  planted drifts among slightly perturbed items, the NaN robust z of Rasch slope ratios, the `items=`
  subset and `stack_screening_tables()`. The thresholds are conventions (DESIGN §22); V50 checks
  that the rules are computed as specified, not that the thresholds are good.
- **Classification accuracy and consistency (V52, #107)**: DESIGN §18.4 in
  `tests/test_r_classification.py` (slow tier; generator `validation/r/classification_reference.R`
  → `tests/reference/v52_classification.json`, mirt 1.48, cacIRT 1.4, 61 points on $[-6, 6]$).
  Each case's parameters go into a flexpyirt `FitResult` with every parameter fixed; mirt's EAP
  scores are stored for the P approach. Cases: `threepl` (3PL, $J=20$, $N=3000$, generating
  parameters fixed in mirt because a 3PL fit of this size did not converge), `grm` (GRM $K=4$,
  $J=10$, $N=2000$, mirt estimates) and `twogroup` (2PL $J=15$, $N=1500$ per group, all items
  invariant, focal mean and variance free). Runs: Lee with summed-score cuts 12, (8, 14),
  (7.5, 13.2) and with $\theta$ cuts (−0.5, 0.8) converted by $\tau$; Rudner with $\theta$ cuts
  0.3, (−0.5, 0.8) and with summed-score cuts (8, 14) converted by $\tau^{-1}$ (`uniroot` in R);
  each with the D approach (fitted nodes and weights) and the P approach (mirt's EAP scores). Two
  references per run:
  - **cacIRT** (Lathrop, 2015): `class.Lee()` for the 3PL, `Lee.poly.D()` / `Lee.poly.P()` for
    the GRM, and `Rud.D()` / `Rud.P()` with the SE $1/\sqrt{I(\theta)}$ from mirt's
    `testinfo()`: "Simultaneous" (overall) and "Marginal" (per cut) accuracy and consistency and
    the conditional values. `class.Rud()` is not used: cacIRT's `iif()` writes the 3PL
    information with $(1-c^2)$ where Birnbaum's formula has $(1-c)^2$, so its SE is wrong
    whenever $c>0$ (it is right for the 2PL). Largest differences: exact functions `1.2e-15`
    (D) and `2.4e-15` (P); `Lee.poly.D()` `3.9e-12`, because cacIRT clips the category
    probabilities to $[10^{-4}, 1-10^{-4}]$ before its recursion. cacIRT's `class.Lee()` puts a
    true score equal to a cut below it in the D approach (`cut(right = TRUE)`) and above it in
    the P approach; flexpyirt puts it above in both (no case hits a tie).
  - **Recomputation in R** from mirt's `probtrace()`, `expected.test()` and `testinfo()` with a
    Lord–Wingersky recursion written in the script: the tables $A$ and $B$, kappa, the per-cut
    false positive / negative rates and the per-category indices (none of which cacIRT has), and
    the conditional values. The script asserts that it reproduces cacIRT's exact functions to
    $10^{-12}$. Largest differences: D `2.0e-14`, P `4.4e-15`; flexpyirt's EAP scores against
    mirt's `7.8e-15`. Tolerances $10^{-12}$ (D, P, EAP) and $10^{-10}$ against `Lee.poly.D()`.
  - **Without R** (`tests/test_classification.py`): brute-force enumeration of all 72 response
    patterns of a mixed 2PL + 3PL + GRM $K=3$ + GPCM $K=3$ test with the item probabilities
    written out in NumPy (tables, conditional values, kappa; `1e-14`), Rudner against
    `scipy.stats.norm` with `test_information()`, identities between the indices (tables sum to
    1, equal observed margins, accuracy + false positive + false negative = 1 per cut), the
    scale conversions against `expected_score()`, multiple groups and the errors (fast tier).
    Slow tier: a Monte Carlo simulation of two parallel administrations ($N=200\,000$,
    $\theta\sim N(0,1)$) agrees with the D approach on a 2401-node grid within 4.5 Monte Carlo
    SEs for accuracy, consistency and the accuracy table, for Lee and Rudner.
  - **Discretization**: on the default 61-node grid the D approach's accuracy differs from the
    continuous-latent value (2401 nodes) by up to `7.6e-3` (Lee, 5-item mixed test) and `2.5e-3`
    (Rudner); consistency and kappa by `1e-9`. This is the jump of the conditional accuracy
    where the true category changes (DESIGN §18.4), not an error of the implementation; cacIRT
    gives the same value on the same grid.
  - **Not validated**: `sirt::class.accuracy.rasch` (Rasch, cuts on $\theta$) was not used,
    since cacIRT covers the same indices; latent distributions other than `Normal` (the code
    path is the same weights; not compared with R).
- **Bifactor models (V32, M7)**: two simulated cases in `tests/test_mirt_m7_bifactor.py` (plain
  EM and SQUAREM; generator `validation/r/m7_bifactor_reference.R` → `tests/reference/v32_bifactor.json`,
  mirt 1.46.1 `bfactor()`, `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`,
  `SE.type = "Oakes"`, 31 points per dimension on $[-6, 6]$, i.e. one $31^2$ grid per specific
  factor). mirt's `fscores()` has no bifactor path and integrates over the full four-dimensional
  product grid, which equals the factorized posterior on the same one-dimensional grid; EAPs are
  therefore compared at 15 points per dimension ($15^4$ nodes in mirt) with every parameter fixed at
  mirt's estimates. In both cases the reduced log-likelihood at mirt's estimates equals mirt's to
  $10^{-15}$ (relative), EM and SQUAREM are monotone, and Oakes agrees with the Hessian to
  $3\times10^{-16}$ (covariance entries):
  - **V32a (`bifactor_2pl`, 2PL, $J=15, N=2000$, `s1 = q01-q04`, `s2 = q05-q08`, `s3 = q09-q12`,
    `q13-q15` general only)**: `loglik` rel `2.2e-16`, item parameters `1.9e-7` (EM, 498 iterations;
    SQUAREM `5.3e-9`, 80 iterations), Oakes SE rel `3.5e-7` (SQUAREM `8.9e-9`), EAP $\theta$ `4.9e-14` /
    SD `8.7e-15`.
  - **V32b (`bifactor_grm`, GRM $K=4$, $J=12, N=2000$, three testlets of four items)**: `loglik` rel
    `8.9e-16`, item parameters `1.9e-8` (EM, 188 iterations; SQUAREM `2.5e-9`, 29 iterations), Oakes
    SE rel `2.5e-8` (SQUAREM `3.7e-9`), EAP $\theta$ `4.9e-14` / SD `7.9e-15`.
  - The unit tests (`tests/test_m7_bifactor.py`) check the reduced likelihood, expected counts and
    EAP against brute-force integration over the full $(1+S)$-dimensional product grid (to
    $10^{-12}$ or better, with missing data, a GRM item, 3PL items and general-only items), and a
    one-specific-factor bifactor model against the validated 2D path
    (`with_factors(..., correlated=False)`; log-likelihood to $10^{-12}$).
- **Two-tier models (V33, M7)**: two simulated cases in `tests/test_mirt_m7_two_tier.py` (plain
  EM and SQUAREM; generator `validation/r/m7_two_tier_reference.R` → `tests/reference/v33_two_tier.json`,
  mirt 1.48 `bfactor(data, model, model2 = )`, `accelerate = "none"`, `TOL = 1e-10`,
  `SE.type = "Oakes"`, 15 points per dimension on $[-6, 6]$, i.e. one $15^3$ grid per specific
  factor). As in the 3D case of V14, the M-step optimizer is `BFGS` with
  `control = list(reltol = 1e-13, maxit = 500L)`, because mirt's `NR` uses numerical Hessians for
  `GroupPars` and is very slow on 3,375 nodes. The primary correlation is mirt's `COV_21` (both
  variances are fixed at 1). mirt's `fscores()` integrates over the full product grid, which equals
  the factorized posterior on the same one-dimensional grid; EAPs are compared with every parameter
  fixed at mirt's estimates. In both cases the reduced log-likelihood at mirt's estimates equals
  mirt's to $2.1\times10^{-15}$ (relative), `flexpyirt.loglik >= fixed_at_mirt.loglik` holds, EM
  and SQUAREM are monotone, and Oakes agrees with the Hessian to $6.2\times10^{-17}$ (covariance
  entries):
  - **V33a (`two_tier_2pl`, 2PL, $J=16, N=2000$, `F1 = q01-q08`, `F2 = q09-q16`, $\rho=0.5$,
    `s1 = q01-q04`, `s2 = q09-q12`, `s3 = q05, q06, q13, q14`; four items primary only)**: `loglik`
    rel `2.1e-15`, item parameters `3.0e-7` (EM, 192 iterations; SQUAREM `3.1e-7`, 33 iterations),
    `cor_12` `2.2e-8` (SQUAREM `1.3e-8`), item Oakes SE rel `4.4e-7` (SQUAREM `4.6e-7`), `cor_12_se`
    rel `7.3e-6`, EAP $\theta$ `6.9e-15` / SD `4.0e-15` ($9^5$ nodes in mirt).
  - **V33b (`two_tier_grm`, GRM $K=4$, $J=12, N=2000$, `F1 = g01-g06`, `F2 = g06-g12` with `g06`
    cross-loading, $\rho=0.4$, `s1 = g01-g04`, `s2 = g07-g10`)**: `loglik` rel `1.1e-15`, item
    parameters `7.8e-9` (EM, 192 iterations; SQUAREM `1.6e-8`, 30 iterations), `cor_12` `6.3e-9`
    (SQUAREM `3.9e-9`), item Oakes SE rel `1.0e-7` (SQUAREM `1.1e-7`), `cor_12_se` rel `1.6e-6`, EAP
    $\theta$ `7.1e-15` / SD `3.4e-15` ($15^4$ nodes in mirt).
  - The unit tests (`tests/test_m7_two_tier.py`) check the reduced likelihood, expected counts and
    EAP against brute-force integration over the full $(P+S)$-dimensional product grid (to
    $10^{-12}$ or better, with missing data, a GRM item, a cross-loading item and items without a
    specific factor), a one-primary-factor two-tier model against the bifactor path, and a
    one-specific-factor model against the validated 3D path (`with_factors()` with the specific
    factor's correlations fixed at 0; log-likelihood to $10^{-12}$).
- **Summed-score distribution kernel (V36, M7)**: `summed_score_distribution()` (DESIGN §18.1) in
  `tests/test_r_m7_sumscore.py` (fast tier; generator `validation/r/m7_sumscore_reference.R` →
  `tests/reference/v36_sumscore.json`, poibin 1.6, mirt 1.48).
  - **Dichotomous items (Poisson-binomial)**: five probability vectors (`uniform_10`: $J=10$,
    $p\sim U(0.05, 0.95)$; `twopl_50`: 2PL at $\theta=0.4$; `extreme_200`: 2PL at $\theta=-3$;
    `rare_300`: $p\sim U(0.001, 0.02)$; `equal_40`: $p=0.3$). Largest absolute differences:
    `"recursion"` vs poibin's recursive formula (cdf) `2.9e-15` and vs its DFT-CF pmf `5.8e-16`;
    `"fft"` vs the DFT-CF pmf `6.1e-16`; `"normal"` vs `method = "NA"` `1.2e-15`;
    `"refined_normal"` vs `"RNA"` `1.3e-15`; `"poisson"` vs `dpois` `8.5e-16` (tolerance
    $10^{-13}$ absolute; `"poisson"` $10^{-11}$ relative). poibin's own exact values come from a
    cdf or from DFT-CF, so they check absolute accuracy only; relative accuracy in the tails is
    checked by enumeration and the binomial special case in the unit tests.
  - **Polytomous items**: mirt's summed-score table for a GRM ($K=4$, $J=10$, $N=1500$) and for
    five 2PL plus five GRM items. The reference stores mirt's category probabilities on the 61
    nodes (`probtrace`) and the weights, so only the kernel is under test:
    $N\sum_qw_qL(s\mid\theta_q)$ equals mirt's `expected` column to `3.1e-15` (relative), and the
    EAP and posterior SD of $\theta$ given the summed score built from the kernel equal mirt's to
    `5.3e-15` / `3.1e-15`.
  - **S-$X^2$ (V15)**: `_itemfit.lord_wingersky` now calls the kernel; the V15 agreement tests
    pass with unchanged references and tolerances.
  - **Unit tests** (`tests/test_m7_sumscore.py`): enumeration of all response patterns for mixed
    $K\in\{2,3,4\}$ (relative $10^{-13}$, both exact methods at $5\times10^{-16}$ absolute), the
    binomial special case, the log-space recursion at $P=10^{-4000}$ (2000 items), `jit` / `vmap`
    / `grad` for every method, and the gradient identity
    $\partial P(S=s)/\partial p_j=P_{-j}(s-1)-P_{-j}(s)$.
  - **Accuracy and speed of the two exact methods** (61 nodes, jitted, this validation machine's
    CPU, milliseconds per call): $J=50, K=2$: recursion 0.10, log-space recursion 2.7, FFT 0.51;
    $J=200, K=2$: 1.0 / 49 / 6.8; $J=1000, K=2$: 57 / 1200 / 174; $J=100, K=5$: 2.9 / 178 / 9.7.
    The FFT's largest absolute error against the recursion is $3\times10^{-16}$–$1.1\times10^{-15}$
    in every case, while the smallest probabilities are $10^{-31}$–$10^{-308}$: the FFT has no
    correct digits there. In this implementation the recursion is both the more accurate and the
    faster exact method (the characteristic function itself costs $O(JS_{\max})$), so it stays
    the default and the FFT is offered for comparison.
- **Information, expected score, reliability and summed-score EAP (V37, M7)**: DESIGN §18.2 in
  `tests/test_mirt_m7_information.py` (slow tier, about 30 s; generator
  `validation/r/m7_information_reference.R` → `tests/reference/v37_information.json`, mirt 1.48,
  61 points on $[-6, 6]$). mirt fits every model (`TOL = 1e-8`, converged); its estimates go into
  a flexpyirt `FitResult` with every parameter fixed (`table=`, `free = False`), so only the
  statistics are compared. Cases: `twopl` ($J=12$, $N=2000$), `twopl_missing` ($N=800$, 8% MCAR),
  `threepl` ($J=12$, $N=5000$), `grm` and `gpcm` ($K=4$, $J=8$, $N=2000$), `nominal` ($K=4$,
  $J=6$, $N=3000$), `mixed` (4 2PL, 3 3PL, 3 GRM $K=4$, 2 GPCM $K=3$, 2 nominal $K=3$; $N=5000$)
  and `twogroup` (6 2PL + 4 GRM $K=4$, all items invariant, $N=1500$ per group; focal
  $\hat\mu=0.480$, $\hat\sigma^2=1.590$). Largest differences over all cases:
  - **`iteminfo()` / `testinfo()`** on $\theta\in\{-6,-4,-3.5,\dots,4,6\}$: item information
    `1.4e-14` absolute (relative `4.5e-9`, at nominal items where the information is about
    $10^{-6}$; mirt evaluates $\sum_k[(P'_k)^2/P_k-P''_k]$, which cancels there); test
    information `8.3e-13` relative. Tolerance $10^{-9}$ relative plus $10^{-12}$ absolute.
  - **`expected.item()` / `expected.test()`**: `7.5e-15` absolute / `2.1e-14` relative (same
    tolerance).
  - **Marginal reliability**: against `marginal_rxx()` recomputed with
    `integrate(rel.tol = 1e-12)` `1.4e-15`; against `marginal_rxx()` itself (R's default
    tolerance, about $1.2\times10^{-4}$ relative) `4.5e-11` (tolerances $10^{-10}$ and $10^{-8}$
    absolute). For the two-group model the reference is `marginal_rxx()` of the standardized
    equivalent single-group model ($a^*=a\sigma$, $d^*=d+a\mu$, $N(0,1)$): `1.0e-15` (tight) and
    `1.4e-11` (default) over both groups; the direct `integrate()` of
    $\sigma^2I/(\sigma^2I+1)\,\phi(\theta;\mu,\sigma^2)$ with mirt's `testinfo()` agrees to
    `1.0e-15` as well. mirt's own formula applied to the focal group
    (`marginal_rxx(extract.group(mg, "focal"))`) gives 0.7288 where the invariant definition
    gives 0.8337; this is the documented difference of definition (DESIGN §18.2), not an
    accuracy difference, and the test asserts that the two differ.
  - **`empirical_rxx()`** of `fscores(method = "EAP", full.scores.SE = TRUE)`, including the case
    with missing responses and each group of the two-group model: `4.4e-16` (tolerance
    $10^{-9}$).
  - **`fscores(method = "EAPsum", full.scores = FALSE)`**: EAP `6.2e-15`, posterior SD `9.4e-16`
    (tolerance $10^{-9}$ absolute); `probability` $\times N$ against `expected` `7.8e-15`
    relative (tolerance $10^{-9}$).
  - **`fscores(method = "EAPsum", full.scores = TRUE)`** for every unique response pattern: EAP
    `6.2e-15`, SD `9.4e-16`. In `twopl_missing`, 453 of the 660 unique patterns are incomplete:
    mirt returns `NA` for exactly those and flexpyirt NaN (mirt refuses the
    `full.scores = FALSE` table for data with missing responses, so that case has no table
    reference).
  - **Not validated, raises `SpecificationError`**: reliability, the summed-score table and
    EAPsum for multidimensional models (their information and expected scores are V43);
    reliability, the summed-score table and EAPsum with `EmpiricalHistogram`, `DavidianCurve`
    or `NormalMixture`. The pooled `empirical_rxx()` of a `multipleGroup` fit (stored as
    `empirical_rxx_pooled`) has no flexpyirt counterpart; flexpyirt reports it per group.
  - **Unit tests** (`tests/test_m7_information.py`, fast tier): closed forms for 2PL
    ($a^2PQ$) and 3PL (Birnbaum) at $10^{-12}$, central finite differences of the category
    probabilities for 2PL / 3PL / 4PL / GRM / GPCM / nominal at $10^{-8}$, expected scores from
    the category probabilities, the summed-score table against enumeration of all 288 response
    patterns of a six-item mixed test at $10^{-12}$, the law of total expectation, invariance of
    the marginal reliability under a linear change of the latent scale ($10^{-13}$),
    reliability bounds and closeness of the marginal and empirical values on simulated data,
    and the error paths.
- **Multidimensional information matrix, directional information and expected score (V43,
  M7)**: DESIGN §18.3 in `tests/test_mirt_m7_information_multidim.py` (slow tier, about 40 s;
  generator `validation/r/m7_information_multidim_reference.R` →
  `tests/reference/v43_information_multidim.json`, mirt 1.46.1). Nothing is estimated: each case
  sets chosen item parameters with `pars = "values"` and builds the mirt object with
  `TOL = NaN` (`mirt()` for the confirmatory models, `bfactor()` / `bfactor(model2 = )` for the
  bifactor and two-tier models), and flexpyirt gets the same values with every parameter fixed.
  Cases: `cfa2_mixed` (10 items: 2PL, 3PL, 4PL with $u<1$, GRM $K=3,4$, GPCM $K=3,4$, nominal
  $K=3$; four items load on both factors), `cfa3` (9 items, 2PL, 3PL, GRM, GPCM; two
  cross-loadings), `bifactor` (12 items, 2PL + GRM $K=3$, general and three specific factors)
  and `two_tier` (8 items, 2PL + GRM $K=3$, two correlated primary factors with one
  cross-loading and two specific factors). Each has 15 points: 12 uniform on $[-3,3]^D$, all
  $-5$, alternating $\pm5$, and the origin; and 4–5 directions given as angles with
  $\sum_m\cos^2\alpha_m=1$, including the axes, the diagonal and directions with a negative
  cosine. The generator also asserts that mirt's `testinfo(degrees = )` equals
  $\mathbf u^\top(\sum_j\mathbf I_j)\mathbf u$ of its own `multidim_matrix = TRUE` matrices
  ($10^{-10}$). Largest differences over all cases:
  - **Item information matrices** (`iteminfo(multidim_matrix = TRUE)`): `1.1e-15` absolute,
    `1.0e-12` relative (at entries of size about $10^{-4}$ at the extreme points); the test
    matrix (their sum) `1.8e-15` absolute.
  - **Directional item and test information** (`iteminfo` / `testinfo(degrees = )`, with
    `direction = cos(degrees)`): item `1.0e-12` relative, test `5.8e-14` relative.
  - **Composite SE** $\sqrt{\mathbf u^\top\mathbf I^{-1}\mathbf u}$ against the inverse of the
    sum of mirt's matrices (`solve()` in R; mirt's `plot(type = "SE")` shows $1/\sqrt{I_{\mathbf u}}$ instead, DESIGN §18.3): `2.8e-14` relative.
  - **`expected.item()` / `expected.test()`**: `4.8e-14` absolute / `4.7e-15` relative.
  Tolerance for all: $10^{-10}$ relative plus $10^{-13}$ absolute (for entries that are 0 because
  a slope is fixed at 0).
  - **Unit tests** (`tests/test_m7_information_multidim.py`; the simple-structure and
    multiple-group tests are in the slow tier): a two-factor simple-structure model equals the two
    unidimensional sub-tests along the axes (item and test information, composite SE, expected
    scores; $10^{-12}$) and its matrices are block diagonal; matrices are exactly symmetric, PSD and
    of rank 1 for the built-in families; the 2PL closed form $P(1-P)\mathbf a\mathbf a^\top$
    ($10^{-12}$); directional information $=\mathbf u^\top\mathbf I\mathbf u$, invariance under
    scaling and sign of the direction, the composite SE against a direct inverse and the bound
    $\mathrm{SE}_{\mathbf u}\ge I_{\mathbf u}^{-1/2}$; infinite SE for a numerically singular matrix;
    in a bifactor model each item's matrix is 0 outside the general and own specific factors; a
    multiple-group model uses each group's parameters; unidimensional `information_matrix` is
    the scalar information; the `theta` layouts (array, DataFrame by factor name, one point) and
    the error paths.
- **The numbers behind the plots (V48, #103)**: DESIGN §21 in `tests/test_mirt_v48_plots.py`
  (slow tier, about 20 s; generator `validation/r/v48_plots_reference.R` →
  `tests/reference/v48_plots.json`, mirt 1.48, TAM 4.3-25, 61 points on $[-6, 6]$). mirt fits
  every model (`TOL = 1e-8`, converged) and its estimates go into a `FitResult` with every
  parameter fixed, as in V37. The data are stored row by row (unique patterns plus each row's
  pattern) because the bins depend on the order of the respondents. Cases: `twopl` ($J=10$,
  $N=1003$, so the remainder 3 puts one extra respondent in the first bin and two in the last),
  `twopl_missing` ($N=1200$, 10% MCAR, bins per item over the respondents who answered it),
  `grm` ($K=4$, $J=6$, $N=1500$), `mixed` (3 2PL, 3 3PL, 2 GPCM $K=4$, 2 nominal $K=3$;
  $N=3000$), `twogroup` (5 2PL + 3 GRM $K=4$, all items invariant, $N=805$ per group, focal mean
  and variance free) and `gpcm_tam` (TAM's own GPCM fit). Largest differences over all cases:
  - **EAP scores** (the input of the bins) against `fscores(method = "EAP")`: `7.1e-15`
    (tolerance $10^{-9}$).
  - **`empirical_item_curve()`** against `itemfit(fit_stats = "X2", return.tables = TRUE,
    group.bins = 10, mincell.X2 = 0, Theta = eap)`, item by item (mirt returns one item's tables
    per call; `mincell.X2 = 0` turns its cell collapsing off; a `multipleGroup` fit is refused,
    so each group's `extract.group()` is called with that group's rows of `Theta`, which is what
    mirt's own per-group `itemfit()` does): the bin sizes equal mirt's row sums exactly;
    observed proportions are identical (`0`); expected probabilities `4.6e-15` (tolerance
    $10^{-9}$); bin means against mirt's labels, which are rounded to 4 decimals, `5.0e-5`
    (the rounding; tolerance $5\times10^{-5}$). The expected tables are $N\,P(\bar\theta_b)$, so
    they also check the bin means to full precision.
  - **`item_probabilities()`** against `probtrace()` on $\theta\in\{-6,-4,-3.5,\dots,4,6\}$:
    `2.9e-15` (tolerance $10^{-12}$).
  - **`wright_map_table()` item locations** against `uniroot(tol = 1e-13)` of
    $\sum_{c\ge k}$`probtrace()`$-1/2$ on $[-20, 20]$: `2.1e-14`; against
    `coef(IRTpars = TRUE)` $b$ (2PL) and $b_k$ (GRM): `1.1e-14` (tolerance $10^{-9}$). For the
    3PL items the threshold is below $b$, as it must be. Against TAM's `tam.threshold()` at TAM's
    own GPCM estimates (slopes and intercepts read from `B` and `AXsi`): `8.4e-5`, TAM's bisection
    stops when the interval is below $10^{-4}$ (tolerance $10^{-4}$).
  - **Not validated**: mirt's $X^2$ / $G^2$ statistics and cell collapsing, and the
    Clopper–Pearson interval of `empirical.plot` (not computed); the figures themselves.
  - **Unit and smoke tests** (`tests/test_plotting.py`, fast tier): mirt's bin-size rule, a
    brute-force recomputation of the bins with missing responses, closed-form thresholds (2PL
    $-d/a$, GRM $-d_k/a$, a 3PL item with and without a crossing), two groups with DIF, the
    error paths; every plot function on the Agg back end with the plotted data equal to the
    DataFrames; `MissingDependencyError` when matplotlib cannot be imported. CI also runs
    `scripts/check_plotting_without_matplotlib.py` in an environment without matplotlib.
- **DIF screening, Wald DIF tests and effect sizes (V49, #104)**: DESIGN §16.22 in
  `tests/test_mirt_m5_dif_screen.py` (slow tier, about 7 min; generator
  `validation/r/m5_dif_screen_reference.R` → `tests/reference/v49_dif_screen.json`, mirt 1.48;
  `optimizer = "NR"`, `accelerate = "none"`, `TOL = 1e-10`, 81 points on $[-8,8]$ for every fit and
  refit; set `V49_CACHE=<file.rds>` to keep the long `DIF()` runs between runs of the generator).
  Data: two groups of 1200 (8 2PL items and 4 graded items $K=3$; DIF in q02's slope, q04's and
  q07's intercepts and q09's intercepts; focal $N(0.4, 1.3)$) and three groups of 800 (8 2PL items,
  intercept DIF in q03 for g3).
  - **`"all_others"`** against `DIF(<fully invariant>, scheme = "drop", p.adjust = "BH")` for all
    parameters and for intercepts only (two and three groups): $X^2$ to $1.5\times10^{-11}$, `df`
    exact, flagged sets equal.
  - **Sequential strategies** against `DIF(scheme = "drop_sequential" / "add_sequential",
    seq_stat = 0.05, p.adjust = "BH")` with `max_run` 2, 3 and `Inf` (our `max_iter` = `max_run` − 1):
    flagged sets equal after every number of rounds, the flagged items' final tests (mirt's output)
    $X^2$ to $1.5\times10^{-11}$, and the final model's log-likelihood
    (`return_seq_model = TRUE`) to $1.5\times10^{-15}$ relative. In these data the flags are the
    same from round 1 on (both strategies converge in two rounds); the iteration itself is unit-tested
    (`tests/test_m5_dif_screen.py`). mirt's sequential loop indexes item names by position within
    `items2test`, so the generator tests a prefix `1:k` (the anchors are the last items).
  - **Wald** against `DIF(Wald = TRUE)` on the anchored model with Oakes SEs (`SE.type = "Oakes"`),
    both fitted independently: $W$ to $2.7\times10^{-7}$ relative, `p` / `adj_p` (Holm) to
    $1.2\times10^{-7}$ relative.
  - **Effect sizes** at mirt's estimates of the partially invariant model (q02, q04, q07, q09 free)
    and the invariant model, put into fixed `FitResult`s on 201 points on $[-6,6]$ (DRF's
    `quadpts`, `theta_lim` and `RMSD_DIF()`'s grid): signed / unsigned differences (focal and
    reference densities) and DTF against `mirt:::calc_DRFs(best_fitting = TRUE)` to
    $1.1\times10^{-15}$; EAP scores to $6.7\times10^{-15}$; ESSD and ETSSD against `empirical_ES()`
    (with `Theta.focal` the EAPs on the same grid) to $2.2\times10^{-15}$; RMSD of both groups and
    both models against `RMSD_DIF(probfun = FALSE)` to $9.7\times10^{-16}$. Tolerances: $X^2$
    $10^{-8}$, $p$ $10^{-3}$ relative (mirt's `p` is `1 - pchisq()`, imprecise for tiny $p$),
    log-likelihood $10^{-8}$ relative, $W$ $10^{-5}$ relative, effect sizes $10^{-12}$.
  - **mirt differences**: `DRF(den.type = "focal" / "reference")` stops in mirt 1.48
    (`add_std.DRF()` refers to an undefined `VAR`), so the generator calls `calc_DRFs()`, which
    computes `DRF()`'s point estimates. mirt's `DRF()` default density (`best_fitting = FALSE`) is
    the posterior counts of the data, not reproduced (DESIGN §16.22). For the sequential schemes
    mirt reports only the flagged items; `DIFScreenResult.table` has every tested item.
- **Robust standard errors (V35, M7)**: `fit(se="crossprod" | "sandwich")` (DESIGN §17.5) against
  mirt 1.48 `SE.type = "crossprod"` / `"sandwich"`. The generator stores mirt's parameter table
  (`mod2values`) and the full `vcov` matrices, each row labelled by the first `parnum` of its
  parameter, so the tests rebuild mirt's solution and compare whole matrices.
  **Cases** (`validation/r/m7_robust_se_reference.R`, `tests/reference/v35_*.json`): a 2PL fitted
  to 3PL data with a skewed trait and 5% MCAR ($J=12$, $N=2000$; the model is wrong on purpose,
  and the sandwich differs from the Oakes covariance by up to 0.49 in the scaled measure below);
  4 3PL + 3 GRM(4) + 3 GPCM(3) + 2 nominal(3) items ($N=5000$); a 2PL MAP fit with
  $a\sim N(1.2, 0.5)$, $d\sim N(0, 2)$ and $N=300$; two groups with six anchors and a free focal
  mean and variance (81 points on $[-8,8]$); FIPC with five fixed items, a free latent mean and
  variance and equal slopes on four items; two correlated factors with a cross-loading ($31^2$
  nodes); three correlated factors ($15^3$ nodes); EH under FIPC.
  **At mirt's solution** (flexpyirt's problem compiled with mirt's estimates; the measure is
  $\max_{ij}\lvert v_{ij}-v^{\text{mirt}}_{ij}\rvert/\sqrt{v^{\text{mirt}}_{ii}v^{\text{mirt}}_{jj}}$
  over the whole matrix): the log-likelihood agrees to $\le3.6\times10^{-15}$ (relative), which
  confirms the parameter mapping; the cross-product covariance to $\le1.3\times10^{-11}$
  (2PL $8.9\times10^{-15}$, mixed $4.9\times10^{-13}$, MAP $2.6\times10^{-15}$, two groups
  $1.3\times10^{-11}$, FIPC $9.9\times10^{-15}$, two factors $6.2\times10^{-15}$, three factors
  $2.0\times10^{-14}$); the sandwich to $\le2.7\times10^{-8}$ in the six cases with NR M-steps in mirt
  ($2.3\times10^{-10}$, $1.4\times10^{-9}$, $1.1\times10^{-10}$, $2.7\times10^{-8}$,
  $1.1\times10^{-8}$, $1.6\times10^{-9}$) and to $1.4\times10^{-5}$ for three factors, where
  the remainder is mirt's Oakes Hessian (finite differences). Tolerances $10^{-9}$ and $10^{-6}$
  ($10^{-4}$ for the three-factor sandwich; see below).
  **End to end** (`fit()` from flexpyirt's own starting values, `BAEM(tol=1e-9)`): free parameters
  within $4.1\times10^{-8}$ of mirt's and standard errors within $4.6\times10^{-8}$ (relative)
  for both types in the six cases with NR M-steps in mirt; three factors: $5.0\times10^{-7}$ and
  $6.6\times10^{-6}$. Tolerances $10^{-6}$ and $10^{-5}$. mirt fits the three-factor case with
  BFGS (as in V14; NR is very slow on the $15^3$ grid) and stops with a score of
  $5.6\times10^{-6}$ at its estimates, against $\le2.9\times10^{-7}$ with NR, so that case is
  checked with tolerances 100 times wider ($10^{-4}$ for the sandwich at mirt's solution and
  for the parameters, $10^{-3}$ for the standard errors).
  **Lower asymptotes.** mirt's `vcov` is for $\operatorname{logit}g$; the tests transform mirt's
  matrix with the Jacobian $g(1-g)$.
  **Priors.** mirt adds $-\nabla^2\log\pi$ to the cross-product (`SE.simple`:
  `Igrad + Iprior`), as flexpyirt does. Only normal priors are compared: mirt's Hessians of
  `lnorm` and `expbeta` priors are wrong (V7), and those priors are covered by the unit tests
  (the difference between the matrices with and without priors is the autodiff Hessian of the
  log prior).
  **Empirical histogram.** mirt's cross-product holds the histogram fixed, like flexpyirt's, and
  is compared directly at mirt's item estimates and histogram: $6.2\times10^{-15}$. mirt's `"sandwich"` uses
  its Oakes Hessian, which for EH is not the Hessian with the histogram held fixed (V6): its
  matrix differs from flexpyirt's by 0.049 (standard errors by up to 2.4%). The sandwich is therefore compared with
  $V_R V_C^{-1} V_R$ assembled from mirt's Richardson covariance $V_R$ (the fixed-histogram
  Hessian, the definition of DESIGN §15.35) and mirt's cross-product covariance $V_C$: $1.4\times10^{-8}$
  (Richardson is a finite-difference Hessian; flexpyirt's Oakes covariance against $V_R$:
  $7.1\times10^{-9}$). Tolerance $10^{-6}$. The EH reference uses `TOL = 1e-7` in mirt (about
  11,000 cycles; mirt did not reach $10^{-9}$ in 100,000), which does not matter for a comparison
  at mirt's own estimates and histogram; there is no end-to-end comparison for EH, for the
  reasons given in V6.
  **`"sandwich.Louis"`.** The same estimator with the bread computed by Louis' formula. Where no
  latent parameter is free (2PL, mixed, MAP) mirt's matrix equals its `"sandwich"` one (flexpyirt
  agrees with both to $\le8.8\times10^{-10}$). Where a latent mean, variance or correlation is
  free (two groups, FIPC, two factors) mirt's Louis bread is wrong (`SE.type = "Louis"` itself
  returns no matrix there) and `"sandwich.Louis"` is off by 1.0 to 7.0 in the scaled measure. It
  is not a distinct estimator and is not offered.
  **Not covered.** `DavidianCurve` and `NormalMixture`: mirt computes neither type for them, so
  flexpyirt raises `SpecificationError`. Bifactor and two-tier models raise `SpecificationError` as well (mirt supports them; the
  score kernel does not use the dimension reduction yet). Multiple-group multidimensional models, 4PL, PCM and
  Rasch items go through the same code but have no V35 case of their own. Sampling-weighted and
  cluster-robust versions are not implemented; `weights=` with either option raises
  `SpecificationError`.
  Tests: `tests/test_mirt_m7_robust_se.py`. Unit tests (`tests/test_m7_robust_se.py`): the
  chunked scores against a dense forward-mode Jacobian for any chunk size (missing data, mixed
  formats with priors, two groups with shared parameters, two factors, EH), $\sum_p c_ps_p$
  against the gradient, $B=H$ within sampling error for a correct model with $N=20{,}000$
  (standard errors within 3%), the sandwich equal to the Oakes covariance when $B=H$ is forced,
  doubled rows halving the covariance, `b_se` and `link()` reading the chosen covariance, JSON
  round trip, and the rejected cases.

- **Sampling weights (V34, M7)**: mirt's `survey.weights` (mirt 1.48, `optimizer = "NR"`,
  `TOL = 1e-10`, `SE.type = "Oakes"`); the reference files store the raw rows with one weight per
  row, so the weighted pattern compression is part of what is compared. V34a: simulated 2PL
  ($J=10$, $N=1000$), log-normal weights related to the sum score, not normalized
  ($\sum w=1541.7$), five rows with weight 0. V34b: Science GRM ($N=392$), uniform weights
  normalized to sum to $N$. V34c: two groups of 800, anchors q01–q05, focal mean and variance
  free, 5% MCAR missing, gamma weights with different totals in the two groups (81 points on
  $[-8,8]$). V34d: FIPC (q01–q05 fixed, latent mean and variance free, 81 points), log-normal
  weights normalized to $N$. V34e: bifactor 2PL (`bfactor()`, $J=9$, three specific factors of
  three items, $N=1000$, 21 points per dimension on $[-6,6]$), log-normal weights not normalized
  ($\sum w=1467.6$): log-likelihood $4.7\times10^{-16}$ (relative), parameters
  $9.5\times10^{-7}$, Oakes and Hessian SEs $2.2\times10^{-6}$ (relative), AIC and BIC
  $3.8\times10^{-11}$. Measured differences of V34a–d (plain EM / SQUAREM, at most): log-likelihood
  $1.8\times10^{-15}$ (relative), parameters $7.6\times10^{-8}$, Oakes SEs $1.1\times10^{-7}$
  (relative), flexpyirt Hessian SEs vs mirt Oakes $1.1\times10^{-7}$ (relative), AIC and BIC
  $5.1\times10^{-11}$; all within the common tolerances. mirt's BIC uses the sum of the weights as
  $N$ (so does flexpyirt, DESIGN §17.4); V34a, whose weights do not sum to the number of rows,
  checks this. Every reference fit differs from the unweighted fit of the same data
  (log-likelihoods differ by more than $10^{-3}$ and parameters by more than $10^{-3}$), so
  ignoring the weights would fail the tests. **Not validated against R**: integer weights against
  row replication (2PL with missing data; two-group GRM with plain EM and SQUAREM; FIPC with an
  empirical histogram; MAP; a bifactor and a two-tier model), zero-weight rows against dropped rows, and rescaled weights are
  checked as identities in `tests/test_m7_weights.py`. The Oakes / Hessian SEs agree with mirt's
  but are not valid variance estimates under sampling weights (DESIGN §17.4); item fit, M2,
  local dependence, JSI, DIF and model-based parallel analysis raise an error for a weighted fit.
  Tests: `tests/test_mirt_m7_weights.py`. Reference generator:
  `validation/r/m7_weights_reference.R`.
- **Model comparison and simulation of responses (V54, #109; DESIGN §24)**:
  - **`compare_fits()` against `anova()`**: every model is fitted by mirt (`optimizer = "NR"`,
    `accelerate = "none"`, `TOL = 1e-10`) and by flexpyirt (`BAEM(tol=1e-9)`) on the same grid,
    and the whole table is compared: Rasch / 2PL / 3PL on the V11a data (61 points on
    $[-6, 6]$), PCM / GPCM on the V8b data, and the fully invariant two-group 2PL (focal mean and
    variance free) against the model with anchors q01–q05 only and against the model with only
    the last item free across groups (V4 data, 81 points on $[-8, 8]$; $p=0.188$, so the
    $p$-value is checked away from 0). `n_params` equals mirt's `nest` exactly; log-likelihoods
    and the four criteria agree to $5\times10^{-11}$, $X^2$ to $1.4\times10^{-11}$ and $p$ to
    $7\times10^{-13}$ (tolerances $10^{-7}$, $10^{-7}$, $10^{-9}$, absolute). mirt prints
    $p=0$ where $p<10^{-16}$ (`1 - pchisq()`); flexpyirt reports the upper tail.
  - **`simulate_responses()`, deterministic part**: random streams cannot match R's
    (`simdata()` uses R's generator), so the reference fixes the item parameters, $\theta$ and
    the uniform numbers. For 9 items (Rasch, 2PL ×2, 3PL, 4PL, graded $K=3,4$, GPCM $K=4$,
    nominal $K=4$ with free scoring coefficients) at 400 values of $\theta$ (including
    $\pm6$, $\pm3$, 0) and for a two-factor model (2PL, graded $K=3$, 3PL; one cross-loading
    item) at 300 points, flexpyirt's category probabilities equal `probtrace()` to $10^{-12}$
    and the responses obtained in R by $x=\#\{k<K-1: u>F_k\}$ are reproduced exactly.
  - **`simulate_responses()`, distributional part** (Monte Carlo, $|z|<4.5$): category
    proportions and the summed-score distribution of 60 000 simulated respondents (mixed
    Rasch, 2PL, 3PL, 4PL, GRM, GPCM and nominal items, $N(0.3, 1.4)$) and of two groups
    (30 000 and 40 000) agree with the model-implied marginals (81-point Gauss–Hermite
    quadrature of the same kernels and of the Lord–Wingersky recursion, V36); the latent draws
    have the specified covariance (two factors; bifactor with two specific factors, where items
    sharing a specific factor are more correlated); `NormalMixture` and fitted `DavidianCurve`
    draws pass a Kolmogorov–Smirnov test (0.1% level) against the exact CDF; fitted
    `EmpiricalHistogram` draws are grid nodes with frequencies matching the fitted weights.
  - **Recovery**: `fit()` on 20 000 simulated respondents (the mixed-format test) returns every
    item parameter within 4.5 SEs of the generating value, and a two-group fit on 8 000 per
    group recovers the focal mean and variance (0.5, 1.3) within 4.5 SEs. A `FitResult` and a
    `Model` with the fit's estimates produce identical data from the same key, and data are
    reproducible from `key`.
  Tests: `tests/test_mirt_v54_compare_fits.py`, `tests/test_v54_simulate.py`. Reference generator:
  `validation/r/v54_compare_simulate_reference.R`.

- **IRT equating and raw-to-scale tables (V51, #106)**: DESIGN §23 in `tests/test_r_equating.py`
  (generator `validation/r/v51_irt_equating_reference.R` → `tests/reference/v51_irt_equating.json`;
  mirt 1.48, equateIRT 2.5.2, SNSequate 1.3.5, equate 2.0.9). No parameters are estimated: the
  item parameters are chosen in R and stored, the latent distributions are $N(0,1)$ on 61
  equispaced points on $[-6,6]$, and the link is $\theta_Y=A\theta_X+B$ ($A=1.15$, $B=-0.25$
  dichotomous; $A=0.9$, $B=0.3$ polytomous). The fast tier builds the forms from the stored
  parameters and checks the kernels and the conversion rules; the slow tier repeats every
  comparison end to end through `irt_equate()`, `score_distribution()` and `scale_table()` on
  fits fixed at those values. Largest differences:
  - **Dichotomous true score** (20 + 20 items; 12 and 9 3PL items, $\sum c=2.317$ and $1.477$):
    against equateIRT's own Newton solution for the scores inside the range of the curve
    (`score_compute`, whose $|\Delta\theta|<10^{-6}$ stop leaves about $10^{-12}$): equated
    scores `2.4e-12`, $\theta$ `1.2e-12` (tolerance $10^{-9}$). Against SNSequate for every
    score: the Kolen & Brennan interpolation below $\sum c$ and the maximum score `2.4e-15`
    (tolerance $10^{-11}$); inside, `1.2e-4`, the default `uniroot()` tolerance SNSequate uses
    (tolerance $10^{-3}$, a cross-check only).
  - **Dichotomous observed score**: equateIRT's OSE with the same nodes and weights for both
    populations, $w_1=1$, $0$, $0.5$: `5.7e-14` (tolerance $10^{-11}$).
  - **Polytomous and mixed true score** (new: 4 GRM $K=4$ + 6 2PL, maximum 18; old: 3 GPCM
    $K=3$ + 2 GRM $K=4$ + 5 3PL + 3 2PL, maximum 20, $\sum c=0.795$): no R package equates
    polytomous forms by IRT; the reference solves `expected.test()` of the new form with
    `uniroot(tol = 1e-13)` and evaluates `expected.test()` of the old form at $A\theta+B$:
    $\theta$ `5.5e-13`, equated scores `9.3e-13` (tolerance $10^{-9}$).
  - **Model-implied distributions**: the R Lord–Wingersky recursion on `probtrace()` equals
    mirt's `fscores(method = "EAPsum", full.scores = FALSE)` `expected` / $N$ to $10^{-12}$ in R;
    flexpyirt's distributions equal both to `5.6e-15` relative, in every population (tolerance
    $10^{-11}$ relative).
  - **Polytomous observed score**: `equate::equate(type = "equipercentile")` on the two
    synthetic distributions: `4.5e-9`. equate rounds percentile ranks and the cumulative
    distribution to 10 decimals (`equate:::equip`), about $10^{-10}/g(y)$ in the score
    (tolerance $10^{-8}$). On the same distributions flexpyequate's equipercentile equating
    agrees to `2.5e-14`, and the unit tests compare the two on random distributions with zero
    probabilities to $10^{-12}$.
  - **Raw-to-scale table** of the old polytomous form (slope 10, intercept 50, bounds 25 and
    75, rounded to integers): true score ($\theta$ by `uniroot()` on `expected.test()`, SE
    $1/\sqrt{\texttt{testinfo}}$): $\theta$ `2.3e-12`, SE `1.2e-12` relative; EAPsum (mirt
    `fscores`): $\theta$ `5.6e-15`, SE `1.6e-15` relative; the rounded scale scores are equal.
  - **Not covered by R**: the 4PL upper-asymptote interpolation, multiple-group forms and
    populations, and links are unit-tested (`tests/test_equating.py`) against `expected_score()`,
    `item_information()`, `summed_score_table()` and the formulas of DESIGN §23. EH, Davidian and
    mixture populations come from the same fitted grid as `link()` (`_fitted_grid`) and have no
    equating test of their own.
- **Person fit and residual item fit (V53, #108)**: DESIGN §25. The cases reuse V37's data and
  mirt estimates (every parameter fixed, 61 points on $[-6, 6]$); per case and estimator (EAP,
  MAP, ML, WLE) the R script computes mirt's `fscores()`, `personfit(m, Theta = )` on the full
  data (stored for the first 300 unique patterns) and `itemfit(m, fit_stats = "infit", Theta = )`
  (EAP, MAP and WLE; ML's infinite estimates are replaced by mirt). **At mirt's $\theta$** (the
  kernel evaluated at the stored estimates): $l_z$, infit, outfit and their $z$ values agree to
  $8.3\times10^{-9}$ (relative) over all eight cases and four estimators, and $l_z^*$ agrees with
  the R recomputation of DESIGN §25 item 3 (`probtrace()` with central differences, `iteminfo()`)
  to $2.8\times10^{-8}$ for $|\hat\theta|\le10$; beyond that (mirt's ML on 3PL flat tails and on
  its missing-data extremes, V13) the central differences lose precision, and flexpyirt agrees
  with an analytic NumPy evaluation instead. **End to end** (flexpyirt's own estimates, which
  differ from mirt's by up to $5.6\times10^{-15}$ (EAP), $2.9\times10^{-6}$ (MAP),
  $1.2\times10^{-5}$ (ML) and $2.6\times10^{-6}$ (WLE)): person statistics within
  $2.0\times10^{-5}$, item statistics within $3.1\times10^{-5}$. **Rasch**: $l_z$, infit and outfit
  agree with `sirt::personfit.stat()` to $3.7\times10^{-13}$ (relative) at mirt's WLE estimates
  (mirt and sirt agree to $1.8\times10^{-15}$). **Known differences from mirt** (DESIGN §25
  item 5): person `z_outfit` uses $n^2$ (answered items) where mirt's `personfit()` uses $J^2$,
  and mirt's `itemfit()` recycles the per-item respondent count over rows in the item
  `z.outfit`; both matter only with missing responses, so the references recompute these $z$
  values from mirt's own residual moments (`attr(m, "inoutfitreturn")`), and on complete data
  the tests also compare with mirt's values directly. **$l_z^*$ has no R package reference**
  (`PerFit` could not be installed here, and its `lzstar()` covers dichotomous items only); in
  addition to the R recomputation, `tests/test_person_fit.py` checks it against Snijders' (2001)
  closed form for 2PL items (ML, MAP, WLE; $10^{-10}$) and simulates its null distribution (2PL,
  $J=20$, $N=3000$, true parameters): $l_z^*$ has mean within sampling error of 0 and SD in
  $[0.93, 1.07]$ for ML and WLE, while $l_z$ has SD below 0.93. Whether this suffices to list
  $l_z^*$ as validated is the owner's decision (#108). Tests:
  `tests/test_mirt_v53_person_fit.py`, `tests/test_person_fit.py`. Reference generator:
  `validation/r/person_fit_reference.R`.

- **Many groups (V55, #79; DESIGN §26)**: `tests/test_mirt_v55_many_groups.py` (reference
  `validation/r/v55_many_groups_reference.R`, mirt `optimizer = "NR"`, `accelerate = "none"`,
  `TOL = 1e-10`, 81 points on $[-8, 8]$; mirt needed 2596 EM cycles and 20 minutes). With ten
  groups flexpyirt takes the group-blocked E-step (checked). At `BAEM(tol=1e-9, squarem=True)`
  (329 iterations): log-likelihood $1.5\times10^{-11}$ apart, all 318 item and latent
  estimates within $5.6\times10^{-8}$, Oakes SEs within $5.7\times10^{-8}$ relative, EAP and its
  SD within $4.5\times10^{-8}$ (tolerances $10^{-6}$ relative, $10^{-4}$, $10^{-3}$ relative,
  $10^{-5}$). The blocked path against the flat path (every pattern under every group;
  `tests/test_many_groups.py`, two and three groups, a GRM item and 5% missing): expected counts,
  log-likelihood, EAP at a fixed parameter vector within $10^{-10}$, Oakes and Hessian
  information within $10^{-9}$ (relative to the largest entry), Oakes sub-chunking of a block
  identical; fitted estimates within $10^{-6}$ and SEs within $10^{-5}$ relative (the two paths
  sum in a different order, so the EM iterates differ at rounding level).

## Validation beyond agreement
- **Normal-mixture latent distributions (V22, M5)**: flexpyirt's `NormalMixture` (DESIGN §16.7)
  uses mirt's class parameterization for the shape and standardizes the mixture analytically, so
  on the grid $\theta_q$ it is exactly mirt's `multipleGroup(dat, 1, itemtype = ..., dentype =
  "mixture-k", invariance = c(<all items>, "free_means", "free_var"))` (one set of items, class 1
  fixed at $N(0,1)$, `MEAN_1`, `COV_11`, `PI` free for classes $2..k$) on the image grid
  $z_q=M+\sqrt V\theta_q$, with $a=a^{\text{mirt}}\sqrt V$ and $d_k=d_k^{\text{mirt}}+s_k a^{\text{mirt}}M$
  ($s_k=1$ for 2PL/GRM, $k$ for GPCM). `validation/r/m5_mixture_reference.R`
  (→ `tests/reference/v22_mixture_*.json`) therefore records, per case:
  1. `exact`: flexpyirt's discretized likelihood re-coded independently in R (2PL, GRM and GPCM
     category probabilities, per-component normalized weights, standardization), maximized by
     Newton steps (central-difference gradient, Richardson Hessian) from mirt's solution on
     $[-6,6]$ until max|grad| < 2e-5 (final 1.8e-6–3.6e-6), with Richardson-Hessian SEs and EAPs.
  2. `mirt`: built-in mirt refitted on the image grid at the exact maximum (asymmetric
     `theta_lim`, e.g. `c(-7.76, 9.50)` for V22a; `optimizer = "NR"`, `TOL = 1e-10`), its
     estimates and `fscores()` EAPs mapped to flexpyirt's scale. R's own mirt-scale likelihood
     reproduces mirt's `logLik` to 1e-12 (mirt's estimates are stationary for it, max|grad| <=
     3.6e-6) and mirt's `fscores()` EAP to 1e-8.
  Findings on mirt 1.46.1:
  - `dentype = "mixture-k"` estimates class-specific item parameters by default (a mixture IRT
    model); the common-items model needs `invariance = <all items>` (which emits a harmless
    recycling warning from `ESTIMATION`). `multipleGroup`'s third positional argument is
    `group`, so `itemtype` must be named.
  - **mirt's mixture Oakes SEs are not the observed-information SEs.** On mirt's own scale and
    grid, the R Richardson Hessian of mirt's discretized likelihood gives latent SEs (`MEAN_1`,
    `COV_11`, `PI`) of (0.159, 0.139, 0.135) on V22a, while `SE.type = "Oakes"` gives
    (0.0315, 0.0048, 0.0401), ratios 0.014–0.34 over the three cases; `SE.type = "complete"`
    gives (0.0262, 0.0042, 0.0380), and 1/sqrt(N pi_1 pi_2) = 0.0380 is the complete-data SE of
    `PI` with known classes. Item SEs are also too small (V22a q01 `a1`: 0.0539 vs 0.0898). Other
    observed-information SE types are refused for mixtures. The SE reference is therefore the R
    Richardson Hessian; mirt's Oakes, complete and R-Hessian SEs on mirt's scale are stored as
    evidence (`mirt.se_*_mirt_scale`), and a test checks that flexpyirt's latent SEs equal the R
    Hessian ones on mirt's scale (the raw shape coordinates are shared) while mirt's Oakes SEs are
    below 0.6 times them.
  - `fscores(method = "MAP")` is refused for mixtures (flexpyirt rejects MAP too); EAP columns are
    named `Class_1`, `SE_F1`.
  - mirt's class-augmented EM is slow: 3,092–4,555 (61 points, V22a/b) and 24,961 (121 points, V22c) EM cycles at `TOL = 1e-10` from the generating
    values. In mirt's parameterization flexpyirt's (node-level) EM also stalled at inferior points
    from 2 of 4 starting shapes on V22a and 3 of 4 on V22b (0.73–53.2 log-likelihood points
    lower); standardized, every start reached the maximum in 168–661 plain EM iterations (27–93
    with SQUAREM; DESIGN §16.7).
  - **Grid resolution (V22c)**: with narrow components (standardized $\sigma_c\approx0.26$) and
    20 three-category items, posteriors are about as narrow as the 0.2 spacing of 61 points on
    $[-6,6]$: the 61-point maximum is 5.7e-3 log-likelihood points from the 121- and 241-point
    ones (which agree to 1e-9), and mirt's fixed grid and flexpyirt's moving standardized grid
    gave maxima 2.6e-4 points and up to 6e-3 in the parameters apart. V22c therefore uses 121
    points on $[-6,6]$ (mirt with `accelerate = "squarem"`, the same fixed point; it still needed
    24,961 EM cycles).
  Results (maxima over plain EM and SQUAREM; flexpyirt fitted from its default starting shape,
  `BAEM(tol=1e-9)`; standardized components compared sorted by mean, since component labels are
  arbitrary; SE checks fit from a start with mirt's labelling, a gauge choice with the same
  likelihood):
  - **V22a (`v22_mixture_2pl_k2`, 2PL, $J=20$, $N=3000$, $0.65N(0,1)+0.35N(2.5,0.5^2)$, 61 pts)**:
    `loglik` rel `4.5e-16` (exact and mirt), item params `3.5e-8` / `4.3e-8`, standardized
    components `5.4e-8` / `5.5e-8`, raw shape vs mirt's `MEAN_1, COV_11, PI` `1.2e-8`; item SEs vs
    R Hessian `1.0e-5` rel (latent `3.9e-6`), Oakes vs Hessian `3.6e-14`; EAP at the exact
    parameters `4.7e-15`, vs mirt `4.5e-9` (SE `3.1e-9`). Plain EM 734 iterations, SQUAREM 54.
  - **V22b (`v22_mixture_grm_k2`, GRM $K=4$, $J=12$, $N=2500$, $0.8N(0,1)+0.2N(-2.2,0.4^2)$, 61
    pts)**: `loglik` rel `8.8e-16`, item params `3.3e-8` / `1.7e-9`, components `9.7e-9` / `7.8e-9`,
    raw shape vs mirt `9.3e-9`; item SEs vs R Hessian `1.1e-5` rel (latent `3.2e-5`), Oakes vs
    Hessian `5.0e-15`; EAP `4.9e-15`, vs mirt `2.6e-9`. Plain EM 227, SQUAREM 34. (flexpyirt's
    default start converges to the relabelled solution whose raw component 1 is the small
    component: same likelihood and standardized components.)
  - **V22c (`v22_mixture_gpcm_k3`, GPCM $K=3$, $J=20$, $N=4000$, $0.5N(0,1)+0.3N(2.3,0.45^2)+
    0.2N(-2.3,0.5^2)$, 121 pts)**: `loglik` rel `2.7e-16`, item params `2.6e-8 / 2.6e-8`, components
    `9.6e-8 / 9.4e-8`, raw shape vs mirt `2.9e-7`; item SEs vs R Hessian `3.2e-5` rel (latent
    `1.9e-4`), Oakes vs Hessian `1.8e-13`; EAP `4.9e-15`, vs mirt `1.9e-8 (SE 1.4e-8)`. Plain EM
    416, SQUAREM 55.
  Tolerances are those of the table above. Unit tests (spec validation, raw starting shape, exact
  standardization, grid weights on equispaced and Gauss–Hermite grids, NaN guards, monotone EM
  and SQUAREM, Oakes == Hessian, relabelled starts, delta-method components, Rasch/FIPC
  identification, nested Normal fit, degenerate-solution warnings, three components, scoring,
  rejected methods, link weights, JSON): `tests/test_m5_mixture.py`; mirt agreement:
  `tests/test_mirt_m5_mixture.py`.
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
- **Polychoric correlations (V24, M5)**: not a mirt comparison. lavaan's `lavCor` on ordered
  variables (two-step ML by `nlminb` with analytic gradient and Hessian, $|\rho|\le0.999$, default
  `zero.add = c(0.5, 0)`) is the tight reference; a lavaan warning about non-convergence stops the
  script. psych's `polychoric` uses `optimize()` at its default tolerance and `mnormt::sadmvn`, so
  it is checked to 1e-3 only, on complete data and excluding pairs with a corrected zero cell
  (psych with `global = TRUE` applies no correction); with unequal category counts psych takes
  per-pair thresholds, which differs under missing data (6e-3 on the MCAR case), so that case is
  lavaan-only. A further self-check confirms that each pair's two-step gradient vanishes at the
  estimate. The bivariate normal CDF is unit-tested against Plackett's integral.
- **Number of dimensions (V25, M5)**: deterministic quantities are computed in R from the V24
  lavaan matrices, so both sides see the same input. EFA.dimensions' `PARALLEL()` masks
  EFAtools' when both are attached; the script calls `EFAtools::` explicitly. EFA.dimensions'
  fourth-power MAP is $(\operatorname{tr}(R^{*4})-p)/(p(p-1))$ (a matrix power; it exceeds 1 on
  these data), so the fourth power is compared with an independent elementwise implementation
  and EFA.dimensions' values are only recorded. Normal-data references are stochastic on both
  sides; the test uses a fixed key and compares the means by $z$ (standard error from the
  replications, $\sqrt2$ for two independent means) and the 95th percentiles in absolute value.
- **Local dependence and model-based PA (V26, M5)**: mirt fits use `quadpts = 61`, `TOL = 1e-10`,
  `optimizer = "NR"`; flexpyirt refits with `BAEM(tol=1e-9)`, so the comparison includes the
  estimation agreement (log-likelihoods to $10^{-9}$ relative). ltm ships its own `Science` and
  `LSAT` data (factor-coded), so the script names mirt's data with `mirt::`. ltm's `unidimTest`
  is called with data and parameters rather than a fitted ltm object, so it simulates
  $\theta\sim N(0,1)$ as flexpyirt does with a standard normal fit; its tetrachorics come from
  polycor (`optimise()` tolerance), which bounds the observed-statistic agreement.
- **Local dependence beyond single-group unidimensional Normal models (V26, M7, issue #81)**:
  `validation/r/m7_local_dependence_reference.R` (mirt 1.48) writes
  `tests/reference/v26_local_dependence_extensions.json`; `tests/test_mirt_m7_local_dependence.py`
  puts mirt's estimates into a flexpyirt `FitResult` with every parameter fixed (`fit(...,
  table=)`; the EH weights are mirt's `Internals$Prior`), so the comparison is of the statistics
  alone (the log-likelihood at mirt's estimates agrees to $10^{-9}$ relative, observed
  $\le4\times10^{-15}$). Nine cases, 641 group-pairs. Largest observed differences: $X^2$
  $3\times10^{-12}$ and $G^2$ $8\times10^{-9}$ (relative; tolerance $10^{-7}$), $V$
  $3\times10^{-15}$ and Q3 $2\times10^{-15}$ (absolute; tolerance $10^{-8}$).

  | Case | mirt call | Grid | Pairs |
  |---|---|---|---|
  | two groups, 2PL, 4% MCAR | `multipleGroup(invariance = c(anchors, "free_means", "free_var"))` | 121 | 2 x 45 |
  | two groups, GRM (4 categories) | same | 121 | 2 x 15 |
  | two groups, two factors, 2PL | same with `mirt.model` | $61^2$ | 2 x 45 |
  | EH, 2PL, skewed trait, 3% MCAR | `mirt(dentype = "EH")`, `use_dentype_estimate = TRUE` | 61 | 66 |
  | Davidian curve of degree 2, same data | `mirt(dentype = "Davidian-2")`, same | 61 | 66 |
  | normal mixture, V22a's estimates | mixture weights put into `Internals$Prior` | 61 | 190 |
  | two factors, 2PL, cross-loading, 3% MCAR | `mirt(mirt.model(...))` | $31^2$ | 45 |
  | two factors, GRM (3 categories) | same | $31^2$ | 28 |
  | three factors, 2PL | same | $15^3$ | 36 |

  Known differences from mirt's defaults, all handled in the script: (1) for EH and Davidian
  fits `residuals()` takes a $N(0,1)$ prior unless `use_dentype_estimate = TRUE`; with it,
  `type = "Q3"` evaluates the residuals at the grid nodes (mirt 1.48), so Q3 is requested with
  `Theta = fscores(m, use_dentype_estimate = TRUE)`. (2) Q3 scores come from `fscores()`, whose
  grid is 121 / 61 / 31 points per dimension for 1 / 2 / 3 factors whatever the fit used.
  Single-group cases pass `Theta = fscores(m, quadpts = <the fit's>)`; `multipleGroup` cases,
  where `Theta` cannot be passed per group, are fitted on `fscores()`'s grid. mirt's default call
  is stored as `q3_default`: against flexpyirt on the fitted grid it differs by
  $2\times10^{-9}$ for $31^2$ against $61^2$ and by $1.3\times10^{-4}$ for $15^3$ against
  $31^3$ (consistent with the $4\times10^{-8}$ of the M5 cases: 61 against 121 points).
  (3) mirt has no `residuals()` for its mixture class, so the mixture case checks the statistics
  given the density, not a mirt mixture fit (the fit is V22). (4) A category not observed in a
  group makes mirt's `table()` smaller than $K_i\times K_j$; flexpyirt keeps the full table and
  this is unit-tested only. One end-to-end check refits the two-group 2PL with flexpyirt and
  agrees with mirt (fitted to `TOL = 1e-6`) to $4\times10^{-4}$ ($X^2$, $G^2$) and
  $5\times10^{-6}$ ($V$, Q3). mirt orders groups alphabetically, so its $N(0,1)$ reference group
  is `focal`; the test standardizes the same group. With the other group standardized the
  model is the same but its discretization is not (the grid is cut at $\pm6$ and the free group
  has SD 1.34): the log-likelihood differs by $1.9\times10^{-3}$ and $X^2$ by $3\times10^{-3}$. **Not validated, not
  implemented**: `jsi()` beyond single-group unidimensional Normal
  models (mirt: one factor only, no SE for Davidian curves, no mixtures, and for
  `multipleGroup` single-group $N(0,1)$ refits per group, which is another statistic).
  Bifactor and two-tier models are V38.
- **Local dependence for bifactor and two-tier models (V38, M7, issue #81)**:
  `validation/r/m7_local_dependence_bifactor_reference.R` (mirt 1.48) writes
  `tests/reference/v38_local_dependence_bifactor.json`;
  `tests/test_mirt_m7_local_dependence_bifactor.py` (fast tier) puts mirt's estimates into a
  flexpyirt `FitResult` with every parameter fixed, as V26 (M7), so the comparison is of the
  statistics alone (the log-likelihood at mirt's estimates agrees to $10^{-9}$ relative,
  observed $\le8\times10^{-16}$). mirt's `residuals(type = "LD")` sums over the full product
  grid `thetaComb(theta, nfact)` with the fit's `quadpts`; flexpyirt sums over the reduced grids
  (DESIGN.md §17.3), which is the same sum rearranged.

  | Case | mirt call | Points per dimension (mirt's grid) | Pairs | $X^2$ rel. | $G^2$ rel. | $V$ abs. | Q3 abs. |
  |---|---|---|---|---|---|---|---|
  | bifactor 2PL, three specific factors, two general-only items, 3% MCAR | `bfactor(data, c(1,1,1,2,2,2,3,3,3,NA,NA))` | 21 ($21^4$) | 55 | $5\times10^{-13}$ | $2\times10^{-11}$ | $2\times10^{-15}$ | $9\times10^{-16}$ |
  | bifactor GRM (3 categories), two specific factors | `bfactor(..., itemtype = "graded")` | 21 ($21^3$) | 28 | $2\times10^{-14}$ | $7\times10^{-13}$ | $3\times10^{-16}$ | $1\times10^{-15}$ |
  | two-tier 2PL, two correlated primary factors (one cross-loading item), two specific factors | `bfactor(..., model2 = mirt.model("F1 = 1-5\nF2 = 5-10\nCOV = F1*F2"))` | 15 ($15^4$) | 45 | $1\times10^{-11}$ | $2\times10^{-8}$ | $8\times10^{-15}$ | $3\times10^{-15}$ |

  Tolerances: $10^{-7}$ relative ($X^2$, $G^2$), $10^{-8}$ absolute ($V$, Q3). Q3 is requested
  with `Theta = fscores(m, quadpts = <the fit's>)`; mirt's default call scores on `fscores()`'s
  own grid and is stored as `q3_default` (it differs from flexpyirt on the fitted grid by
  $2\times10^{-6}$, $1\times10^{-8}$ and $2\times10^{-4}$ in the three cases). In the first
  case two items of a three-item specific factor were simulated with extra shared variance; the
  specific factor absorbs it and no pair is flagged, in mirt and in flexpyirt alike. Unit tests
  (`tests/test_m7_local_dependence_reduced.py`) compare the factorized tables with brute-force
  sums over the product grid ($7^4$ nodes, bifactor and two-tier with a GRM item and missing
  data; $X^2$ to $10^{-10}$ relative), Q3 with its definition at `score()`'s EAP, and check
  that a testlet left out of `specific` is flagged. **Not validated**: nothing estimated by
  flexpyirt is compared here (the fits are V32 and V33); item models other than 2PL and GRM;
  more than three specific factors; Gauss–Hermite grids (mirt's `residuals()` is equispaced).
  **Not implemented**: `jsi()` for these models (an item has a primary and a specific slope,
  and mirt's `type = "JSI"` requires one factor); multiple groups (bifactor and two-tier models
  are single-group).
- **DETECT (V27, M5)**: sirt's `conf.detect()` is called with `smooth = FALSE`; with the
  default `smooth = TRUE` and `use_sum_score = TRUE` it subtracts raw item scores from the
  standardized total score, which mis-forms the rest-score groups (DESIGN §16.13). `expl.detect()`
  forces `smooth = FALSE` itself. Its returned `detect.unweighted` holds only the validation-half
  indices, so the script recomputes the estimation-half indices with `conf.detect()` on the
  estimation rows.
- **EFA and Hull (V28, M5)**: unrotated loadings are compared through the implied
  $\Lambda\Lambda^\top$ (invariant to column signs and to the order of equal eigenvalues).
  `HULL()`'s bound $J$ comes from a random parallel analysis, so the script records
  `settings$n_fac_max` and the test passes it as `max_factors`. EFAtools' `optim()`-based solutions
  are less tightly converged than flexpyirt's, so the test also checks that flexpyirt's objective is
  not above EFAtools' at EFAtools' $\psi=1-h^2$; the remaining differences are largest at Heywood
  solutions (bounded uniquenesses), which Hull includes for $k$ beyond the true number of factors.
- **Comparison data (V29, M5)**: CD is random on both sides. The generator is checked
  exactly by reproducing EFAtools' random draws in R (`set.seed`, then the same `sample()` /
  `rnorm()` calls in the generator's order), exporting them and feeding them to flexpyirt; this is
  exact for one factor, where the loadings' sign is fixed by the generator. The selection is
  checked on data sets where the answer is clear; the mean RMSEs are compared loosely because
  each side draws its own populations: 15% for the retained numbers of factors, and 60% for the
  rejected extra factor, whose single population varies much more (0.044 to 0.072 over six seeds
  where EFAtools' draw gave 0.048). JSON `auto_unbox` would turn a one-element $p$-value vector
  into a scalar, so the script writes `as.list()`.
- **DIMTEST (V30, M5)**: accepted on a simulation study instead of a reference
  comparison (DESIGN §16.16 has the table): with 200 replications the type I error at .05 is
  0.5–1.0% (conservative; the bootstrap removes the bias but $T$'s SD under the null is about 0.72)
  and the power 0.99–1.00. Rerun with `uv run python packages/flexpyctt/validation/simulation/dimtest_study.py`
  (about 4 minutes); the tests run 20 null and 10 alternative replications.
- **Classical item analysis (V46, flexpyctt, #101)**:
  `packages/flexpyctt/validation/r/ctt_item_analysis_reference.R` (R 4.6.2, psych 2.6.9, ltm
  1.2.0, polycor 0.8.2, lavaan 0.7.3) writes `packages/flexpyctt/tests/reference/v46_item_analysis.json`;
  `packages/flexpyctt/tests/test_item_analysis.py` compares. Four simulated cases: 10
  multiple-choice items ($N=500$, keys and distractors of varying pull, one item with five
  options) complete and with 5% MCAR, and 6 items scored 0–4 ($N=300$, graded model) complete and
  with 6% MCAR; plus three items for a just-identified omega and the complete 0–4 data with one
  item reverse-scored, where psych's default `flip = TRUE` reverses it (its `flip = FALSE` value
  is also recorded, 0.18 lower, and equals the unreversed formula). Observed differences: closed forms
  (alpha, standardized alpha, if-deleted values, Feldt bounds, means, SDs, item–total and
  item–rest correlations, point-biserial, polyserial and biserial, option proportions,
  point-biserials, mean scores and group proportions) $\le6\times10^{-15}$; keyed scores
  identical; omega total $\le7\times10^{-8}$ and loadings $\le1.4\times10^{-6}$ (psych's `fa()`
  stops its minres optimization earlier); polychoric matrices against lavaan $\le2\times10^{-8}$; the reversed item is the one psych
  reverses, and omega with it reversed equals omega of the data reverse-scored by hand.
  Known differences, each asserted in the tests: (1) psych's `raw.r` correlates with
  `rowSums(na.rm = TRUE)`, so with missing data it is compared with the $C$-based
  `colSums(C)/sqrt(diag(C) * sum(C))` computed in R, and `raw.r` itself only on complete or
  listwise data (DESIGN §16.21); `r.drop` is $C$-based in psych too and is compared in every
  case. (2) ltm's raw `cronbach.alpha(na.rm = TRUE)` takes the total variance from complete rows
  and the item variances from all observed scores, which is neither rule; it is compared on
  complete data only (its standardized alpha uses complete rows and matches `"listwise"`).
  (3) psych's `biserial()` equals the two-step biserial times $\sqrt{(n-1)/n}$. (4) psych's
  `polyserial()` counts missing rows in its $n$, so it is compared on complete data; with missing
  data the two-step factor $\sigma_j/\sum_k\phi(\tau_{jk})$ is recomputed in R. (5) Option
  statistics have no installed R implementation (TAM's `tam.ctt3` is not installed); the script
  computes them in base R from their definitions, and option proportions also match psych's
  `responseFrequency()`. lavaan's one-factor CFA, listed in the issue, was not used (ML on the
  covariance matrix is another estimator and scale).
- **Observed-score DIF (V47, flexpyctt, #102)**: references from difR 6.1.0 and base R,
  committed in `packages/flexpyctt/tests/reference/v47_dif.json`. Measured agreement: MH
  $\chi^2$, $\hat\alpha_{MH}$ and Phillips–Holland variance, GMH statistics (difGMH and
  `mantelhaen.test()` on groups x categories x strata tables), Mantel statistics and odds ratios
  to $10^{-10}$ relative (the computations are closed form); purification paths (`difPur`),
  iteration counts, convergence flags and final flags equal. Logistic regression: `difLogistic()`
  calls `glm()` with its default `epsilon = 1e-8`, which leaves the coefficients and statistics
  up to $2\times10^{-4}$ (relative) from the MLE on these data, so the script replaces
  `stats::glm.control` by one with `epsilon = 1e-14`; then LRT statistics, $\Delta R^2$ and
  coefficients agree to $10^{-8}$ and SEs and Wald statistics to $10^{-6}$ (`glm()`'s `vcov()`
  uses the IRLS weights of the previous iterate; the deviance criterion stops at a step of about
  $\sqrt{\epsilon}$). Known differences, checked in the tests: (1) `mantelhaen.test()` drops the
  continuity correction when $\lvert\sum A-\sum E(A)\rvert<1/2$ (difR and flexpyctt do not; three
  items of the simulated set); (2) `difMantel.poly()` rounds its output (statistic to 3, odds
  ratio to 4 decimals) and its purification removes a flagged item from its own matching
  score, so Mantel's statistic and its purification path are recomputed in R with difR's rule
  for the other methods, and `difMantel.poly()` is a rounded cross-check; (3)
  `liu_agresti_ccor(se = "liu_agresti")` differentiates $T=\sum ad/N-\hat\psi\sum bc/N$ with
  respect to the focal cumulative count as $a/N-\hat\psi b/N$ (the sign of $a/N$ is wrong) and
  divides by $\sum bc/N$ instead of $\sum ad/N$, so its `SE_log_Psi` is neither
  $SE(\ln\hat\psi)$ nor $SE(\hat\psi)$ (0.031 against a Monte Carlo SD of 0.0296 for $\psi=1.48$,
  300 replications of $N=20000$ in three strata; 0.14 against the Phillips–Holland 0.36 on a
  `verbal` item). flexpyctt and the R script's `la_eq3()` use instead Liu & Agresti's (1996)
  eq. 3, which stays valid with many small strata (DESIGN §16.20). Its R implementation, written
  in the paper's notation and independently of the Python one, reproduces the paper's worked
  example (Table 1, 28 centres: $\ln\hat\theta=-1.153$, SE .571), and the Python tests check
  the same published values, Mantel's 4.84 included. A plug-in delta method (used before
  review; 0.0291 against the Monte Carlo 0.0296 with large strata) gives 0.375 there and is about
  40% too small with four respondents per stratum. A Python Monte Carlo test (150 strata of 4
  and of 10 respondents, a four-category proportional-odds item, 400 replications) checks that
  the mean eq. 3 SE is within 10% of the Monte Carlo SD of $\ln\hat\psi$, and another checks
  that for 0 / 1 items with large strata it is within 1% of the Phillips–Holland SE. (4) `liu_agresti_ccor()` reports $1/\hat\psi$ of
  flexpyctt (difR's orientation: above 1 favours the focal group).
- **Parameter-recovery simulations**: bias / RMSE over conditions (N, J, number of groups, number
  of anchors) × 100 replications, with results and figures in `validation/recovery/`.
- **Unconstrained M-step scale and `LogitNormalPrior` (V45, #15)**: DESIGN §15.47–§15.48.
  1. *Interior solutions on both M-step scales.* The mirt comparisons of V3, V4, V5 and V9
     (`tests/test_mirt_m2.py`: log-likelihood, item and latent parameters; the focal variance of V4
     is stepped on $\log\sigma^2$), V11 (`tests/test_mirt_m3.py`: 3PL/4PL, $\operatorname{logit}g$,
     $\operatorname{logit}u$, plus Oakes and Hessian SEs), V7 (`tests/test_mirt_m3_priors.py`: MAP,
     the slopes with `LogNormalPrior` stepped on $\log a$) and V14 (`tests/test_mirt_m4_mirt.py`:
     2–3 factor models, correlations stepped on $2\operatorname{atanh}\rho$) are repeated with
     `BAEM(m_step_scale="unconstrained")` at the same tolerances as the natural scale; all pass.
     `tests/test_m_step_scale.py` adds unit tests of the transforms and of the Newton step (same
     interior maximum, boundary approached from inside, fixed slots untouched, rejected step
     exact), a simulated 3PL MAP fit (estimates within 1e-5 of the natural scale with
     `tol=1e-8`, log posterior to 1e-12, Oakes covariance to 1e-4 relative, with and without
     SQUAREM), a two-group fit with a free focal variance, and a boundary case (items generated
     with $g=0$): both scales end with $g<10^{-4}$ on those items, raise `HeywoodWarning`, and
     agree in log-likelihood to 1e-5.
  2. *`LogitNormalPrior`.* mirt has no prior with this density on the natural scale (its `norm`
     prior on `g` acts on the internal $\operatorname{logit}g$ without the Jacobian, like
     flexMIRT's `Guessing : Normal`), so there is no mirt reference for the default
     `mode_scale="natural"`. `tests/test_logitnormal_prior.py`
     checks instead: `log_pdf` against $\log\phi(\operatorname{logit}x;\mu,s)-\log x-\log(1-x)$
     from `scipy.stats.norm` (1e-13 relative) for four $(\mu,s)$; by `scipy.integrate.quad`, the
     density integrates to 1 (1e-10) and $\operatorname{logit}X$ has mean $\mu$ (1e-9) and variance
     $s^2$ (1e-9 relative); the Core `eval_log_prior` equals `log_pdf` and inactive slots have zero,
     finite gradients; and the MAP of a simulated 3PL ($J=8$, $N=1200$,
     `LogitNormalPrior(-1.3863, 0.6)` on $g$, `NormalPrior(0, 3)` on $d$, `tol=1e-10`) on both
     M-step scales is a stationary point of $\ell(\eta)+\sum\log\pi$ with the prior gradient
     $-1/g+1/(1-g)-(\operatorname{logit}g-\mu)/(s^2g(1-g))$ written in the test (max |gradient|
     3e-8 natural, 3e-9 unconstrained; tested at 1e-6) and a strict local maximum (negative
     definite Hessian). The likelihood gradient is that of the mirt-validated marginal
     log-likelihood. The same test maximizes flexMIRT's logit-scale objective (the kernel
     without the Jacobian) by Newton's method: every $g$ lies nearer $1/2$ than with
     `LogitNormalPrior`, up to 0.037 apart, which confirms that the two estimates differ
     (DESIGN §15.48).
  3. *`LogitNormalPrior(..., mode_scale="logit")` (logit-metric mode, DESIGN §15.48).* Here mirt
     is the reference: its `PRIOR = (..., g, norm, μ, s)` is a normal prior on its internal
     $\operatorname{logit}g$ without a Jacobian. `validation/r/m7_logit_norm_prior_reference.R`
     fits a simulated 3PL ($J=8$, $N=1200$, `norm(-1.3863, 0.6)` on $g$, `norm(0, 3)` on $d$;
     NR, `TOL = 1e-10`, Oakes SEs) and checks in R that mirt's `logPrior` equals
     $\sum\log\phi(\operatorname{logit}g)+\sum\log\phi(d)$ (1e-9) and that mirt's estimate is a
     stationary point of the logit-metric objective with the gradient written in the script
     (max |gradient| 2e-9). mirt prints $\operatorname{logit}g$ and its SE; the reference stores
     $g$ and the delta-method $\mathrm{SE}(g)=g(1-g)\,\mathrm{SE}(\operatorname{logit}g)$. On both
     M-step scales flexpyirt matches mirt's log-likelihood and log posterior (tested at 1e-9
     relative; observed 4e-10 and 5e-12 absolute), $a,d,g$ (tested at 1e-6; observed 2e-8)
     and Oakes SEs of $a,d,g$ (tested at 1e-5 relative; observed 3e-8). Independently of mirt,
     on the V45 data of item 2 the MAP on both M-step scales is a stationary point of
     $\ell+\sum\log\phi(\operatorname{logit}g;\mu,s)+\sum\log\pi(d)$ with the gradient written in the
     test (max 2e-8 natural, 2e-9 unconstrained; tested at 1e-6) and a strict local maximum; it
     equals the flexMIRT mode computed there by Newton's method on $\lambda=\operatorname{logit}g$
     (5e-9; tested at 1e-7), with the same log posterior (1e-12; tested at 1e-8); its Hessian SEs
     equal the delta-method SEs from the inverse logit-scale Hessian of that independent
     objective (7e-9 relative; tested at 1e-6); and it differs from the `"natural"` mode by up to
     0.037 in $g$. Unit tests check `log_pdf` against `scipy.stats.norm.logpdf` of
     $\operatorname{logit}x$ (1e-13), the Core code `PRIOR_LOGITNORMAL_LOGIT = 5` and its
     gradient, validation of `mode_scale`, and the JSON key (round trip; absent → `"natural"`).
- **EM monotonicity**: in every validation case, the log-likelihood (posterior with MAP) is
  non-decreasing across iterations.
- **Internal SE consistency**: Oakes SEs agree with SEs from the direct `jax.hessian` of the
  marginal log-likelihood at the converged point (small cases).
- **Invariance**: results do not change with item reordering, group relabelling, or with or
  without pattern compression.
- **User-defined item models (#86, DESIGN §20)**: `tests/test_custom_items.py` re-implements the
  2PL and the GRM against the public `ItemModel` / `ItemKernel` protocols only. In one mixed model
  they reproduce the built-in fit to $10^{-12}$ (log-likelihood, iteration count, estimates, Oakes
  SEs, display thresholds and SEs, EAP/MAP/ML/WLE scores, S-X2, M2, item information); a
  two-factor model agrees to $10^{-5}$ (different starting values). This checks the extension
  machinery, not a new model: an example model that is not built in (e.g. GGUM) needs its own
  mirt comparison (V44 is reserved for GGUM) before it is presented as validated.
