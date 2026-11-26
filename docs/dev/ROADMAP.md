# Roadmap

English | [日本語](ROADMAP.ja.md)

A milestone is complete when all of its acceptance criteria are met. Milestones are not skipped.

## M0 Scaffold
- `uv` project, `src/flexpyirt/` layout, ruff / mypy / pytest configuration, GitHub Actions
  (Linux, Python 3.11–3.13).
- `flexpyirt.errors`, the x64 check, and empty `Model` / `fit` / `FitResult` types only.
- **Acceptance**: CI passes. The package imports after an editable install
  (`uv run --isolated --no-project --with-editable .`). Environments are managed with uv only, no
  pip.

## M1 Single-group 2PL / Rasch BA-EM
- Compile layer: pattern compression, ParameterTable, FreeMap (the mechanism for fixing and
  equality constraints).
- Core: E-step, item-wise Newton M-step (vmap), EM loop in `lax.while_loop`, Normal(0,1) fixed.
- SE: Oakes and direct Hessian.
- Scoring: EAP.
- **Acceptance**: V1, V2, V10 (Oakes) within tolerance. The EM monotonicity test passes.

## M2 Multiple groups, FIPC, GRM, missing data
- Per-group latent distributions (Normal, free), equality across groups (shared `free_id`),
  identification checks.
- FIPC (fixed items + latent distribution estimated).
- GRM. Missing data (MAR).
- **Acceptance**: V3, V4, V5, V9 within tolerance. Unidentified specifications raise
  `IdentificationError`.

## M3 EH, MAP, 3PL/4PL, GPCM, SQUAREM
- [x] EmpiricalHistogram (`EmpiricalHistogram`, reference group standardized by a moment-constrained M-step, square-root-scale SQUAREM; validated in V6).
- [x] Priors and MAP (`NormalPrior`, `LogNormalPrior`, `BetaPrior`, `Model.prior()`, `FitResult.log_posterior`, validated in V7).
- [x] 3PL / 4PL (`ThreePL`, `FourPL`, `HeywoodWarning`, validated in V11).
- [x] GPCM / PCM (`GPCM`, `PCM`, validated in V8).
- [x] SQUAREM (`BAEM(squarem=True)`, scheme S3 with monotonicity/feasibility safeguard; all mirt agreement suites pass with and without SQUAREM).
- [x] Benchmarks (wall time against mirt; see `benchmarks/`).
- **Acceptance**: V6, V7, V8, V11 within tolerance. Agreement also holds with SQUAREM.

## M4 Results and first multidimensional models
- [x] `FitResult.summary()`, `FitResult._repr_html_()`, and delta-method SEs for the display IRT
  parameterization (`b_se` / `bk_se`, validated in V12).
- [x] `FitResult` JSON I/O (`FitResult.to_json`, `FitResult.from_json`).
- [x] Scoring (MAP, ML, WLE, plausible values; validated in V13).
- [x] Confirmatory 2–3 dimensional MIRT (`Model.with_factors()`, product grids, MVN correlations,
  Oakes/Hessian SEs, EAP, MAP, ML and WLE scoring; validated in V14; multiple groups validated in V21).
- **Acceptance**: EAP / MAP agree with mirt's `fscores`. A two-dimensional confirmatory model
  agrees with mirt.

## M5 and later (candidates)
- [x] M-step for equality constraints across items (`Model.equal_across_items()`, shared
  `free_id`s across items and item models; joint Newton per connected component; validated
  in V16).
- [x] Davidian-curve latent distributions (`DavidianCurve`, single-group unidimensional, analytically standardized, Oakes/Hessian SEs; validated in V19).
- [x] Normal-mixture latent distributions (`NormalMixture`, single-group unidimensional, mirt's
  mixture parameterization analytically standardized, Oakes/Hessian SEs; validated in V22).
- [x] Differentiable linking (`link()`, `LinkResult`: Haebara / Stocking–Lord, non-symmetric and
  symmetric, Ogasawara delta-method SEs via autodiff; validated in V20 for 2PL, 3PL, GRM, GPCM and
  mixed 2PL + GPCM common items).
- [x] Item fit: S-X2 (`FitResult.item_fit()`, unidimensional, per group for multiple groups; validated in V15).
- [x] DIF: multiple-group likelihood-ratio tests (`FitResult.dif()`, schemes `add` / `drop`,
  Holm / BH adjustment; validated in V17).
- [x] M2 limited-information goodness of fit (`FitResult.m2()`, single-group unidimensional
  $M_2$ / $M_2^*$, RMSEA + CI, SRMSR, dichotomous TLI / CFI; validated in V18).
- [x] Nominal response model (`Nominal(n_categories)`, Thissen, Cai & Bock (2010) parameterization as in
  mirt's `itemtype = "nominal"`; validated in V23 for one and two factors, three factors
  unit-tested only).
- [x] Dimensionality diagnostics (DESIGN.md §16.8; question A "how many dimensions" kept apart
  from question B "is essential unidimensionality tenable"; no exploratory MIRT):
  - [x] D1 polychoric / tetrachoric correlations with smoothing (`polychoric()`, `PolychoricResult`;
    validated in V24 against `lavaan::lavCor` and `psych::polychoric` / `cor.smooth`).
  - [x] D2 eigenvalues, parallel analysis, revised MAP, EKC (`parallel_analysis()`, `velicer_map()`,
    `empirical_kaiser()`; validated in V25 against EFAtools, psych and EFA.dimensions).
  - [x] D3 local dependence Q3 / Q3* / LD-X2 / G2 (`FitResult.local_dependence()`, vs `mirt::residuals`)
    and model-based parallel analysis (`model_parallel_analysis()`, vs `ltm::unidimTest`); the ensemble
    `dimensionality()` (validated in V26; the ensemble by recomputation).
  - [x] JSI (Edwards, Houts & Cai, 2018; `FitResult.jsi()` with the refit and two approximations
    without refitting; the refit validated in V31 against `mirt::residuals(type = "JSI")`).
  - [x] D4 DETECT, confirmatory and exploratory (`detect()`, `exploratory_detect()`; validated in V27
    against `sirt::conf.detect` / `expl.detect` with sum-score conditioning; kernel-smoothed
    conditioning and the genetic partition search not implemented).
  - [x] D5a ULS / ML EFA implemented in flexpyirt (`efa()`; `factor_analyzer` fails with scikit-learn
    1.8+, DESIGN §16.8) and the Hull method (`hull()`); validated in V28 against EFAtools `EFA` / `HULL`
    and `psych::fa`.
  - [x] D5b comparison data (`comparison_data()`; validated in V29 against EFAtools `CD()` and
    its generator).
  - [x] D6 DIMTEST with bootstrap bias correction (`dimtest()`; accepted in V30 on a simulation study:
    type I error 0.5–1.0% at .05, conservative; power 0.99–1.00).

## M6 Lifting the single-group / unidimensional restrictions
Each item is accepted when its mirt (or independent) comparison passes; the issue lists the reference.
- [ ] Davidian curve and normal-mixture latent distributions in multiple-group models (#75).
- [ ] M2 for multiple groups, missing data and MAP fits (#76).
- [x] M2 and S-X2 for confirmatory multidimensional models (#77): `FitResult.m2()` and
  `FitResult.item_fit()` for single-group `with_factors()` models (2–3 factors, product grid;
  validated in V39). Bifactor / two-tier and multiple-group multidimensional models still raise.
- [x] Linking for 4PL, nominal and multidimensional models (#78; `link(..., transformation=)` with a
  diagonal, orthogonal or oblique $\mathbf A$ for confirmatory two- and three-factor models;
  validated in V40).
- [ ] Many groups: per-group latent distributions vmapped over groups (#79; DESIGN §12 open question 2).
- **Acceptance**: each item validated against mirt (or an independent R implementation) within the
  VALIDATION.md tolerances.

## M7 Bifactor, diagnostics, information and survey designs (in this order)
1. [x] Bifactor and two-tier models with dimension reduction (#80).
   - [x] Bifactor models (`Model.with_bifactor()`, single group, any number of specific factors,
     Oakes/Hessian SEs, EAP scoring; validated in V32 against mirt `bfactor()`).
   - [x] Two-tier models (`Model.with_two_tier()`, one or two correlated primary factors plus
     specific factors, single group, Oakes/Hessian SEs, EAP scoring; validated in V33 against mirt
     `bfactor(model2 = )`).
2. [x] Local dependence (Q3, LD-X², JSI) for multiple groups, non-Normal latent and
   multidimensional / bifactor models (#81).
   - [x] `FitResult.local_dependence()` (Q3, Q3\*, LD-X², G²) for multiple groups, `EmpiricalHistogram` /
     `DavidianCurve` / `NormalMixture` latent distributions and confirmatory 2–3 factor models
     (validated in V26 against mirt `residuals()`; the mixture case given the density only).
   - [x] `FitResult.local_dependence()` for bifactor and two-tier models, with the expected tables on
     the reduced grids (validated in V38 against mirt `residuals()` on `bfactor()` fits).
   - Not implemented (decided, #81): JSI beyond single-group unidimensional `Normal` models. JSI is
     defined only for single-group unidimensional models; its multidimensional / multiple-group
     extension has not been studied and there is no reference implementation (mirt's
     `type = "JSI"` needs one factor). `jsi()` raises `SpecificationError` for these models
     (DESIGN.md §16.17).
3. [ ] Item / test information, reliability and summed-score EAP (#82), on a public Lord–Wingersky /
   Poisson-binomial kernel (#83).
   - [x] Kernel (#83): `summed_score_distribution()`, the Lord–Wingersky recursion as a
     Poisson-binomial kernel in linear and log space, the DFT-CF method, and normal / refined normal /
     Poisson approximations for dichotomous items; `jit` / `vmap` / autodiff; validated in V36 against
     `poibin`, `dpois` and mirt's EAPsum tables.
   - [x] Unidimensional models, single and multiple groups (#82): `item_information()`,
     `test_information()`, `expected_score()`, `reliability()` (marginal and empirical; `Normal` latent
     distributions), `summed_score_table()` and `FitResult.score(method="EAPsum")`; validated in V37
     against `iteminfo()`, `testinfo()`, `expected.item()` / `expected.test()`, `marginal_rxx()`,
     `empirical_rxx()` and `fscores(method = "EAPsum")` for 2PL, 3PL, GRM, GPCM, nominal, a
     mixed-format test and a two-group model.
   - [ ] Multidimensional models: information matrix and directional information.
4. [ ] Sampling (survey) weights (#84).
   - [x] `fit(..., weights=...)`: weighted marginal likelihood with weighted pattern compression
     for EM, SQUAREM, multiple groups, missing data and FIPC; BIC with the sum of the weights;
     Oakes / Hessian SEs with a warning (valid for frequency weights only); validated in V34
     against mirt's `survey.weights` (2PL, GRM, two groups, FIPC, bifactor). Design-consistent SEs come with
     #85; item fit, M2, local dependence, JSI and DIF raise an error for weighted fits.
5. [ ] Robust standard errors: sandwich and cross-product (#85).
   - [x] Without sampling weights: `fit(se="sandwich" | "crossprod")` for single and multiple groups,
     FIPC, EH, 2–3 factors and priors (not with Davidian curves, normal mixtures, bifactor or two-tier models); validated in V35
     against mirt `SE.type = "sandwich"` / `"crossprod"`. Weighted and cluster-robust versions still
     open.
- **Acceptance**: V-numbered comparisons with mirt `bfactor()`, `residuals()`, `iteminfo()` /
  `testinfo()` / `marginal_rxx()` / `fscores(method = "EAPsum")`, `survey.weights` and
  `SE.type = "sandwich"` / `"crossprod"`.

## Field features
- [x] Classification accuracy and consistency for cut scores (#107; `classification_accuracy()`,
  `ClassificationResult`: Lee, 2010, on summed-score cuts and Rudner, 2001, 2005, on $\theta$
  cuts, D and P approaches, kappa, per-cut false positive / negative rates, unidimensional models,
  single and multiple groups; validated in V52 against cacIRT and a recomputation from mirt).

## Ongoing
- [ ] Validate against the latest R mirt on GitHub, with a scheduled reference check (#74).
- [x] Public extension API for user-defined item models, with docs examples (#86): the
  `ItemModel` / `ItemKernel` contract, `register_item_model` for JSON, and a 2PL worked example
  in the guide (DESIGN §20); custom 2PL / GRM reproduce the built-ins to machine precision.
- [ ] Validated examples of models that are not built in (e.g. GGUM against mirt
  `itemtype = "ggum"`, V44). New item models are added this way rather than as built-ins.
