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
- [ ] Dimensionality diagnostics (DESIGN.md §16.8; question A "how many dimensions" kept apart
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
    type I error 0.0–2.0% at .05, conservative; power 0.99–1.00).
