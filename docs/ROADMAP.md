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
- EmpiricalHistogram (after deciding DESIGN §12-1).
- Priors and MAP, reporting `log_posterior`.
- [x] 3PL / 4PL (`ThreePL`, `FourPL`, `HeywoodWarning`, validated in V11).
- [ ] GPCM. SQUAREM.
- Benchmarks (wall time against mirt).
- **Acceptance**: V6, V7, V8, V11 within tolerance. Agreement also holds with SQUAREM.

## M4 Results and first multidimensional models
- A richer `FitResult` (summary, `_repr_html_`, JSON I/O, delta-method SEs for the IRT
  parameterization).
- Scoring (MAP, ML, WLE, plausible values).
- Confirmatory 2–3 dimensional MIRT (product grids).
- **Acceptance**: EAP / MAP agree with mirt's `fscores`. A two-dimensional confirmatory model
  agrees with mirt.

## M5 and later (candidates)
- M-step for equality constraints across items.
- Davidian-curve and normal-mixture latent distributions.
- Differentiable linking (Haebara / Stocking–Lord and generalizations) on top of the library.
- DIF (multiple-group likelihood-ratio tests), item fit (S-X2), M2.
