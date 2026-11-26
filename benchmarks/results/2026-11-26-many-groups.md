# Benchmark results: 2026-11-26 (intel-xeon)

## Environment

- **Machine**: `intel-xeon` (Linux-7.0.0-1018-gcp-x86_64-with-glibc2.39)
- **CPU**: Intel(R) Xeon(R) CPU @ 2.80GHz (16 logical cores, 31.33 GiB RAM)
- **Python / flexpyirt**: Python 3.12.3, flexpyirt `0.0.1.dev627+gecf80dac1`, JAX `0.12.1` (`JAX_PLATFORMS=cpu`, devices: `CpuDevice(id=0)`), NumPy `2.5.5`, SciPy `1.18.1`, pandas `3.1.1`
- **R / mirt**: R version 4.6.2 (2026-10-29), mirt `1.48`
- **Settings**: `TOL = 0.0001`, quadrature `61` points on `[-6, 6]`, max iterations `2000`, flexpyirt warm repeats: 1, mirt repeats: 1 *(quick mode)*

## Wall time (seconds)

Cold time is the first `fit()` call in a fresh Python process (includes JAX/XLA compilation). Warm time is the median of repeated calls after the first. Ratios `< 1.00×` mean flexpyirt (or SQUAREM) is faster; `> 1.00×` means slower.

| Case | SE | plain cold (s) | plain warm (s) | SQUAREM cold (s) | SQUAREM warm (s) | mirt default (s) | mirt NR/none (s) | plain cold / mirt def | plain warm / mirt def | SQUAREM warm / mirt def | SQUAREM / plain (warm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10-group 2PL (10×200 × 20) | None | 3.752 | 1.597 | 4.411 | 0.539 | 31.345 | 125.044 | 0.12× | 0.05× | 0.02× | 0.34× |
| 10-group 2PL (10×200 × 20) | Oakes | 5.029 | 1.630 | 5.751 | 0.583 | 49.460 | 150.099 | 0.10× | 0.03× | 0.01× | 0.36× |
| 50-group 2PL (50×200 × 20) | None | 28.504 | 26.081 | 11.156 | 7.084 | 626.332 | — | 0.05× | 0.04× | 0.01× | 0.27× |
| 50-group 2PL (50×200 × 20) | Oakes | 30.198 | 26.814 | 13.147 | 7.884 | 844.333 | — | 0.04× | 0.03× | 0.01× | 0.29× |
| 200-group 2PL (200×200 × 20) | None | 387.351 | 382.858 | 92.496 | 91.208 | — | — | — | — | — | 0.24× |
| 200-group 2PL (200×200 × 20) | Oakes | 393.251 | 392.904 | 97.759 | 93.453 | — | — | — | — | — | 0.24× |

## Iterations and log-likelihood sanity check

SQUAREM reports `iter (EM evals, safeguards)`. `|Δ ll|` is the absolute difference in marginal log-likelihood against mirt default (at `TOL = 1e-4`).

| Case | SE | Patterns | Params | plain iter | SQUAREM iter (evals, sg) | mirt def iter | mirt NR/none iter | |Δ ll| plain vs mirt | |Δ ll| SQUAREM vs mirt | Status / notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10-group 2PL (10×200 × 20) | None | 1956 | 58 | 200 | 18 (54, 8) | 89 | 200 | 7.61e-05 | 1.16e-03 | ok |
| 10-group 2PL (10×200 × 20) | Oakes | 1956 | 58 | 200 | 18 (54, 8) | 89 | 200 | 7.61e-05 | 1.16e-03 | ok |
| 50-group 2PL (50×200 × 20) | None | 9594 | 138 | 777 | 56 (168, 20) | 411 | — | 5.10e-03 | 6.39e-03 | ok |
| 50-group 2PL (50×200 × 20) | Oakes | 9594 | 138 | 777 | 56 (168, 20) | 411 | — | 5.10e-03 | 6.39e-03 | ok |
| 200-group 2PL (200×200 × 20) | None | 38952 | 438 | 1439 | 101 (303, 37) | — | — | — | — | ok |
| 200-group 2PL (200×200 × 20) | Oakes | 38952 | 438 | 1439 | 101 (303, 37) | — | — | — | — | ok |

## Compile time and peak memory

flexpyirt compile time is cold minus warm time; peak memory is the worker process's peak resident set size (`ru_maxrss`, includes the JAX runtime). mirt memory is R's `gc()` "max used" since a reset before the fit (R heap only).

| Case | SE | plain compile (s) | plain peak RSS (MiB) | SQUAREM compile (s) | SQUAREM peak RSS (MiB) | mirt default R heap (MiB) |
|---|---|---:|---:|---:|---:|---:|
| 10-group 2PL (10×200 × 20) | None | 2.155 | 452 | 3.873 | 493 | 328 |
| 10-group 2PL (10×200 × 20) | Oakes | 3.398 | 479 | 5.167 | 508 | 341 |
| 50-group 2PL (50×200 × 20) | None | 2.422 | 526 | 4.072 | 683 | 521 |
| 50-group 2PL (50×200 × 20) | Oakes | 3.384 | 541 | 5.263 | 687 | 1103 |
| 200-group 2PL (200×200 × 20) | None | 4.493 | 814 | 1.288 | 980 | — |
| 200-group 2PL (200×200 × 20) | Oakes | 0.347 | 867 | 4.306 | 1003 | — |

## mirt default settings resolved per case

- `manygroups_2pl_g10`: optimizer=`nlminb`, accelerate=`Ramsay`
- `manygroups_2pl_g50`: optimizer=`nlminb`, accelerate=`Ramsay`
