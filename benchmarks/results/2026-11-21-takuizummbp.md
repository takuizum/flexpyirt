# Benchmark results: 2026-11-21 (takuizummbp)

## Environment

- **Host**: `takuizummbp` (macOS-27.2.1-arm64-arm-64bit)
- **CPU**: Apple M5 (10 logical cores, 24.0 GiB RAM)
- **Python / flexpyirt**: Python 3.11.14, flexpyirt `0.0.1.dev61+ge0e93e434`, JAX `0.10.2` (`JAX_PLATFORMS=cpu`, devices: `CpuDevice(id=0)`), NumPy `2.4.6`, SciPy `1.17.1`, pandas `3.1.1`
- **R / mirt**: R version 4.6.0 (2026-04-24), mirt `1.46.1`
- **Settings**: `TOL = 0.0001`, quadrature `61` points on `[-6, 6]`, max iterations `2000`, flexpyirt warm repeats: 3, mirt repeats: 3

## Wall time (seconds)

Cold time is the first `fit()` call in a fresh Python process (includes JAX/XLA compilation). Warm time is the median of repeated calls after the first. Ratios `< 1.00×` mean flexpyirt (or SQUAREM) is faster; `> 1.00×` means slower.

| Case | SE | plain cold (s) | plain warm (s) | SQUAREM cold (s) | SQUAREM warm (s) | mirt default (s) | mirt NR/none (s) | plain cold / mirt def | plain warm / mirt def | SQUAREM warm / mirt def | SQUAREM / plain (warm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2PL (1000 × 20) | None | 0.666 | 0.027 | 0.982 | 0.022 | 0.053 | 0.044 | 12.57× | 0.51× | 0.42× | 0.83× |
| 2PL (1000 × 20) | Oakes | 0.882 | 0.036 | 1.210 | 0.031 | 0.165 | 0.152 | 5.34× | 0.22× | 0.19× | 0.87× |
| 2PL (5000 × 40) | None | 0.773 | 0.135 | 1.061 | 0.093 | 0.393 | 0.405 | 1.97× | 0.34× | 0.24× | 0.69× |
| 2PL (5000 × 40) | Oakes | 1.405 | 0.354 | 1.562 | 0.300 | 1.931 | 1.935 | 0.73× | 0.18× | 0.16× | 0.85× |
| 2PL (20000 × 40) | None | 1.019 | 0.399 | 1.235 | 0.269 | 0.891 | 1.212 | 1.14× | 0.45× | 0.30× | 0.67× |
| 2PL (20000 × 40) | Oakes | 7.560 | 4.007 | 9.596 | 4.489 | 6.768 | 7.102 | 1.12× | 0.59× | 0.66× | 1.12× |
| GRM K=4 (2000 × 20) | None | 0.901 | 0.097 | 1.228 | 0.069 | 0.171 | 0.145 | 5.27× | 0.57× | 0.40× | 0.71× |
| GRM K=4 (2000 × 20) | Oakes | 1.221 | 0.145 | 1.564 | 0.118 | 0.595 | 0.577 | 2.05× | 0.24× | 0.20× | 0.81× |
| 3PL (5000 × 30) | None | 1.067 | 0.209 | 1.483 | 0.122 | 1.005 | 0.657 | 1.06× | 0.21× | 0.12× | 0.58× |
| 3PL (5000 × 30) | Oakes | 1.658 | 0.376 | 2.063 | 0.285 | 2.350 | 2.019 | 0.71× | 0.16× | 0.12× | 0.76× |
| 2-group 2PL (2×3000 × 30) | None | 1.111 | 0.461 | 1.172 | 0.176 | 0.871 | 1.920 | 1.28× | 0.53× | 0.20× | 0.38× |
| 2-group 2PL (2×3000 × 30) | Oakes | 1.733 | 0.743 | 1.737 | 0.462 | 2.943 | 3.985 | 0.59× | 0.25× | 0.16× | 0.62× |

## Iterations and log-likelihood sanity check

SQUAREM reports `iter (EM evals, safeguards)`. `|Δ ll|` is the absolute difference in marginal log-likelihood against mirt default (at `TOL = 1e-4`).

| Case | SE | Patterns | Params | plain iter | SQUAREM iter (evals, sg) | mirt def iter | mirt NR/none iter | |Δ ll| plain vs mirt | |Δ ll| SQUAREM vs mirt | Status / notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2PL (1000 × 20) | None | 966 | 40 | 18 | 4 (12, 0) | 16 | 16 | 1.42e-07 | 1.36e-06 | ok |
| 2PL (1000 × 20) | Oakes | 966 | 40 | 18 | 4 (12, 0) | 16 | 16 | 1.42e-07 | 1.36e-06 | ok |
| 2PL (5000 × 40) | None | 4990 | 80 | 32 | 6 (18, 0) | 27 | 29 | 4.47e-04 | 1.65e-04 | ok |
| 2PL (5000 × 40) | Oakes | 4990 | 80 | 32 | 6 (18, 0) | 27 | 29 | 4.47e-04 | 1.65e-04 | ok |
| 2PL (20000 × 40) | None | 19859 | 80 | 28 | 5 (15, 0) | 18 | 27 | 9.59e-04 | 1.45e-03 | ok |
| 2PL (20000 × 40) | Oakes | 19859 | 80 | 28 | 5 (15, 0) | 18 | 27 | 9.59e-04 | 1.45e-03 | ok |
| GRM K=4 (2000 × 20) | None | 1999 | 80 | 24 | 5 (15, 0) | 31 | 25 | 1.55e-04 | 1.77e-04 | ok |
| GRM K=4 (2000 × 20) | Oakes | 1999 | 80 | 24 | 5 (15, 0) | 31 | 25 | 1.55e-04 | 1.77e-04 | ok |
| 3PL (5000 × 30) | None | 4906 | 90 | 51 | 8 (24, 1) | 118 | 57 | 6.45e-04 | 8.38e-04 | ok |
| 3PL (5000 × 30) | Oakes | 4906 | 90 | 51 | 8 (24, 1) | 118 | 57 | 6.45e-04 | 8.38e-04 | ok |
| 2-group 2PL (2×3000 × 30) | None | 5950 | 102 | 105 | 11 (33, 1) | 48 | 108 | 7.00e-05 | 7.84e-04 | ok |
| 2-group 2PL (2×3000 × 30) | Oakes | 5950 | 102 | 105 | 11 (33, 1) | 48 | 108 | 7.00e-05 | 7.84e-04 | ok |

## mirt default settings resolved per case

- `2pl_n1000_j20`: optimizer=`BFGS`, accelerate=`Ramsay`
- `2pl_n5000_j40`: optimizer=`BFGS`, accelerate=`Ramsay`
- `2pl_n20000_j40`: optimizer=`BFGS`, accelerate=`Ramsay`
- `grm4_n2000_j20`: optimizer=`BFGS`, accelerate=`Ramsay`
- `3pl_n5000_j30`: optimizer=`BFGS`, accelerate=`Ramsay`
- `twogroup_2pl_n6000_j30`: optimizer=`nlminb`, accelerate=`Ramsay`
