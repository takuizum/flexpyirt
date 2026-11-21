# Benchmarks (flexpyirt vs. R `mirt`)

English | [日本語](README.ja.md)

Wall-time comparison between **flexpyirt** (JAX on CPU) and R **`mirt`** on identical deterministic
simulated datasets at `TOL = 1e-4` with 61 quadrature points on `[-6, 6]`.

Latest committed results: [results/2026-11-21-takuizummbp.md](results/2026-11-21-takuizummbp.md)
([JSON](results/2026-11-21-takuizummbp.json)).

## How to run

Manage Python with `uv` only (never `pip`). R benchmarks require R 4.4+ with `mirt` and `jsonlite`
installed.

### Full suite (~4 minutes on Apple M5; under ~15 minutes on a typical laptop)

```bash
uv run python benchmarks/data.py
uv run python benchmarks/run_flexpyirt.py
Rscript benchmarks/run_mirt.R
uv run python benchmarks/report.py
```

### Quick mode (< 1 minute)

Runs a smaller subset (`2pl_n1000_j20` and `grm4_n2000_j20`) with 1 warm repeat:

```bash
uv run python benchmarks/data.py --quick
uv run python benchmarks/run_flexpyirt.py --quick
Rscript benchmarks/run_mirt.R --quick
uv run python benchmarks/report.py --stem quick-check
```

To run flexpyirt only (without R), pass `--no-mirt` to `report.py`. To select specific cases, pass
`--cases <name>...` to `data.py`, `run_flexpyirt.py`, and `run_mirt.R`.

Simulated CSV files and raw per-engine JSON files are written to `benchmarks/data/` (git-ignored).
Merged reports are written to `benchmarks/results/<date>-<host-short>.{json,md}`.

## What is measured

### Datasets (`benchmarks/data.py`)

All datasets are simulated with fixed NumPy seeds so Python and R fit identical response matrices:

| Case | Model | Size `(N, J)` | Notes |
|---|---|---:|---|
| `2pl_n1000_j20` | 2PL | `(1000, 20)` | Single group, $N(0, 1)$ |
| `2pl_n5000_j40` | 2PL | `(5000, 40)` | Single group, $N(0, 1)$ |
| `2pl_n20000_j40` | 2PL | `(20000, 40)` | Single group, $N(0, 1)$ |
| `grm4_n2000_j20` | GRM ($K=4$) | `(2000, 20)` | Single group, $N(0, 1)$ |
| `3pl_n5000_j30` | 3PL | `(5000, 30)` | Single group, interior asymptotes ($g \in [0.12, 0.25]$) |
| `twogroup_2pl_n6000_j30` | 2-group 2PL | `(2 × 3000, 30)` | 10 anchors (`q01`–`q10`), focal $N(0.5, 1.2)$ free |

### Estimators and timing

- **flexpyirt** (`benchmarks/run_flexpyirt.py`):
  - `BAEM(tol=1e-4, max_iter=2000, squarem=False)` and `BAEM(tol=1e-4, max_iter=2000, squarem=True)`.
  - `se=None` and `se="oakes"`.
  - **Cold time**: first `fit()` call in a fresh Python subprocess (`JAX_PLATFORMS=cpu`), so it
    genuinely includes JAX/XLA compilation and cannot reuse another configuration's JIT cache.
  - **Warm time**: median of 3 repeated `fit()` calls after the cold call in the same process,
    timed with `time.perf_counter()` after results are materialized on the host as NumPy arrays.
- **R `mirt`** (`benchmarks/run_mirt.R`):
  - **Default settings** (`TOL = 1e-4`, default acceleration `"Ramsay"` and default optimizer
    `"BFGS"` / `"nlminb"`), and **validation settings** (`optimizer = "NR"`, `accelerate = "none"`).
  - `SE = FALSE` and `SE = TRUE, SE.type = "Oakes"`.
  - Same grid: `quadpts = 61` and `theta_lim = c(-6, 6)` passed inside `technical = list(...)`
    (verified against `m@Model$Theta`).
  - Median elapsed time of 3 runs via `system.time()`.
- **Sanity check** (`benchmarks/report.py`):
  - Reports iterations, flexpyirt EM map evaluations and SQUAREM safeguard counts, convergence
    status, and $|\Delta \log L|$ between flexpyirt and `mirt`.

## Caveats

- **Compile time vs. warm time**: JAX compiles each `(model family, padded pattern bucket, se)`
  computation on the first call (~0.6–1.5 s for EM on CPU, more with Oakes SE on large problems).
  For a single one-off fit of a small dataset, R `mirt` (precompiled C++) is faster than a cold
  flexpyirt call. Within a long-lived session (notebooks, simulations, repeated fits), only warm
  time is paid after the first call.
- **CPU only and threading**: `run_flexpyirt.py` forces `JAX_PLATFORMS=cpu`. XLA CPU may use
  multiple threads internally, whereas R `mirt` runs single-threaded by default unless
  `mirtCluster()` is configured. GPU acceleration is not measured here; see
  [DESIGN §9](../docs/DESIGN.md#9-performance).
- **Machine dependence**: Absolute times and ratios depend strongly on CPU architecture, memory
  bandwidth, OS, BLAS/XLA backend, and library versions. Treat the committed table as one hardware
  data point and rerun on your own machine when making performance decisions.
- **Oakes SE at large `(N, J)`**: `se="oakes"` differentiates $Q(\eta \mid \eta')$ via forward-over-reverse
  autodiff (`jacfwd(grad)`). At `N = 20000, J = 40` (19,859 unique patterns padded to 32,768),
  Oakes SE dominates total wall time (~4 s vs. ~0.3–0.4 s for EM) and shows higher run-to-run
  variance due to memory pressure.

## Note on the SQUAREM default

`BAEM(squarem=False)` is kept as the default; this benchmark does not change it. In the
[2026-11-21 Apple M5 results](results/2026-11-21-takuizummbp.md):

- **Warm runs**: `squarem=True` cuts EM evaluations by 33–69% across all 6 cases and reduces warm
  `se=None` wall time on every case (`0.38×–0.83×` of plain EM, up to 2.6× faster on two-group 2PL
  and 1.7× faster on 3PL), with at most 1 safeguard intervention per case.
- **Cold runs**: `squarem=True` adds ~0.2–0.4 s of extra JIT compilation time, making plain EM
  faster for a single cold call on 5 of the 6 cases (and essentially tied on two-group 2PL).

Whether to flip the default therefore depends on the expected workload (one-off scripts vs.
repeated/slow-converging fits) and is left to the maintainer.
