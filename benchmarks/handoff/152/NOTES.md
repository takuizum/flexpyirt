# #152 handoff (E-step fast paths) — WIP, delete this folder before the PR is merged

Machine: Apple M3, 8 cores, **heavily loaded by other sessions** (load average 34–300 during every
measurement below). Wall-clock numbers are noisy; treat them as indicative only and re-measure on
a quiet machine (`benchmarks/run_flexpyirt.py`, baseline = `feature/150-newton-speedup` in a
separate worktree with its own `uv sync`, run back to back).

## Scripts (all take the repo root as first argument; run with `uv run python`)

- `micro.py <repo> [case ...]` — cold + warm (median of 3) `fit()` time, plain EM, `se=None`.
  Cases: `2pl20k 2pl5k 3pl5k grm gpcm mirt2d 2grp` (needs `uv run python benchmarks/data.py`).
- `prof.py <repo> case...` — jitted `_log_joint`, `marginal_loglik`, `e_step` timings at eta0.
- `prof2.py <repo> case...` — E-step pieces: softmax with 1 vs 2 exps, the r-bar einsum with/without
  the group one-hot, half-width `Y^T W`.
- `protob.py` — isolated shapes: generic vs half-width vs rank-D log joint, softmax, r-bar
  (wall and process CPU time).
- `sens.py` — sensitivity of the *baseline* EM trajectory to a 1e-14 start perturbation vs the
  fast-vs-generic difference (needs `tests/` on the path; run from the repo root).

## Measurements so far

`micro.py` warm, plain EM (s), load ~66–78, same session back to back:

| case | base (#150) | after (c) buckets only |
|---|---|---|
| 2PL 20000x40 | 4.08 | 1.94 |
| 3PL 5000x30 | 1.00 | 0.88 |

`prof.py` before any change (ms, load ~35): 2PL 20k e_step 93.8 (log_joint 10.9, marg 21.1);
2D MIRT 5000x20 Q=961 e_step 189 (log_joint 13.9, marg 127).

`prof2.py` (2PL 20k, P=24576 after buckets): r-bar einsum with group one-hot 49.8 ms, without
18.4 ms, half-width `Y^T W` 9.0 ms.

`protob.py` process CPU ms (median):

| shape P,J,Q,D | log joint gen / half / rank-D | softmax 2 exp / 1 exp | r-bar gen / half |
|---|---|---|---|
| 24576,40,61,1 | 16.3 / 9.8 / 2.5 | 17.6 / 12.5 | 28.1 / 15.7 |
| 6144,20,961,2 | 32.7 / 23.8 / 10.7 | 58.0 / 38.0 | 47.2 / 29.2 |
| 6144,30,61,1 | 2.7 / 2.0 / 0.4 | 4.6 / 3.4 | 4.8 / 3.0 |

End-to-end timings of the final code (all four changes) were NOT measured yet.

## Sensitivity finding (`sens.py`)

Max |eta difference| after N iterations (generic path with start*(1+1e-14) vs generic;
fast path vs generic):

| groups | method | N | generic perturbed | fast vs generic |
|---|---|---|---|---|
| 1 | EM | 5 | 5.8e-9 | 1.2e-9 |
| 1 | EM | 25 | 3.5e-10 | 2.7e-10 |
| 1 | SQUAREM | 25 | 1.1e-14 | 2.6e-14 |
| 3 | EM | 25 | 2.6e-9 | 1.1e-8 |
| 3 | SQUAREM | 5 | 5.1e-8 | 2.7e-8 |
| 3 | SQUAREM | 25 | 4.7e-4 | 1.6e-3 |

So the EM map (M-step / SQUAREM decisions) amplifies rounding by up to ~1e10 over 25 iterations on
its own; multi-iteration trajectories cannot be compared at 1e-12. Tests therefore compare
fixed-point quantities (E-step, loglik, Oakes/Hessian information) at 1e-12/1e-10, one EM
iteration at 1e-10, and converged fits to the convergence tolerance.
