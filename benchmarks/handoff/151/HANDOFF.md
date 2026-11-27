# #151 Oakes SE overhaul — handoff

Status: committed as aad85b7 on feature/151-oakes-se (WIP). Original note below; the "staged" wording predates that commit.
`.claude/worktrees/101-ctt-item-analysis-0ce4b8` (branch claude/151-oakes-se-2b54db, HEAD f7937d5 = #150 tip),
and saved as `151-oakes-se-wip.patch` (in this folder). Apply on top of f7937d5 with `git apply --index`.
Commit message draft: "WIP perf(oakes): drop category-0 column for complete families, triangular C, fused sandwich (#151)".
Push target: `git push origin HEAD:feature/151-oakes-se`; the PR goes into develop (builds on PR #158, Closes #151).

## Implemented (staged)
- `FamilyBlock.complete` (static field), set in `compile_problem` from the real pattern rows. `_core.with_responses` resets it to
  False wherever `y` is replaced (`_jsi.without_item`, `fit._sandwich_meat`). Recompile cost is per family, not per item.
- Complete families drop the category-0 one-hot column: row 0 of `L_q` gets `d_j0`, and the other columns become `d_jk - d_j0`.
- `C_q` is stored as the upper triangle, built in one GEMM `wg.T @ (x[:,iu]*x[:,ju])` with shape `(Q, D(D+1)/2)`, and is
  rebuilt symmetric in `_oakes_t1`. The chunk size uses `n_pairs + n_nodes`.
- `information_oakes_and_cross_product`: one pass that returns Oakes and the sandwich meat (raw score = centered s̄ + the
  gathered q-independent means; meat minus diag(d2_prior)). `fit()` uses it for `se="sandwich"` with no weights and no cluster.
  Fallback: `reduction` set or n_free == 0.
- Tests: 9 new cases in `tests/test_oakes.py` (complete vs missing, many chunks, blocked 4 groups, jsi reset, one-pass
  complete/missing/blocked). They pass, and so does the rest of test_oakes.
- DESIGN §26 item 4 is updated in EN and JA (triangle shape). The **§15.20 sub-bullet is not written yet**.

## Verified
- test_oakes.py: all pass, including comparison against `information_hessian` and the forced full-column path.
- Benchmark cases (5 single- and two-group): base (#150 tip) vs new, max|Δ|/max|I| ≤ 8e-16 for Oakes and for the cross
  product.
- NOT run: full `uv run pytest`, ruff check, ruff format --check, mypy, benchmarks/run_flexpyirt.py, the many-group (g200)
  variants, and rep 2.

## Measurements (rep 1, warm min of 7; heavily loaded shared machine, so compare within a row only)
| case | base oakes | tri | drop | both(kept) | base oakes+crossprod | fused |
|---|---|---|---|---|---|---|
| 2pl 20000x40 | 0.744 | 0.761 | 0.584 | 0.354 | 1.363 | 0.392 |
| 2pl 5000x40 | 0.336 | 0.161 | 0.093 | 0.046 | 0.533 | 0.060 |
| 3pl 5000x30 | 0.157 | 0.073 | 0.059 | 0.031 | 0.345 | 0.026 |
| grm4 2000x20 | 0.037 | 0.025 | 0.047 | 0.032 | 0.080 | 0.020 |
| twogroup 6000x30 | 0.083 | 0.045 | 0.020 | 0.013 | 0.166 | 0.015 |
| manygroups g200 | 2.37 (crossprod 9.7, median 35) | – | – | – | – | – |
Cold times are 1–7 s and noisy. Neither variant shows a consistent cold regression.

## Decisions
- Keep all three: (1) the column drop, (2) the triangle, (3) the fused sandwich. Both changes together beat either one alone
  in 4 of 5 cases. The exception is grm4 (tri 0.025 vs both 0.032, at 30 ms scale and under load); recheck it on an idle
  machine. The fused sandwich costs about the same as Oakes alone, roughly halving se="sandwich" time.
- No variant has been rejected yet. Only if the idle-machine rep shows tri-alone winning for polytomous items, consider
  skipping the drop for families with K > 2.

## Remaining steps
1. On an idle machine, rerun `benchmarks/handoff/151/prof.py` for all cases, including manygroups_2pl_g200 (base vs kept),
   as rep 2.
2. Write the DESIGN §15.20 sub-bullet (EN ~L679 item 20, JA ~L532): method, static-flag recompile tradeoff, fused sandwich,
   measurements, and grm4/rejected notes.
3. Run `uv run ruff check src tests`, `uv run ruff format --check`, `uv run mypy src`, then the full `uv run pytest`
   (30–40 min).
4. Run `uv run python benchmarks/data.py`, then `benchmarks/run_flexpyirt.py --out ...` for new vs base (not while tests run).
5. Remove `benchmarks/handoff/151`, commit (signed), push to feature/151-oakes-se, and open the PR into develop (builds on
   #158, Closes #151, benchmark table, attribution). Wait for green CI. Never merge.
6. If feature/150-newton-speedup moves, merge it in.

## Gotchas
- `GROUP_BLOCK_MIN_GROUPS` defaults to 2, so every multigroup problem takes the blocked path. Tests pin it explicitly.
- In prof.py, fit with se=None first; otherwise the "cold" timing is already compiled.
- `flexpyirt.fit` the attribute is the function. Use `sys.modules["flexpyirt.fit"]` to reach the module.
- Padding pattern rows are category 0 with count 0, so `complete` must be computed from `patterns[:n_patterns]`.
- Don't edit src while a test run is in progress (stale import → spurious TypeError).
