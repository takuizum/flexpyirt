# Handoff for #153 (temporary; delete before the PR is merged)

- `prof.py CASE [--oakes]`: times one jitted E-step and M-step on a benchmark case (run from a checkout root).
- `bench.sh`: runs `benchmarks/run_flexpyirt.py --quick` per case, new tree then baseline tree back-to-back.
  `SP` (scratchpad dir) and the baseline worktree path `$SP/base` (detached at f7937d5) are hard-coded; edit them.
- `bench_partial.log`: partial run on an 8-core Apple machine at load average 4-7 (stopped during g50 baseline).
- `equiv.py OUT.json` (run from a checkout root, once on this branch and once on the base) and
  `compare.py BASE.json NEW.json`: equivalence check (estimates, SEs, loglik, iterations). Not run yet.
