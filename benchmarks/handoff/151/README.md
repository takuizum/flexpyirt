# #151 Oakes SE overhaul — handoff (WIP, delete before merge)

- `prof.py`: run from a checkout that has `benchmarks/data` generated; fits with `se=None`,
  recompiles the problem, then times `information_oakes` / `information_cross_product` /
  the fused `information_oakes_and_cross_product` (cold, warm min and median of 7) and saves the
  matrices to `$OUTDIR`. Usage:
  `OUTDIR=out PYTHONPATH=<src-to-test> uv run python prof.py <checkout> <case> oakes [crossprod|fused]`.
- `variants.log`: rep 1 of the variant comparison (rep 2 not run; the machine was heavily loaded by
  other sessions, so absolute numbers are inflated — compare within a line only).
  `base` = feature/150-newton-speedup (f7937d5), `both` = this branch (kept code),
  `tri` = upper-triangle C only (ignores `complete`), `drop` = complete-family column drop only
  (full `(Q, D, D)` C via vmap, old chunk-size formula).
- `var_tri_vs_kept.diff`, `var_drop_vs_kept.diff`: `_core.py` of each rejected single-change variant
  vs the kept code.
- Equivalence (same runs): base vs kept, max|Δ|/max|I| ≤ 8.0e-16 for Oakes and for the
  cross product (fused) on the five single/two-group cases.
