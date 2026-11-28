#!/usr/bin/env bash
# Regenerate every mirt reference in tests/reference/ (VALIDATION.md, "R and mirt versions").
# Run from the repository root after `Rscript validation/r/install_mirt.R`:
#
#   validation/r/run_references.sh            # 2 scripts at a time
#   JOBS=4 validation/r/run_references.sh     # more parallelism
#   validation/r/run_references.sh m1 m2      # only these scripts (no .R / _reference suffix)
#
# Stage 2 scripts read JSON written in stage 1 (data, and for some the fitted estimates), so they
# start after stage 1 has finished. Logs go to $LOG_DIR (default: validation/r/logs, git-ignored).
# tests/test_reference_meta.py checks that every script that loads mirt is listed below.
set -euo pipefail

STAGE1=(
  m1 m2 m3 m3_gpcm m3_eh m3_priors m4_irtpars m4_mirt m4_mirt_multigroup
  m5_davidian m5_dif_screen m5_equal_items m5_jsi m5_linking m5_linking_extensions
  m5_local_dependence m5_mixture m5_nominal
  m7_bifactor m7_information m7_information_multidim m7_local_dependence_bifactor
  m7_logit_norm_prior m7_robust_se m7_sumscore m7_two_tier m7_weights
  m6_m2_groups_missing_map classification v48_plots v50_screening v51_irt_equating
  v55_many_groups v56_multigroup_shapes v57_weighted_cluster_se v61_projection
)
# Inputs: m4_scoring (v4, v9, v11), m4_mirt_map / m4_mirt_mlwle / m6_fit_multidim (v14_mirt2d),
# m5_dif / m5_itemfit_multigroup (v4), m5_itemfit (v3, v5, v8, v11), m5_m2 (v1, v2, v11),
# m7_local_dependence (v22), person_fit (v37), v54_compare_simulate (v4, v8, v11).
STAGE2=(
  m4_scoring m4_mirt_map m4_mirt_mlwle m5_dif m5_itemfit m5_itemfit_multigroup m5_m2
  m6_fit_multidim m7_local_dependence person_fit v54_compare_simulate
)

JOBS="${JOBS:-2}"
LOG_DIR="${LOG_DIR:-validation/r/logs}"
mkdir -p "$LOG_DIR"

run_one() {
  local name="$1" script="validation/r/$1_reference.R" log="$LOG_DIR/$1.log"
  local start=$SECONDS
  if Rscript "$script" >"$log" 2>&1; then
    echo "ok     $name ($((SECONDS - start)) s)"
  else
    echo "FAILED $name ($((SECONDS - start)) s), see $log"
    return 1
  fi
}
export -f run_one
export LOG_DIR

run_stage() {
  [ "$#" -eq 0 ] && return 0
  printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
}

if [ "$#" -gt 0 ]; then
  run_stage "$@"
else
  run_stage "${STAGE1[@]}"
  run_stage "${STAGE2[@]}"
fi
