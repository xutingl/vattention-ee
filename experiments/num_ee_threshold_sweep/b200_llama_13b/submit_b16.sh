#!/usr/bin/env bash
# Submit all batch-16 cells of num_ee_threshold_sweep as CONCURRENT sbatch jobs
# (one B200 GPU each). Run from a login node:  bash submit_b16.sh
#
# Each cell writes results/<SUB>/req_*.csv and a SLURM log under logs/.
# After they all finish, summarize with:  python summarize.py --results-dir results
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

BS=16

submit () {  # <ee_policy> <num_ee_threshold> <results-subdir>
  sbatch --job-name="drex-$3" \
         --export=ALL,POLICY="$1",BS="$BS",THR="$2",SUB="$3" \
         cell.sbatch
}

# Baselines (num_ee_threshold ignored by off/eager/average; pass -1).
submit off     -1 "b${BS}_off"
submit eager   -1 "b${BS}_eager"
submit average -1 "b${BS}_average"
# Rebatching: ART (auto = -1) + fixed thresholds.
submit rebatching -1 "b${BS}_rebatching_thrauto"
submit rebatching  0 "b${BS}_rebatching_thr0"
submit rebatching  2 "b${BS}_rebatching_thr2"
submit rebatching  4 "b${BS}_rebatching_thr4"
submit rebatching  6 "b${BS}_rebatching_thr6"

echo "Submitted 8 batch-16 cells. Watch: squeue --me"
