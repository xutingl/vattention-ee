#!/usr/bin/env bash
#
# num_ee_threshold_sweep / b200_llama_13b — DIRECT compute-node runner for batch 16.
# Use this ON a B200 compute node (NOT the login node, and NOT via sbatch — Ray's
# GPU handoff under the SLURM batch cgroup hangs/fails on this cluster; the direct
# recipe with `ray stop --force` is the validated path, same as batch 8 used).
#
# Launch (on a compute node):
#     cd experiments/num_ee_threshold_sweep/b200_llama_13b
#     export CUDA_VISIBLE_DEVICES=0        # a free GPU on the node
#     nohup bash run_b16.sh > run_b16.log 2>&1 &
#     tail -f run_b16.log
#
# Runs all 8 batch-16 cells sequentially (3 baselines + rebatching {auto,0,2,4,6}),
# each into results/<cell>/, then summarizes. ~45 min with real weights.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/drex_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_ADDRESS=local
# Cap Ray's logical CPUs. On this shared node Ray otherwise autodetects all 224 node
# CPUs and prestarts that many workers, which hang during startup under the job cgroup
# (the "Started a local Ray instance" stall). See benchmark_runner.py ray.init.
export DREX_RAY_NUM_CPUS="${DREX_RAY_NUM_CPUS:-16}"

BS=16

run_cell () {  # <policy> <num_ee_threshold> <results-subdir>
  local out="$RESULTS_DIR/$3"
  mkdir -p "$out"
  echo "=== $3  (policy=$1 num_ee_threshold=$2)  $(date) ==="
  ray stop --force >/dev/null 2>&1 || true     # validated direct-run hygiene
  python -u "$REPO_ROOT/scripts/run_ee.py" \
    --model llama-2-13b --model_load_format auto \
    --ee_policy "$1" \
    --shallow_exit_layer 20 --conf_threshold 0.2 \
    --max_batch_size "$BS" --num_requests 100 \
    --num_ee_threshold "$2" \
    --kv_method copy --collect_conf false \
    --csv_path "$out/"
  ls "$out"/req_*.csv >/dev/null 2>&1 && echo "OK $3" || echo "FAIL $3 (see output above)"
  echo
}

run_cell off        -1 "b${BS}_off"
run_cell eager      -1 "b${BS}_eager"
run_cell average    -1 "b${BS}_average"
run_cell rebatching -1 "b${BS}_rebatching_thrauto"
run_cell rebatching  0 "b${BS}_rebatching_thr0"
run_cell rebatching  2 "b${BS}_rebatching_thr2"
run_cell rebatching  4 "b${BS}_rebatching_thr4"
run_cell rebatching  6 "b${BS}_rebatching_thr6"

echo "=== summarize  $(date) ==="
python -u "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== done  $(date) ==="
