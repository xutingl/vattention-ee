#!/usr/bin/env bash
#
# sanity_check / b200_llama_13b
# -----------------------------
# Decode-throughput sweep: every EE policy + the non-EE baseline, on Llama-2-13B,
# at exit layer 20 / conf 0.2, for batch sizes 4 and 8 (7 policies x 2 = 14 runs).
#
# Direct-run (current setup). Launch on a B200 compute node with nohup:
#     cd experiments/sanity_check/b200_llama_13b
#     export CUDA_VISIBLE_DEVICES=0          # a free GPU on the shared node
#     nohup bash run.sh > run.log 2>&1 &
#     tail -f run.log
#
# Each run writes a CSV to results/ (named by run_ee.py / benchmark_runner.py) and a
# per-run log to results/log_<policy>_batch<bs>.txt. After the sweep, summarize.py
# prints the decode-throughput table to results/summary.txt.
#
# NOTE: -e is intentionally OFF. run_ee.py swallows subprocess errors, and we want
# one failing policy to be a recorded data point (missing CSV + its log), not an
# abort of the whole sweep.
set -uo pipefail

# --- locate paths (this file: experiments/sanity_check/b200_llama_13b/run.sh) ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

# --- environment (venv + local CUDA toolkit + library paths) ---
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/drex_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_ADDRESS=local

# --- experiment config (copy + edit this block for a new experiment) ---
MODEL="llama-2-13b"
EXIT_LAYER=20
CONF=0.2
NUM_REQUESTS=100
KV_METHOD="copy"
LOAD_FORMAT="auto"      # real pretrained weights (use "dummy" for a fast plumbing run)
POLICIES=(off eager lazy average median rebatching latency-only)
BATCH_SIZES=(4 8)

echo "=== sanity_check / b200_llama_13b ==="
echo "model=$MODEL load_format=$LOAD_FORMAT exit_layer=$EXIT_LAYER conf=$CONF num_requests=$NUM_REQUESTS kv=$KV_METHOD"
echo "policies=(${POLICIES[*]}) batch_sizes=(${BATCH_SIZES[*]})"
echo "results -> $RESULTS_DIR"
echo "GPU=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

run_idx=0
total=$(( ${#POLICIES[@]} * ${#BATCH_SIZES[@]} ))
for bs in "${BATCH_SIZES[@]}"; do
  for policy in "${POLICIES[@]}"; do
    run_idx=$((run_idx + 1))
    echo "=== [$run_idx/$total] policy=$policy batch=$bs ==="
    # Stale-Ray hygiene before each run (shared node; /tmp/ray sessions linger).
    ray stop --force >/dev/null 2>&1 || true
    python "$REPO_ROOT/scripts/run_ee.py" \
      --model "$MODEL" \
      --model_load_format "$LOAD_FORMAT" \
      --ee_policy "$policy" \
      --shallow_exit_layer "$EXIT_LAYER" \
      --conf_threshold "$CONF" \
      --max_batch_size "$bs" \
      --num_requests "$NUM_REQUESTS" \
      --kv_method "$KV_METHOD" \
      --collect_conf false \
      --csv_path "$RESULTS_DIR/" \
      2>&1 | tee "$RESULTS_DIR/log_${policy}_batch${bs}.txt"
    echo
  done
done

echo "=== sweep done; summarizing decode throughput ==="
python "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== summary written to $RESULTS_DIR/summary.txt ==="
