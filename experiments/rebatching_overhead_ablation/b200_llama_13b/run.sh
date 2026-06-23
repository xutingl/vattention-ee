#!/usr/bin/env bash
#
# rebatching_overhead_ablation / b200_llama_13b
# ---------------------------------------------
# Tests two overhead-reduction levers on Llama-2-13B, in the regime where rebatching
# overhead hurts MOST (batch 16, num_ee_threshold=2 -> heavy flushing; the −7.2% cell
# in num_ee_threshold_sweep):
#   (a) strengthen INLINE DRAINING  -> env DREX_INLINE_DRAIN_MIN
#         (llama.py deep_buffer piggyback threshold; default 2; LOWER = drain survivors
#          onto deep passes more eagerly -> fewer dedicated flushes)
#   (b) raise/adapt MIN_FLUSH_SIZE   -> env DREX_MIN_FLUSH_SIZE
#         (vllm_scheduler dedicated-flush threshold; default 8; HIGHER = wait for a
#          denser flush -> fewer, bigger flush iterations; range [1,16] at batch 16)
#
# !!! REQUIRES the env hooks added to the engine (see README "Execution checklist").
#     Without them these env vars are no-ops. Hooks are default-preserving (2 and 8).
#
# Metric: decode_tok/s vs the off (no-EE) baseline. Do (a)/(b) recover the −7%?
#
# Launch (on a B200 compute node):
#     cd experiments/rebatching_overhead_ablation/b200_llama_13b
#     export CUDA_VISIBLE_DEVICES=0
#     nohup bash run.sh > run.log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/drex_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_ADDRESS=local
export DREX_RAY_NUM_CPUS="${DREX_RAY_NUM_CPUS:-16}"   # Ray-hang fix (shared node)

# --- fixed config (the heavy-flush regime) ---
MODEL="llama-2-13b"
EXIT_LAYER=20
CONF=0.2
NUM_REQUESTS=100
KV_METHOD="copy"
LOAD_FORMAT="auto"
BS=16
THR=2

run_cell() {  # <label> <policy> <num_ee_threshold> <inline_min> <flush_size>
  local label="$1" policy="$2" thr="$3" inline="$4" flush="$5"
  local out="$RESULTS_DIR/$label"
  mkdir -p "$out"
  export DREX_INLINE_DRAIN_MIN="$inline"
  export DREX_MIN_FLUSH_SIZE="$flush"
  echo "=== $label  (policy=$policy thr=$thr inline_drain_min=$inline min_flush_size=$flush)  $(date) ==="
  ray stop --force >/dev/null 2>&1 || true
  python -u "$REPO_ROOT/scripts/run_ee.py" \
    --model "$MODEL" --model_load_format "$LOAD_FORMAT" \
    --ee_policy "$policy" \
    --shallow_exit_layer "$EXIT_LAYER" --conf_threshold "$CONF" \
    --max_batch_size "$BS" --num_requests "$NUM_REQUESTS" \
    --num_ee_threshold "$thr" \
    --kv_method "$KV_METHOD" --collect_conf false \
    --csv_path "$out/"
  ls "$out"/req_*.csv >/dev/null 2>&1 && echo "OK $label" || echo "FAIL $label (see output)"
  echo
}

echo "=== rebatching_overhead_ablation / b200_llama_13b  (batch $BS, thr=$THR) ==="
echo

#         label         policy       thr    inline  flush     # what it isolates
run_cell  off_b16       off          -1     2       8         # no-EE baseline (target throughput)
run_cell  base          rebatching   $THR   2       8         # current defaults (~ −7%)
run_cell  a_inline1     rebatching   $THR   1       8         # (a) strengthen inline draining
run_cell  b_flush4      rebatching   $THR   2       4         # (b) LOWER min_flush (more flushes; direction check)
run_cell  b_flush12     rebatching   $THR   2       12        # (b) raise min_flush
run_cell  b_flush16     rebatching   $THR   2       16        # (b) raise to batch size
run_cell  ab_combined   rebatching   $THR   1       16        # (a)+(b) together

echo "=== summarize  $(date) ==="
python -u "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== done  $(date) ==="
