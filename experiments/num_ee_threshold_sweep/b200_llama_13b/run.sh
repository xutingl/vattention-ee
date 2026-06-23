#!/usr/bin/env bash
#
# num_ee_threshold_sweep / b200_llama_13b
# ---------------------------------------
# Question: does a FIXED rebatching num_ee_threshold let it escape the ART
# auto-tuning trap (negative-overhead -> threshold pinned at batch//2) and beat the
# no-EE baseline? Sweep rebatching's num_ee_threshold at batch 8 and 16, compared to
# non-EE (off) and forced-EE (eager, average) baselines. Report decode throughput and
# involuntary exits (forced_out).
#
# Real weights (load_format=auto) — EE/confidence behavior is only meaningful with them.
#
# Launch on a B200 compute node with nohup:
#     cd experiments/num_ee_threshold_sweep/b200_llama_13b
#     export CUDA_VISIBLE_DEVICES=0
#     nohup bash run.sh > run.log 2>&1 &
#     tail -f run.log
#
# Each run writes its CSV into results/<run-label>/. The benchmark CSV name does NOT
# encode num_ee_threshold, so each run gets its own subdir to avoid collisions.
#
# -e is intentionally OFF (run_ee.py swallows subprocess errors); one failing cell
# becomes a missing subdir, not an aborted sweep.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/drex_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_ADDRESS=local

# --- config (copy + edit for a new run) ---
MODEL="llama-2-13b"
EXIT_LAYER=20
CONF=0.2
NUM_REQUESTS=100
KV_METHOD="copy"
LOAD_FORMAT="auto"               # real pretrained weights ("dummy" = fast plumbing run)
BATCH_SIZES=(8 16)
THRESHOLDS=(auto 0 2 4 6)        # rebatching num_ee_threshold ('auto' = -1 = ART)
BASELINES=(off eager average)    # non-EE + forced-EE references (ignore the threshold)

# run_one <policy> <batch> <num_ee_threshold_arg> <subdir>
run_one() {
  local policy="$1" bs="$2" thr_arg="$3" sub="$4"
  local out="$RESULTS_DIR/$sub"
  mkdir -p "$out"
  echo "=== $sub  (policy=$policy batch=$bs num_ee_threshold=$thr_arg) ==="
  ray stop --force >/dev/null 2>&1 || true
  python "$REPO_ROOT/scripts/run_ee.py" \
    --model "$MODEL" \
    --model_load_format "$LOAD_FORMAT" \
    --ee_policy "$policy" \
    --shallow_exit_layer "$EXIT_LAYER" \
    --conf_threshold "$CONF" \
    --max_batch_size "$bs" \
    --num_requests "$NUM_REQUESTS" \
    --num_ee_threshold "$thr_arg" \
    --kv_method "$KV_METHOD" \
    --collect_conf false \
    --csv_path "$out/" \
    2>&1 | tee "$out/run.log"
  echo
}

echo "=== num_ee_threshold_sweep / b200_llama_13b ==="
echo "model=$MODEL load_format=$LOAD_FORMAT layer=$EXIT_LAYER conf=$CONF nreq=$NUM_REQUESTS kv=$KV_METHOD"
echo "batches=(${BATCH_SIZES[*]}) thresholds=(${THRESHOLDS[*]}) baselines=(${BASELINES[*]})"
echo "results -> $RESULTS_DIR ; GPU=$CUDA_VISIBLE_DEVICES"
echo

for bs in "${BATCH_SIZES[@]}"; do
  # baselines (num_ee_threshold is ignored by off/eager/average; pass -1)
  for policy in "${BASELINES[@]}"; do
    run_one "$policy" "$bs" "-1" "b${bs}_${policy}"
  done
  # rebatching across fixed thresholds (+ auto)
  for thr in "${THRESHOLDS[@]}"; do
    if [ "$thr" = "auto" ]; then thr_arg="-1"; else thr_arg="$thr"; fi
    run_one "rebatching" "$bs" "$thr_arg" "b${bs}_rebatching_thr${thr}"
  done
done

echo "=== sweep done; summarizing ==="
python "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== summary -> $RESULTS_DIR/summary.txt ==="
