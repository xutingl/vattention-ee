#!/usr/bin/env bash
#
# num_ee_threshold_sweep / b200_llama_70b — Llama-2-70B-chat, DIRECT compute-node runner.
# Same design as the 13B instance: sweep rebatching's num_ee_threshold vs non-EE (off)
# and forced-EE (eager, average) baselines, per batch size. Real weights.
#
# Launch (on a B200 compute node, NOT login/sbatch):
#     cd experiments/num_ee_threshold_sweep/b200_llama_70b
#     export CUDA_VISIBLE_DEVICES=0
#     nohup bash run.sh > run.log 2>&1 &
#     tail -f run.log
#
# 70B (~129 GiB fp16) loads on ONE B200 (~179 GiB) with ~40 GiB left for KV/activations.
# BATCH_SIZES below are set from the memory probe (see README); shrink if a cell OOMs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/drex_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_ADDRESS=local
# Cap Ray's logical CPUs (shared node otherwise prestarts ~224 workers and hangs at
# "Started a local Ray instance"). See benchmark_runner.py ray.init.
export DREX_RAY_NUM_CPUS="${DREX_RAY_NUM_CPUS:-16}"

# --- config ---
MODEL="llama-2-70b"
EXIT_LAYER=20
CONF=0.2
NUM_REQUESTS=100
KV_METHOD="copy"
LOAD_FORMAT="auto"
BATCH_SIZES=(16 8)               # confirmed by the memory probe; see README
THRESHOLDS=(auto 0 2 4 6)        # rebatching num_ee_threshold ('auto' = -1 = ART)
BASELINES=(off eager average)    # non-EE + forced-EE references (ignore the threshold)

run_one() {  # <policy> <batch> <num_ee_threshold_arg> <subdir>
  local policy="$1" bs="$2" thr_arg="$3" sub="$4"
  local out="$RESULTS_DIR/$sub"
  mkdir -p "$out"
  echo "=== $sub  (policy=$policy batch=$bs num_ee_threshold=$thr_arg)  $(date) ==="
  ray stop --force >/dev/null 2>&1 || true
  python -u "$REPO_ROOT/scripts/run_ee.py" \
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
    --csv_path "$out/"
  ls "$out"/req_*.csv >/dev/null 2>&1 && echo "OK $sub" || echo "FAIL $sub (see output above)"
  echo
}

echo "=== num_ee_threshold_sweep / b200_llama_70b ==="
echo "model=$MODEL load_format=$LOAD_FORMAT layer=$EXIT_LAYER conf=$CONF nreq=$NUM_REQUESTS kv=$KV_METHOD ray_cpus=$DREX_RAY_NUM_CPUS"
echo "batches=(${BATCH_SIZES[*]}) thresholds=(${THRESHOLDS[*]}) baselines=(${BASELINES[*]})"
echo

for bs in "${BATCH_SIZES[@]}"; do
  for policy in "${BASELINES[@]}"; do
    run_one "$policy" "$bs" "-1" "b${bs}_${policy}"
  done
  for thr in "${THRESHOLDS[@]}"; do
    if [ "$thr" = "auto" ]; then thr_arg="-1"; else thr_arg="$thr"; fi
    run_one "rebatching" "$bs" "$thr_arg" "b${bs}_rebatching_thr${thr}"
  done
done

echo "=== sweep done; summarizing  $(date) ==="
python -u "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== done  $(date) ==="
