#!/usr/bin/env bash
#
# overhead_scaling / b200_llama_13b
# ---------------------------------
# Per batch in {8,16,32,64,128}: off / median / rebatch@thr2 / rebatch@auto /
# rebatch@thr2+profile. Then summarize. Shows how c and c/t_d scale with batch.
#
# Launch (on a B200 compute node):
#     cd experiments/overhead_scaling/b200_llama_13b
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
export DREX_RAY_NUM_CPUS="${DREX_RAY_NUM_CPUS:-16}"

MODEL="llama-2-13b"
EXIT_LAYER=20
CONF=0.2
NUM_REQUESTS=100
KV_METHOD="copy"
LOAD_FORMAT="auto"
BATCHES=(8 16 32 64 128)

echo "results -> $RESULTS_DIR ; GPU=$CUDA_VISIBLE_DEVICES ; batches=${BATCHES[*]}"

# run_cell <label> <bs> <policy> <thr> <profile:0|1>
run_cell() {
  local label="$1" bs="$2" policy="$3" thr="$4" prof="$5"
  local out="$RESULTS_DIR/${label}_b${bs}"
  mkdir -p "$out"
  local extra=()
  unset DREX_EE_PROFILE_OUT DREX_EE_PROFILE_SYNC
  if [[ "$prof" == "1" ]]; then
    extra+=(--ee_profile)
    export DREX_EE_PROFILE_OUT="$RESULTS_DIR/prof_b${bs}.json"
    export DREX_EE_PROFILE_SYNC=1
  fi
  echo "=== ${label} b=${bs} (policy=${policy} thr=${thr} prof=${prof})  $(date) ==="
  ray stop --force >/dev/null 2>&1 || true
  python -u "$REPO_ROOT/scripts/run_ee.py" \
    --model "$MODEL" --model_load_format "$LOAD_FORMAT" \
    --ee_policy "$policy" \
    --shallow_exit_layer "$EXIT_LAYER" --conf_threshold "$CONF" \
    --max_batch_size "$bs" --num_requests "$NUM_REQUESTS" \
    --num_ee_threshold "$thr" \
    --kv_method "$KV_METHOD" --collect_conf false \
    "${extra[@]}" \
    --csv_path "$out/" 2>&1 | tee "$RESULTS_DIR/log_${label}_b${bs}.txt"
  ls "$out"/req_*.csv >/dev/null 2>&1 && echo "OK ${label} b=${bs}" || echo "FAIL ${label} b=${bs}"
  echo
}

echo "=== overhead_scaling / b200_llama_13b  $(date) ==="

# Feasibility gate: smoke b=128 off with few requests first; skip b=128 if it fails.
SKIP_128=0
unset DREX_EE_PROFILE_OUT DREX_EE_PROFILE_SYNC
echo "=== feasibility smoke: off b=128 (10 req) ==="
ray stop --force >/dev/null 2>&1 || true
if ! python -u "$REPO_ROOT/scripts/run_ee.py" --model "$MODEL" --model_load_format "$LOAD_FORMAT" \
      --ee_policy off --shallow_exit_layer "$EXIT_LAYER" --conf_threshold "$CONF" \
      --max_batch_size 128 --num_requests 10 --num_ee_threshold -1 \
      --kv_method "$KV_METHOD" --collect_conf false \
      --csv_path "$RESULTS_DIR/smoke_b128/" > "$RESULTS_DIR/log_smoke_b128.txt" 2>&1 \
   || ! ls "$RESULTS_DIR"/smoke_b128/req_*.csv >/dev/null 2>&1; then
  echo "WARN: b=128 smoke failed (likely OOM) -> skipping b=128 cells. See log_smoke_b128.txt"
  SKIP_128=1
fi

for bs in "${BATCHES[@]}"; do
  if [[ "$bs" == "128" && "$SKIP_128" == "1" ]]; then
    echo "=== SKIP b=128 (feasibility) ==="; continue
  fi
  run_cell off          "$bs" off         -1 0
  run_cell median       "$bs" median      -1 0
  run_cell rebatch      "$bs" rebatching   2 0
  run_cell rebatch_auto "$bs" rebatching  -1 0
  run_cell prof         "$bs" rebatching   2 1   # writes prof_b<bs>.json
done

echo "=== summarize  $(date) ==="
python -u "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== done  $(date) ==="
