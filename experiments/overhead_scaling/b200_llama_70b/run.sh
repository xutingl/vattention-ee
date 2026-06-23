#!/usr/bin/env bash
#
# overhead_scaling / b200_llama_70b
# ---------------------------------
# Same experiment as b200_llama_13b but on Llama-2-70B (80 layers), exit layer 30
# (deep = 50 of 80 layers), conf 0.01 (aggressive early exit). Motivation: on 13B the
# deep flush t_d collapsed to ~= a full pass t_f
# because 13B decode is fixed-cost/bandwidth dominated, so EE showed no benefit. 70B
# has a much heavier per-layer forward, so the deep half should be a real fraction of
# the full pass and EE/rebatching should matter.
#
# Per batch in {8,16,32,64,128}: off / median / rebatch@thr2 / rebatch@auto /
# rebatch@thr2+profile. 70B is ~140GB in fp16, so large batches may OOM; batches run
# ASCENDING and the sweep STOPS at the first batch whose `off` cell fails to produce a
# CSV (OOM is monotonic in batch size). Then summarize.
#
# Launch (on a B200 compute node):
#     cd experiments/overhead_scaling/b200_llama_70b
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

MODEL="llama-2-70b"
EXIT_LAYER=30
CONF=0.01
NUM_REQUESTS=100
KV_METHOD="copy"
LOAD_FORMAT="auto"
BATCHES=(8 16 32 64 128)

echo "results -> $RESULTS_DIR ; GPU=$CUDA_VISIBLE_DEVICES ; model=$MODEL exit=$EXIT_LAYER ; batches=${BATCHES[*]}"

# run_cell <label> <bs> <policy> <thr> <profile:0|1> ; returns 0 if a CSV was produced
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
  if ls "$out"/req_*.csv >/dev/null 2>&1; then echo "OK ${label} b=${bs}"; return 0; else echo "FAIL ${label} b=${bs}"; return 1; fi
}

echo "=== overhead_scaling / b200_llama_70b  $(date) ==="

for bs in "${BATCHES[@]}"; do
  # The off cell doubles as the feasibility test for this batch (KV pool is provisioned
  # for 2*bs+1 slots regardless of policy, so off OOMs iff the whole batch would).
  if ! run_cell off "$bs" off -1 0; then
    echo "=== STOP: off b=${bs} produced no CSV (likely OOM). Skipping b=${bs} and all larger batches. ==="
    break
  fi
  run_cell median       "$bs" median      -1 0
  run_cell rebatch      "$bs" rebatching   2 0
  run_cell rebatch_auto "$bs" rebatching  -1 0
  run_cell prof         "$bs" rebatching   2 1   # writes prof_b<bs>.json
done

echo "=== summarize  $(date) ==="
python -u "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== done  $(date) ==="
