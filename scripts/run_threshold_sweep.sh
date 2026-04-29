#!/bin/bash
# Sweep cost_flush_threshold to validate the router-aware flush mechanism.
# Hypothesis: higher threshold → larger deep batches → higher avg tokens/deep-iter.
#
# Fixed: rebatching policy, conf=0.1, use_router_aware=true, 200 requests.
# Varied: cost_flush_threshold from 0.5 to 8.0.

set -e

MODEL="llama-2-13b"
NUM_REQUESTS=200
SHALLOW_LAYER=20
MAX_BATCH=8
QPS=1.0
KV_METHOD="copy"
DATASET="xsum"
OUTPUT_BASE="/workspace/vattention-ee/outputs_13b_threshold_sweep"
CONF="0.01"
POLICY="rebatching"

mkdir -p "$OUTPUT_BASE"

THRESHOLDS=("0.5" "1.0" "2.0" "3.0" "4.0" "5.0" "6.0" "7.0" "8.0")

# --- count-based baseline (router_aware=false, threshold irrelevant) ---
outfile="${OUTPUT_BASE}/req_${NUM_REQUESTS}_batch_${MAX_BATCH}_conf_${CONF}_layer_${SHALLOW_LAYER}_baseline.txt"
# echo "[run_threshold_sweep.sh] count-based baseline"
# python run_ee.py \
#   --model "$MODEL" \
#   --ee_policy "$POLICY" \
#   --max_batch_size $MAX_BATCH \
#   --num_requests $NUM_REQUESTS \
#   --shallow_exit_layer $SHALLOW_LAYER \
#   --conf_threshold "$CONF" \
#   --qps $QPS \
#   --kv_method "$KV_METHOD" \
#   --dataset_name "$DATASET" \
#   --use_router_aware "false" \
#   --cost_flush_threshold "4.0" \
#   --csv_path "$OUTPUT_BASE/" \
#   > "$outfile"

# --- router-aware sweep over cost_flush_threshold ---
for thresh in "${THRESHOLDS[@]}"; do
    thresh_str="${thresh//./_}"
    outfile="${OUTPUT_BASE}/req_${NUM_REQUESTS}_batch_${MAX_BATCH}_conf_${CONF}_layer_${SHALLOW_LAYER}_thresh_${thresh_str}.txt"
    echo "[run_threshold_sweep.sh] cost_flush_threshold=${thresh}"
    python run_ee.py \
      --model "$MODEL" \
      --ee_policy "$POLICY" \
      --max_batch_size $MAX_BATCH \
      --num_requests $NUM_REQUESTS \
      --shallow_exit_layer $SHALLOW_LAYER \
      --conf_threshold "$CONF" \
      --qps $QPS \
      --kv_method "$KV_METHOD" \
      --dataset_name "$DATASET" \
      --use_router_aware "true" \
      --cost_flush_threshold "$thresh" \
      --csv_path "$OUTPUT_BASE/" \
      > "$outfile"
done

echo "[run_threshold_sweep.sh] Done. Results in $OUTPUT_BASE"
