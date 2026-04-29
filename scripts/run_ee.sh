#!/bin/bash
# Sweep: qwen-14b-chat, 200 requests, shallow_exit_layer=20
# Policies: off, rebatching
# conf_thresholds: 0.01 0.02 0.05 0.1 0.25 0.5
# Modes: baseline (use_router_aware=false) and router-aware (use_router_aware=true)
# Skips any run whose output CSV already exists.

set -e

MODEL="qwen-14b-chat"
NUM_REQUESTS=200
SHALLOW_LAYER=20
MAX_BATCH=8
QPS=1.0
KV_METHOD="copy"
DATASET="xsum"
COST_FLUSH_THRESHOLD="4.0"
OUTPUT_BASE="/workspace/vattention-ee/outputs_qwen_apr_28"

mkdir -p "$OUTPUT_BASE"

CONF_THRESHOLDS=("0.01" "0.02" "0.05" "0.1" "0.25" "0.5")
EE_POLICIES=("rebatching")
ROUTER_MODES=("false" "true")

# CSV filename mirrors benchmark_runner.py naming:
# req_{N}_batch_{B}_layer_{L}_conf_{C}_{policy}_{kv}_{router_tag}[_thresh{T}].csv
csv_name() {
    local conf=$1 policy=$2 router_aware=$3
    local router_tag thresh_tag
    if [ "$router_aware" = "true" ]; then
        router_tag="router_aware"
        thresh_tag="_thresh${COST_FLUSH_THRESHOLD}"
    else
        router_tag="baseline"
        thresh_tag=""
    fi
    echo "${OUTPUT_BASE}/req_${NUM_REQUESTS}_batch_${MAX_BATCH}_layer_${SHALLOW_LAYER}_conf_${conf}_${policy}_${KV_METHOD}_${router_tag}${thresh_tag}.csv"
}

# --- off baseline (no EE) ---
off_csv=$(csv_name "0.5" "off" "false")
if [ -f "$off_csv" ]; then
    echo "[run_ee.sh] skipping policy=off (CSV exists: $(basename $off_csv))"
else
    outfile="${OUTPUT_BASE}/req_${NUM_REQUESTS}_batch_${MAX_BATCH}_layer_${SHALLOW_LAYER}_off.txt"
    echo "[run_ee.sh] policy=off"
    python run_ee.py \
      --model "$MODEL" \
      --ee_policy off \
      --max_batch_size $MAX_BATCH \
      --num_requests $NUM_REQUESTS \
      --shallow_exit_layer $SHALLOW_LAYER \
      --conf_threshold "0.5" \
      --qps $QPS \
      --kv_method "$KV_METHOD" \
      --dataset_name "$DATASET" \
      --use_router_aware "false" \
      --csv_path "$OUTPUT_BASE/" \
      > "$outfile"
fi

# --- EE policies: sweep conf_threshold × router_aware mode ---
for router_aware in "${ROUTER_MODES[@]}"; do
    if [ "$router_aware" = "true" ]; then
        mode_str="router_aware"
    else
        mode_str="baseline"
    fi

    for policy in "${EE_POLICIES[@]}"; do
        for conf in "${CONF_THRESHOLDS[@]}"; do
            expected_csv=$(csv_name "$conf" "$policy" "$router_aware")
            if [ -f "$expected_csv" ]; then
                echo "[run_ee.sh] skipping policy=${policy} conf=${conf} mode=${mode_str} (CSV exists)"
                continue
            fi
            conf_str="${conf/0./}"
            outfile="${OUTPUT_BASE}/req_${NUM_REQUESTS}_batch_${MAX_BATCH}_conf_${conf_str}_layer_${SHALLOW_LAYER}_${policy}_${mode_str}.txt"
            echo "[run_ee.sh] policy=${policy} conf=${conf} mode=${mode_str}"
            python run_ee.py \
              --model "$MODEL" \
              --ee_policy "$policy" \
              --max_batch_size $MAX_BATCH \
              --num_requests $NUM_REQUESTS \
              --shallow_exit_layer $SHALLOW_LAYER \
              --conf_threshold "$conf" \
              --qps $QPS \
              --kv_method "$KV_METHOD" \
              --dataset_name "$DATASET" \
              --use_router_aware "$router_aware" \
              --cost_flush_threshold "$COST_FLUSH_THRESHOLD" \
              --csv_path "$OUTPUT_BASE/" \
              > "$outfile"
        done
    done
done

echo "[run_ee.sh] All done. Results in $OUTPUT_BASE"
