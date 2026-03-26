#!/bin/bash
set -e

DATASET=${1:-cnn}
NUM_REQUESTS=${2:-100}
BATCH_SIZE=${3:-8}
SHALLOW_EXIT_LAYER=${4:-20}
KV_METHOD="copy"

POLICIES=("eager" "lazy" "median" "rebatching" "latency-only")
# POLICIES=("rebatching")

#CONF_THRESHOLDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

CONF_THRESHOLDS=(0 0.025 0.05 0.1 0.25 0.5 0.75)


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS=("llama-2-13b" "qwen-14b-chat")
# MODELS=("llama-2-13b")

for MODEL in "${MODELS[@]}"; do
    for POLICY in "${POLICIES[@]}"; do
        for CONF in "${CONF_THRESHOLDS[@]}"; do
            CSV_DIR="${SCRIPT_DIR}/outputs/varying_conf_${DATASET}/${MODEL}/batch_${BATCH_SIZE}/${POLICY}"
            mkdir -p "$CSV_DIR"

            echo "=== Model: ${MODEL}, Policy: ${POLICY}, Conf: ${CONF}, Batch: ${BATCH_SIZE}, Requests: ${NUM_REQUESTS} ==="

            python "${SCRIPT_DIR}/scripts/run_ee.py" \
                --model "$MODEL" \
                --dataset_name "$DATASET" \
                --num_requests "$NUM_REQUESTS" \
                --max_batch_size "$BATCH_SIZE" \
                --shallow_exit_layer "$SHALLOW_EXIT_LAYER" \
                --conf_threshold "$CONF" \
                --ee_policy "$POLICY" \
                --kv_method "$KV_METHOD" \
                --csv_path "$CSV_DIR" \
                --num_ee_threshold 2 \
            || echo "FAILED: model=${MODEL} policy=${POLICY} conf=${CONF}"
        done
    done
done
