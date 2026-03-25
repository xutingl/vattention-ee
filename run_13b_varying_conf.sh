#!/bin/bash
set -e

NUM_REQUESTS=${1:-100}
BATCH_SIZE=${2:-8}
SHALLOW_EXIT_LAYER=${3:-20}
KV_METHOD="copy"

POLICIES=("eager" "lazy" "median" "rebatching")
# POLICIES=("rebatching")

CONF_THRESHOLDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# CONF_THRESHOLDS=(0.5)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for POLICY in "${POLICIES[@]}"; do
    for CONF in "${CONF_THRESHOLDS[@]}"; do
        CSV_DIR="${SCRIPT_DIR}/outputs/varying_conf/llama-2-13b/batch_${BATCH_SIZE}/${POLICY}"
        mkdir -p "$CSV_DIR"

        echo "=== Policy: ${POLICY}, Conf: ${CONF}, Batch: ${BATCH_SIZE}, Requests: ${NUM_REQUESTS} ==="

        python "${SCRIPT_DIR}/scripts/run_ee.py" \
            --num_requests "$NUM_REQUESTS" \
            --max_batch_size "$BATCH_SIZE" \
            --shallow_exit_layer "$SHALLOW_EXIT_LAYER" \
            --conf_threshold "$CONF" \
            --ee_policy "$POLICY" \
            --kv_method "$KV_METHOD" \
            --csv_path "$CSV_DIR" \
            --num_ee_threshold 2 \
        || echo "FAILED: policy=${POLICY} conf=${CONF}"
    done
done
