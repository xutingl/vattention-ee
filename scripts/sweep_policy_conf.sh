#!/usr/bin/env bash
# Sweep ee_policy x shallow_exit_layer x conf_threshold for Balcony-LLaMA-2-7B.
# Captures per-run stdout under OUT_DIR for later parsing by plot_sweep.py.
#
# Usage:  bash scripts/sweep_policy_conf.sh
#
# Tune NUM_REQUESTS / BATCH_SIZE / QPS via env vars if needed.

set -u

PYTHON=${PYTHON:-/workspace/miniconda3/envs/vattn/bin/python}
RUN_EE=${RUN_EE:-/workspace/vattention-ee/scripts/run_ee.py}
OUT_DIR=${OUT_DIR:-/workspace/vattention-ee/outputs_balcony/sweep}
NUM_REQUESTS=${NUM_REQUESTS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
QPS=${QPS:-10}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-5}

LAYERS=(15 18 21)
CONFS=(0.1 0.2 0.3)
EE_POLICIES=(lazy )

mkdir -p "$OUT_DIR"

run_one() {
    local policy=$1 layer=$2 conf=$3
    local tag="${policy}_layer${layer}_conf${conf}"
    local log="$OUT_DIR/${tag}.log"

    if [ -s "$log" ] && grep -q "Throughput:" "$log" 2>/dev/null; then
        echo "[sweep] $(date +%H:%M:%S) SKIP $tag (log exists with throughput)"
        return
    fi

    echo "[sweep] $(date +%H:%M:%S) RUN  $tag"
    "$PYTHON" "$RUN_EE" \
        --model balcony-llama-2-7b \
        --ee_policy "$policy" \
        --max_batch_size "$BATCH_SIZE" \
        --num_requests "$NUM_REQUESTS" \
        --qps "$QPS" \
        --shallow_exit_layer "$layer" \
        --conf_threshold "$conf" \
        --kv_method off \
        --dataset_name xsum \
        --csv_path "$OUT_DIR/" \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[sweep] $(date +%H:%M:%S) FAIL $tag rc=$rc (log: $log)"
    fi
    sleep "$SLEEP_BETWEEN"
}

# off: EE disabled. conf/layer are ignored but still required by CLI.
run_one off 18 0.25

for policy in "${EE_POLICIES[@]}"; do
    for layer in "${LAYERS[@]}"; do
        for conf in "${CONFS[@]}"; do
            run_one "$policy" "$layer" "$conf"
        done
    done
done

echo "[sweep] $(date +%H:%M:%S) done. logs in $OUT_DIR"
echo "[sweep] next: $PYTHON /workspace/vattention-ee/scripts/plot_sweep.py"
