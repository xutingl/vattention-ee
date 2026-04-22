#!/bin/bash

# Activate conda environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate vattn

# Parameter sweep script for run_ee.py
# Fixed parameters:
#   - max_batch_size=4
#   - num_requests=500
#   - dataset_name=xsum
#
# Sweep parameters:
#   - ee_policies: rebatching, median, latency-only, off, greedy, lazy
#   - conf_thresholds: 0.7, 0.8, 0.9
#   - shallow_exit_layers: 20, 25, 30

# Fixed parameters
MAX_BATCH_SIZE=8
NUM_REQUESTS=500

# Output base (dataset name is inserted: outputs_xsum_sweep_feb_20 or outputs_cnn_sweep_feb_20)
OUTPUT_BASE="../"
CSV_BASE="/workspace/vattention-ee/"
SWEEP_SUFFIX="sweep_feb_20"

# Parameter arrays
DATASETS=("xsum" "cnn")
EE_POLICIES=("rebatching")
# Confidence thresholds: moderate to high confidence for quality early exits
# Lower = more aggressive (more sequences exit), Higher = more conservative
CONF_THRESHOLDS=(-1)

SHALLOW_EXIT_LAYERS=(30)

# Counter for progress tracking
total_runs=$((${#DATASETS[@]} * ${#EE_POLICIES[@]} * ${#CONF_THRESHOLDS[@]} * ${#SHALLOW_EXIT_LAYERS[@]}))
current_run=0
run_id=0

# If base path exists, return path with _1, _2, ... appended before extension
get_unique_path() {
    local base="$1"
    if [[ ! -e "$base" ]]; then
        echo "$base"
        return
    fi
    local dir name ext
    dir=$(dirname "$base")
    name=$(basename "$base")
    ext=""
    if [[ "$name" == *.* ]]; then
        ext=".${name##*.}"
        name="${name%.*}"
    fi
    local c=1
    while [[ -e "${dir}/${name}_${c}${ext}" ]]; do
        c=$((c + 1))
    done
    echo "${dir}/${name}_${c}${ext}"
}

echo "Starting parameter sweep with $total_runs total configurations"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}_${dataset}_${SWEEP_SUFFIX}/"
    CSV_PATH="${CSV_BASE}_${dataset}_${SWEEP_SUFFIX}/"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$CSV_PATH"

    for ee_policy in "${EE_POLICIES[@]}"; do
        for conf_threshold in "${CONF_THRESHOLDS[@]}"; do
            for shallow_exit_layer in "${SHALLOW_EXIT_LAYERS[@]}"; do
                current_run=$((current_run + 1))
                run_id=$((run_id + 1))

                # Create descriptive filename; append _counter if it already exists (include dataset)
                base_txt="${OUTPUT_DIR}/req_${NUM_REQUESTS}_batch_${MAX_BATCH_SIZE}_layer_${shallow_exit_layer}_conf_${conf_threshold}_${ee_policy}_${dataset}.txt"
                output_file=$(get_unique_path "$base_txt")

                # Per-run CSV directory so CSVs don't overwrite (benchmark picks the CSV filename)
                csv_path_run="${CSV_PATH}/run_${run_id}"

                echo ""
                echo "[$current_run/$total_runs] Running: dataset=${dataset}, ee_policy=${ee_policy}, conf=${conf_threshold}, layer=${shallow_exit_layer}"
                echo "Output: $output_file"
                echo "CSV dir: $csv_path_run"

                mkdir -p "$csv_path_run"
                python run_ee.py \
                    --ee_policy="${ee_policy}" \
                    --max_batch_size=${MAX_BATCH_SIZE} \
                    --num_requests=${NUM_REQUESTS} \
                    --shallow_exit_layer=${shallow_exit_layer} \
                    --conf_threshold=${conf_threshold} \
                    --dataset_name="${dataset}" \
                    --csv_path="${csv_path_run}" \
                    --kv_method="copy" \
                    --num_ee_threshold=0 \
                    --buffer_age_factor=100 \
                    > "$output_file" 2>&1
            
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    echo "  Completed successfully"
                else
                    echo "  Failed with exit code $exit_code"
                fi
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Parameter sweep complete! Ran $total_runs configurations."
echo "Results: xsum -> ${OUTPUT_BASE}_xsum_${SWEEP_SUFFIX}/ , cnn -> ${OUTPUT_BASE}_cnn_${SWEEP_SUFFIX}/"

# python scripts/run_ee.py --ee_policy=rebatching --max_batch_size=4 --num_requests=100 --shallow_exit_layer=50 --conf_threshold=0.7 --dataset_name xsum --csv_path /workspace/vattention-ee/outputs_xsum_sweep_70b/
