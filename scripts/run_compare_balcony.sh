#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=${1:-5}
EXIT_LAYER=15
MAX_NEW_TOKENS=${2:-80}

OUTDIR="/workspace/vattention-ee/outputs_balcony"
OUTFILE="${OUTDIR}/compare_native_vs_vattn_samples${NUM_SAMPLES}_layer${EXIT_LAYER}.txt"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate vattn

cd "$(dirname "$0")"

mkdir -p "$OUTDIR"

echo "Writing output to $OUTFILE"

python compare_balcony.py \
    --num_samples    "$NUM_SAMPLES" \
    --exit_layer     "$EXIT_LAYER" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "$OUTFILE"
