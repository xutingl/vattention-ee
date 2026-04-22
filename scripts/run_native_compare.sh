#!/usr/bin/env bash
# Run native Balcony inference (balcony env) and compare against an existing
# vattn-ee log. Results are tee'd to outputs_balcony/.
#
# Usage:
#   bash scripts/run_native_compare.sh [num_samples] [max_new_tokens]
#   bash scripts/run_native_compare.sh 5 80

set -euo pipefail

NUM_SAMPLES=${1:-5}
EXIT_LAYER=15
MAX_NEW_TOKENS=${2:-80}

VATTN_LOG="/workspace/vattention-ee/outputs_balcony/compare_native_vs_vattn_samples${NUM_SAMPLES}_layer${EXIT_LAYER}.txt"
OUTDIR="/workspace/vattention-ee/outputs_balcony"
NATIVE_JSON="${OUTDIR}/native_balcony_samples${NUM_SAMPLES}_layer${EXIT_LAYER}.json"
COMPARE_OUT="${OUTDIR}/comparison_samples${NUM_SAMPLES}_layer${EXIT_LAYER}.txt"

source /workspace/miniconda3/etc/profile.d/conda.sh

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTDIR"

# ── Step 1: native Balcony inference in balcony env ───────────────────────────
echo "==> Running native Balcony inference (balcony env) ..."
conda run -n balcony python "$SCRIPTS_DIR/run_native_balcony.py" \
    --num_samples    "$NUM_SAMPLES" \
    --exit_layer     "$EXIT_LAYER" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output         "$NATIVE_JSON"

# ── Step 2: compare against existing vattn-ee log ────────────────────────────
if [ ! -f "$VATTN_LOG" ]; then
    echo "ERROR: vattn-ee log not found: $VATTN_LOG"
    echo "Run 'bash scripts/run_compare_balcony.sh $NUM_SAMPLES' first to generate it."
    exit 1
fi

echo "==> Comparing native outputs against $VATTN_LOG ..."
conda activate vattn
python "$SCRIPTS_DIR/compare_native_with_vattn.py" \
    --native    "$NATIVE_JSON" \
    --vattn_log "$VATTN_LOG" \
    2>&1 | tee "$COMPARE_OUT"

echo ""
echo "==> Comparison saved to $COMPARE_OUT"
