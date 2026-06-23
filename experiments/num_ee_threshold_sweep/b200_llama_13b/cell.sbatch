#!/bin/bash
# One cell of num_ee_threshold_sweep as a SLURM job (one B200 GPU).
# Parameters are passed via --export (see submit_b16.sh):
#   POLICY  ee_policy (off/eager/average/rebatching)
#   BS      max_batch_size
#   THR     num_ee_threshold (-1 = auto/ART; ignored by non-rebatching policies)
#   SUB     results subdir name, e.g. b16_rebatching_thr2
#SBATCH --job-name=drex-cell
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
# SLURM runs from a spool copy; SLURM_SUBMIT_DIR is the dir we submitted from
# (this instance dir, where logs/ lives and --output resolves).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

REPO_ROOT="/vast/projects/liuv/pennnetworks/xutingl/vattention-ee"
INSTANCE_DIR="$REPO_ROOT/experiments/num_ee_threshold_sweep/b200_llama_13b"

: "${POLICY:?set via --export}" "${BS:?set via --export}" "${THR:?set via --export}" "${SUB:?set via --export}"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  cell: $SUB  $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, SLURM-managed>}"
nvidia-smi -L

# DREX environment (venv on PATH + local CUDA toolkit + lib paths).
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/drex_env.sh"
# Do NOT set CUDA_VISIBLE_DEVICES — SLURM already scoped this job to its GPU.
export RAY_ADDRESS=local
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
# DREX requires clearing stale Ray state before the run, or the Ray worker hangs at
# GPU-resource registration (observed: stuck at "Started a local Ray instance" forever).
# CAVEAT: `ray stop --force` is node-global for your user, so for CONCURRENT same-node
# jobs add `#SBATCH --exclusive` (one node per cell) so a sibling's Ray isn't killed.
# Non-Ray jobs (e.g. SCAR training) are unaffected.
ray stop --force >/dev/null 2>&1 || true
# Ray-on-SLURM GPU fix: SLURM already binds this job's GPU via cgroup (visible as
# cuda:0). By default Ray REWRITES CUDA_VISIBLE_DEVICES for its worker to a Ray-internal
# GPU id that points outside the cgroup allocation -> set_device() fails with
# "CUDA-capable device(s) is/are busy or unavailable". Tell Ray to leave it alone.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_NOSET_CUDA_VISIBLE_DEVICES=1
# Unbuffered stdout so logs/*.out updates live (without this, block-buffering hides
# progress and a timeout kill discards the buffer).
export PYTHONUNBUFFERED=1

OUT="$INSTANCE_DIR/results/$SUB"
mkdir -p "$OUT"

echo ">>> launching run_ee.py at $(date)"
python -u "$REPO_ROOT/scripts/run_ee.py" \
  --model llama-2-13b --model_load_format auto \
  --ee_policy "$POLICY" \
  --shallow_exit_layer 20 --conf_threshold 0.2 \
  --max_batch_size "$BS" --num_requests 100 \
  --num_ee_threshold "$THR" \
  --kv_method copy --collect_conf false \
  --csv_path "$OUT/"

echo "--- result CSV ---"
ls -la "$OUT"/req_*.csv 2>/dev/null || echo "NO CSV — run failed (run_ee.py swallows errors; see traceback above)"
echo "done: $(date)"
