#!/bin/bash
#SBATCH --job-name=drex-raydiag
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
REPO_ROOT="/vast/projects/liuv/pennnetworks/xutingl/vattention-ee"
source "$REPO_ROOT/scripts/drex_env.sh"
export RAY_ADDRESS=local
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"; mkdir -p "$RAY_TMPDIR"
export RAY_NOSET_CUDA_VISIBLE_DEVICES=1 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
ray stop --force >/dev/null 2>&1 || true
echo "host: $(hostname)  CVD=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=compute_mode,memory.used --format=csv

cat > "$RAY_TMPDIR/raytest.py" <<'PY'
import os
PRE = os.environ.get("PREINIT_CUDA", "1") == "1"
print(f"PREINIT={PRE}  CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}")
if PRE:
    import torch
    print("driver device_count:", torch.cuda.device_count())
import ray
ray.init(num_gpus=1, ignore_reinit_error=True)
print("ray GPU resource:", ray.available_resources().get("GPU"))
@ray.remote(num_gpus=1)
def f():
    import torch, os
    torch.cuda.set_device(0)
    x = torch.ones(8, device="cuda")
    return f"OK worker CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} dev={x.device} {torch.cuda.get_device_name(0)}"
try:
    print("RESULT:", ray.get(f.remote()))
except Exception as e:
    print("WORKER FAILED:", repr(e))
ray.shutdown()
PY

echo "### TEST A: driver PRE-initializes CUDA before ray.init (mirrors DREX)"
PREINIT_CUDA=1 python -u "$RAY_TMPDIR/raytest.py" || true
echo; ray stop --force >/dev/null 2>&1 || true; sleep 2
echo "### TEST B: driver does NOT touch CUDA before ray.init"
PREINIT_CUDA=0 python -u "$RAY_TMPDIR/raytest.py" || true
echo "done: $(date)"
