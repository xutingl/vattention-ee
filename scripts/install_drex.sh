#!/usr/bin/env bash
#
# DREX installation script for an NVIDIA B200 (Blackwell, sm_100) node.
#
# This is the Blackwell-era replacement for scripts/install_vattn_old.sh (which
# targeted an A100 with conda + CUDA 12.1 + torch 2.3 + flash-attn 2.5.9 -- none
# of which support sm_100). DREX uses a uv venv + a local CUDA 12.8 toolkit and
# installs everything onto the /vast network filesystem so it survives node
# restarts (see the README "Compute-node restarts" section).
#
# Usage:
#     cd /vast/projects/liuv/pennnetworks/xutingl/vattention-ee
#     bash scripts/install_drex.sh
#
# Re-running is safe-ish but intended for a fresh setup. Heavy steps (CUDA
# toolkit download, extension builds) are skipped if their outputs already exist.

set -euo pipefail

DREX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DREX_ROOT"
echo "[install] DREX_ROOT=$DREX_ROOT"

PY=.venv/bin/python
CUDA_DIR="$DREX_ROOT/cuda-12.8"
CUDA_RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run"

# uv cache on the project filesystem (home quota is small; keeps multi-GB CUDA
# wheels off $HOME and on the same filesystem as the venv so hardlinks work).
export UV_CACHE_DIR="$DREX_ROOT/.uv_cache"

# ----------------------------------------------------------------------------
# 0. Clean the uv cache before first use (removes any stale/cross-machine wheels)
# ----------------------------------------------------------------------------
uv cache clean || true

# ----------------------------------------------------------------------------
# 1. Create the virtual environment (Python 3.12)
# ----------------------------------------------------------------------------
if [ ! -x "$PY" ]; then
    uv venv --python 3.12 .venv
fi

# ----------------------------------------------------------------------------
# 2. PyTorch 2.8.0 + CUDA 12.8 (the first torch with Blackwell / sm_100 support)
# ----------------------------------------------------------------------------
uv pip install --python "$PY" torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# ----------------------------------------------------------------------------
# 3. Local CUDA 12.8 toolkit (nvcc). No system CUDA is installed on this node and
#    nvcc 12.8+ is required to compile kernels for sm_100. Installed (no root) to
#    a local prefix under the repo so it persists across node restarts.
# ----------------------------------------------------------------------------
if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
    if [ ! -f cuda_12.8.1_linux.run ]; then
        wget -O cuda_12.8.1_linux.run "$CUDA_RUNFILE_URL"
    fi
    sh cuda_12.8.1_linux.run --silent --toolkit \
        --toolkitpath="$CUDA_DIR" --no-man-page --override
fi

# ----------------------------------------------------------------------------
# 4. Python dependencies (the serving stack + benchmark/eval libs).
#    NOTE: we deliberately do NOT `pip install -r sarathi-lean/requirements.txt`:
#    it pins flash-attn==2.5.9.post1 and torch>=2.3.0, neither Blackwell-capable.
#    numpy is pinned <2 (pre-numpy-2 research code); transformers <4.50.
# ----------------------------------------------------------------------------
uv pip install --python "$PY" \
    setuptools wheel ninja packaging psutil "ray>=2.5.1" pandas pyarrow \
    sentencepiece "numpy<2" "transformers>=4.37.0,<4.50" matplotlib plotly_express \
    seaborn wandb kaleido ddsketch jupyterlab pillow tiktoken grpcio fastapi \
    uvicorn openai datasets bert_score rouge_score pyyaml tqdm einops

# ----------------------------------------------------------------------------
# 5. Build the CUDA extensions for sm_100. drex_env.sh sets CUDA_HOME / LIBTORCH_PATH
#    / TORCH_CUDA_ARCH_LIST=10.0. --no-build-isolation lets the build see the venv's
#    torch; --no-deps avoids re-pulling the stale requirements.txt pins.
# ----------------------------------------------------------------------------
# shellcheck source=/dev/null
source "$DREX_ROOT/scripts/drex_env.sh"

# sarathi-lean: pos_encoding / layernorm / activation / cache kernels (editable OK)
uv pip install --python "$PY" --no-build-isolation --no-deps -e ./sarathi-lean

# vAttention: install NON-editable so the compiled vattention*.so lands in
# site-packages. An editable install is shadowed by the bare ./vattention dir
# (a namespace package with the same name), so `import vattention` would import
# an empty namespace instead of the extension.
uv pip install --python "$PY" --no-build-isolation --no-deps ./vattention

# ----------------------------------------------------------------------------
# 6. flash-attn (Blackwell). Use the official prebuilt wheel matching
#    torch 2.8 / cu12 / cp312 / cxx11abiTRUE -- it includes sm_100 SASS and
#    runs on the B200 (verified), avoiding a ~30 min source build.
# ----------------------------------------------------------------------------
FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install --python "$PY" --no-deps "$FA_WHL"

# ----------------------------------------------------------------------------
# 7. flashinfer (Blackwell). The 0.6.x line supports sm_100 and keeps the legacy
#    wrapper symbols the attention backends import.
# ----------------------------------------------------------------------------
uv pip install --python "$PY" flashinfer-python

echo
echo "[install] Done. Verify with:"
echo "    source scripts/drex_env.sh"
echo "    CUDA_VISIBLE_DEVICES=0 RAY_ADDRESS=local ray stop --force"
echo "    CUDA_VISIBLE_DEVICES=0 RAY_ADDRESS=local python scripts/run_ee.py \\"
echo "        --ee_policy=rebatching --max_batch_size=4 --num_requests=20 \\"
echo "        --shallow_exit_layer=32 --conf_threshold=0.6"
