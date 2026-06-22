#!/usr/bin/env bash
# DREX environment setup.
#
# Source this file before building or running DREX:
#     source scripts/drex_env.sh
#
# Everything DREX needs lives on the /vast network filesystem (the .venv,
# the local CUDA toolkit, the model weights), so it all survives a compute-node
# restart. After a restart you do NOT need to reinstall anything -- you only need
# to re-source this file in a fresh shell (it just sets environment variables).
# See the "Compute-node restarts" section of the README.

# Resolve the repository root from this script's location (portable across machines).
_DREX_ENV_SRC="${BASH_SOURCE[0]:-$0}"
export DREX_ROOT="$(cd "$(dirname "$_DREX_ENV_SRC")/.." && pwd)"

# --- Python virtual environment (uv venv) ---
export VIRTUAL_ENV="$DREX_ROOT/.venv"

# --- Local CUDA 12.8 toolkit (nvcc for Blackwell / sm_100) ---
export CUDA_HOME="$DREX_ROOT/cuda-12.8"

# --- libtorch shipped inside the venv's torch (used to build the vAttention ext) ---
export LIBTORCH_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch"

# --- PATH / library search ---
export PATH="$CUDA_HOME/bin:$VIRTUAL_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LIBTORCH_PATH/lib:${LD_LIBRARY_PATH:-}"

# --- uv cache on /vast/projects (home quota is small; keeps big CUDA wheels off $HOME) ---
export UV_CACHE_DIR="$DREX_ROOT/.uv_cache"

# Target only the B200 when building CUDA extensions from source (flash-attn etc.).
export TORCH_CUDA_ARCH_LIST="10.0"

echo "[drex_env] DREX_ROOT=$DREX_ROOT"
echo "[drex_env] python=$(command -v python)  cuda=$(command -v nvcc)"
