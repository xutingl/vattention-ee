# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vAttention-EE extends the vAttention memory manager (CUDA virtual memory-based KV-cache) with **Early Exit (EE)** support for LLM serving. It integrates with Sarathi-Serve, an LLM inference scheduler. The core idea: use CUDA virtual memory APIs to decouple virtual/physical memory allocation for KV-cache, enabling contiguous virtual memory with on-demand physical pages — no attention kernel modifications needed.

The EE extension adds a rebatching scheduler that allows sequences to exit early (at a shallow layer) based on confidence thresholds, with KV-cache management for partially-processed batches.

Paper: https://arxiv.org/abs/2512.15705 (EE), https://arxiv.org/abs/2405.04437 (vAttention)

## Environment

- **Conda env**: `vattn` (always activate before any commands)
- **GPU**: A100 80GB
- **Eval models on this machine**: Llama-2-13B (`meta-llama/Llama-2-13b-chat-hf`), Qwen-14B
- **Requirements**: PyTorch 2.3.0, CUDA 12.1, Python 3.10

## Build Commands

```bash
conda activate vattn

# Build vAttention CUDA extension (requires LIBTORCH_PATH env var)
cd vattention && pip install -e . && cd ..

# Build Sarathi-Lean (compiles CUDA kernels for pos_encoding, layernorm, activation, cache ops)
cd sarathi-lean && pip install -e . && cd ..
```

## Running Experiments

The benchmark entry point is `sarathi-lean/sarathi/benchmark/main.py`. Experiment runner scripts in `scripts/` wrap this with appropriate configs.

```bash
# Early exit experiment (main workflow)
python scripts/run_ee.py \
  --ee_policy rebatching \
  --max_batch_size 4 \
  --num_requests 20 \
  --shallow_exit_layer 32 \
  --conf_threshold 0.6 \
  --kv_method copy \
  --csv_path /path/to/output/

# Static trace benchmark (fixed context lengths)
python scripts/benchmark_e2e_static_trace.py [--test]

# Dynamic trace benchmark (Poisson arrivals, arXiv dataset)
python scripts/benchmark_e2e_dynamic_trace.py [--test]
```

Key `run_ee.py` flags: `--ee_policy` (off/rebatching), `--shallow_exit_layer`, `--conf_threshold`, `--num_ee_threshold`, `--kv_method` (copy), `--buffer_age_factor`, `--qps`, `--enable_profiling`.

Results go to `experiments/e2e_dynamic_eval/` or `experiments/e2e_static_eval/` and include CSV metrics, Chrome traces, and JSON request logs.

## Architecture

### Two main components:

1. **`vattention/`** — C++/CUDA extension (`vattention.cu`). Implements `vAttentionCachingAllocator` for virtual-physical memory mapping. Python APIs via pybind11: `init_kvcache()`, `step()`/`step_async()`, `alloc_new_batch_idx()`, `free_batch_idx()`, `reserve_physical_pages()`, `num_free_kvblocks()`.

2. **`sarathi-lean/sarathi/`** — Modified Sarathi-Serve LLM serving system:
   - **`engine/base_llm_engine.py`** — Main orchestrator. Manages sequence lifecycle, scheduling, and EE logic (ee_policy, shallow_exit_layer, conf_threshold).
   - **`core/scheduler/`** — Scheduling strategies. `rebatching_scheduler.py` is the EE-aware scheduler that handles partial batch exits and rebatching buffers. Also: `vllm_scheduler.py`, `sarathi_scheduler.py`, `orca_scheduler.py`.
   - **`worker/cache_engine/`** — `vATTN_cache_engine.py` manages vAttention KV-cache allocations, maps sequence IDs to batch indices, handles async/sync memory allocation. `vLLM_cache_engine.py` for paged approach.
   - **`model_executor/models/llama.py`** — Primary model implementation (~58KB). Contains early exit logic: configurable shallow exit layer, confidence-based exit decisions, KV cache copying for exited requests. Also: `qwen.py`, `mistral.py`, `falcon.py`, `yi.py`.
   - **`model_executor/attention/`** — Pluggable attention backends. Registry in `__init__.py`. Key backends: `FA_VATTN`/`FI_VATTN` (async), `FA_VATTN_SYNC`/`FI_VATTN_SYNC` (sync), `FA_PAGED`/`FI_PAGED` (PagedAttention), `FA3_VATTN` (FlashAttention-3), `FA_VATTN_MEGACACHE` (per-layer KV storage).
   - **`benchmark/`** — Benchmarking suite. `main.py` is the entry point. Config in `config/default.yml`.
   - **`config.py`** — All configuration dataclasses (ModelConfig, CacheConfig, SchedulerConfig, etc.).

### Supporting components:

- **`nvidia-vattn-uvm-driver/`** — Custom NVIDIA UVM driver for sub-2MB page sizes (64KB, 128KB, 256KB).
- **`scripts/utils.py`** — Model registry (HF record, TP degree, log names), path helpers, block/page size extraction from backend names.
- **`sarathi-lean/data/processed_traces/`** — Request trace datasets (arXiv summarization, ShareGPT).

### EE data flow:
1. Requests enter via `base_llm_engine.py` and are scheduled by `rebatching_scheduler.py`
2. During inference in `llama.py`, sequences may exit early at `shallow_exit_layer` if confidence exceeds `conf_threshold`
3. Early-exited sequences go to a rebatching buffer; remaining sequences continue through deep layers
4. KV cache for exited sequences is managed via `vATTN_cache_engine.py` (copy method fills missing KV entries)

## Configuration

All benchmark params are defined in `sarathi-lean/sarathi/benchmark/config/default.yml`. CLI flags override YAML values via flattened naming: `--model_name`, `--model_attention_backend`, `--replica_scheduler_provider`, `--ee_policy`, etc.

Attention backend strings encode both the backend and page size: `fa_vattn_2mb`, `fa_vattn_256kb`, `fi_paged_16`, `fa_paged_256`.

## Key Conventions

- vAttention backends use `_sync` suffix for synchronous memory allocation (e.g., `fa_vattn_2mb_sync`); without suffix means async.
- Block sizes: vAttention uses large pages (64KB–2MB); PagedAttention uses token-count block sizes (e.g., 16, 256).
- Model load format is `dummy` by default in benchmarks (no real weights loaded for throughput testing). Set `--model_load_format auto` for real inference.
- The `kv_method` parameter controls how missing KV entries are filled after early exit; currently `copy` is the primary method.
