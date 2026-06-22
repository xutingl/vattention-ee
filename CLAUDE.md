# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DREX: Dynamic Rebatching for Efficient Early-Exit LLM Inference.** DREX serves
LLMs with **early exit (EE)** — a sequence can stop after a shallow transformer
layer once a confidence threshold is met — combined with a **rebatching scheduler**
that regroups the surviving (non-exited) sequences so the deep layers keep running
on dense batches. The goal is to recover the GPU efficiency that naive early exit
loses when deep-layer batches become small.

It is built on two components:
- **vAttention** (`vattention/`) — CUDA virtual-memory KV-cache manager: contiguous
  virtual KV-cache with on-demand physical pages, so no attention-kernel changes are
  needed. Paper: https://arxiv.org/abs/2405.04437
- **Sarathi-Serve** (`sarathi-lean/`) — chunked-prefill LLM scheduler, forked and
  extended with the EE + rebatching logic.

This checkout is the **Blackwell (B200) port** of `vattention-ee`. See
[README.md](README.md) for the full setup story and [README_old.md](README_old.md)
for the original upstream docs.

## Environment

- **GPU**: NVIDIA **B200** (Blackwell, **sm_100**) on shared `dgx-b200` Slurm nodes. Use one free GPU (`CUDA_VISIBLE_DEVICES`); the node is shared.
- **Python env**: a **uv venv** at `.venv/` (NOT conda). Python 3.12.
- **Stack**: torch 2.8.0+cu128, flash-attn 2.8.3.post1, flashinfer 0.6.12, a local CUDA 12.8 toolkit at `cuda-12.8/` (there is no system CUDA), transformers 4.49, ray 2.55.
- **Always load the env first**: `source scripts/drex_env.sh` (sets the venv on PATH, `CUDA_HOME`, `LIBTORCH_PATH`, `LD_LIBRARY_PATH`, `UV_CACHE_DIR`, `TORCH_CUDA_ARCH_LIST=10.0`).
- Everything lives on `/vast` and survives node restarts; after a restart you only re-source `drex_env.sh` (see README "Compute-node restarts").

## Build / install commands

Full reinstall on a fresh node: `bash scripts/install_drex.sh`. Individual pieces:

```bash
source scripts/drex_env.sh

# Python deps go through uv (cache is on /vast via UV_CACHE_DIR). Home quota is small.
uv pip install --python .venv/bin/python <pkg>

# Rebuild sarathi-lean CUDA kernels (pos_encoding/layernorm/activation/cache).
# --no-build-isolation so the build sees the venv torch; --no-deps to skip the
# stale requirements.txt pins (flash-attn 2.5.9 / torch 2.3 are NOT Blackwell-safe).
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e ./sarathi-lean

# Rebuild the vAttention extension. Install NON-editable: an editable install is
# shadowed by the bare ./vattention namespace dir, so import vattention would load
# an empty namespace instead of the compiled .so.
uv pip install --python .venv/bin/python --no-build-isolation --no-deps ./vattention
```

CUDA extensions are built for sm_100 only (`TORCH_CUDA_ARCH_LIST=10.0`). nvcc comes
from the local `cuda-12.8/` toolkit via `CUDA_HOME`.

## Running experiments

**Two ways to run, depending on where your shell is:**
- **Directly on a compute node (current setup).** When your shell is already on a
  `dgx-b200` node with a GPU (e.g. an interactive Slurm allocation / the current
  session), run `python scripts/run_ee.py ...` directly — no `sbatch` needed.
- **From a login node → must use `sbatch`.** Login nodes have no GPU and kill
  heavy/model-loading jobs. Submit the run as a batch job (`sbatch`) so it lands on
  a compute node; do not run model loads or benchmarks directly on the login node.

Entry point: `sarathi-lean/sarathi/benchmark/main.py`, wrapped by `scripts/run_ee.py`.

```bash
source scripts/drex_env.sh
export CUDA_VISIBLE_DEVICES=0 RAY_ADDRESS=local
ray stop --force            # clear stale /tmp/ray sessions before each run

# Validated early-exit rebatching example (fa_vattn backend, Llama-2-13B-chat):
python scripts/run_ee.py \
  --ee_policy rebatching --max_batch_size 4 --num_requests 20 \
  --shallow_exit_layer 32 --conf_threshold 0.6 --kv_method copy
```

Key `run_ee.py` flags: `--ee_policy` (off/rebatching), `--shallow_exit_layer`,
`--conf_threshold`, `--num_ee_threshold`, `--kv_method` (copy), `--buffer_age_factor`,
`--qps`, `--model`, `--enable_profiling`. Results: CSV summary under `outputs/`,
full metrics/traces under `experiments/e2e_dynamic_eval/`.

Performance/logging toggles (set per-run at launch — no source edits; `run_ee.py`
forwards them to the benchmark + Ray workers as env vars):
- `--collect_conf true|false` (default `true`). Per-step confidence scores are
  computed every decode step and copied to the host (`.tolist()`/`.item()`) only to
  log the "varying conf" experiments. Pass `--collect_conf false` on pure
  **throughput** runs to drop those host syncs (sets `DREX_COLLECT_CONF=0`). The
  conf-summary columns are then meaningless (seeded to 0), which is expected.
- `--ee_profile` (default off). Records per-step buffer/copy/kvcache timing into
  in-memory lists (sets `DREX_EE_PROFILE=1`); leave off for real runs — the lists
  are unbounded and only feed debug prints. Env vars also work directly, e.g.
  `DREX_COLLECT_CONF=0 python scripts/run_ee.py ...`.

`run_ee.py` forces the `fa_vattn_2mb` backend (→ `fa_vattn_megacache`). It also
swallows subprocess failures (`try/except: continue`), so its exit code is not a
reliable pass/fail signal — read the `main.py` traceback / the "Replica 0 exiting"
line instead.

## Architecture

### Two main components

1. **`vattention/`** — C++/CUDA extension (`vattention.cu`). `vAttentionCachingAllocator`
   for virtual↔physical KV-cache mapping. Python APIs (pybind11): `init_kvcache()`,
   `step()`/`step_async()`, `alloc_new_batch_idx()`, `free_batch_idx()`,
   `reserve_physical_pages()`, `num_free_kvblocks()`. Built into site-packages as
   `vattention.cpython-312-*.so`.

2. **`sarathi-lean/sarathi/`** — the serving engine:
   - **`engine/base_llm_engine.py`** — orchestrator: sequence lifecycle, scheduling, EE logic (`ee_policy`, `shallow_exit_layer`, `conf_threshold`).
   - **`core/scheduler/`** — `rebatching_scheduler.py` is the DREX EE-aware scheduler (partial exits, rebatching buffers). Also `vllm_scheduler.py`, `sarathi_scheduler.py`, `orca_scheduler.py`.
   - **`worker/cache_engine/`** — `vATTN_cache_engine.py` (vAttention KV allocations, seq→batch-idx mapping, async/sync alloc; `megacache` = per-tensor multi-layer KV). `vLLM_cache_engine.py` for the paged path.
   - **`model_executor/models/llama.py`** — primary EE model (~58 KB). Configurable shallow exit layer, confidence-based exit, `HiddenStatesBuffer` (the rebatching buffer; width must equal `config.hidden_size`), KV copy for exited requests. Also `qwen.py`, `mistral.py`, `falcon.py`, `yi.py`.
   - **`model_executor/attention/`** — pluggable backends, registry in `__init__.py`. The validated one is `FA_VATTN_MEGACACHE` (FlashAttention + vAttention). `FA*` use flash-attn; `FI*` use flashinfer; `FA_POD*` need the unbuilt `pod_attn` module (expect an import warning).
   - **`benchmark/`** — `main.py` entry point, `config/default.yml` defaults.
   - **`config.py`** — all config dataclasses.

### Supporting

- **`nvidia-vattn-uvm-driver/`** — custom UVM driver for sub-2MB pages (64/128/256 KB). Not needed for the 2 MB page path used here.
- **`scripts/utils.py`** — model registry (HF record / TP degree / log name), path + block/page-size helpers. `llama-2-13b` points at the local weights.

### EE data flow
1. Requests enter via `base_llm_engine.py`, scheduled by `rebatching_scheduler.py`.
2. In `llama.py`, sequences may exit at `shallow_exit_layer` if confidence ≥ `conf_threshold`.
3. Exited sequences go to the rebatching buffer (`deep_buffer`); the rest continue through deep layers. The scheduler flushes the buffer (empty schedule) when it fills or starves.
4. KV for exited sequences is filled via `vATTN_cache_engine.py` (`kv_method=copy`).

## Configuration

Benchmark params live in `sarathi-lean/sarathi/benchmark/config/default.yml`; CLI
flags override via flattened names (`--model_name`, `--model_attention_backend`,
`--replica_scheduler_provider`, `--ee_policy`, …). Attention backend strings encode
backend + page size: `fa_vattn_2mb`, `fa_vattn_256kb`, `fi_paged_16`, `fa_paged_256`.

Default `load_format` is `dummy` (in-process weight init — a throughput/plumbing
run, no checkpoint read). Set `--model_load_format auto` for real weights.

## Key conventions

- vAttention backends use a `_sync` suffix for synchronous allocation; without it, async.
- vAttention uses large pages (64 KB–2 MB); PagedAttention uses token-count block sizes (16, 256).
- `kv_method` controls how post-exit missing KV is filled (currently `copy`).
- Per-run, set `CUDA_VISIBLE_DEVICES` to one free GPU and `RAY_ADDRESS=local` + `ray stop --force` (shared node + stale-Ray hygiene).

## Gotchas (B200 port)

- **Ray GPU detection**: importing the serving stack initializes CUDA before `ray.init`, which makes Ray autodetect 0 GPUs. `benchmark_runner.py` passes `num_gpus` explicitly; keep that.
- **vattention must be installed non-editable** (namespace-dir shadowing, above).
- **Don't `pip install -r sarathi-lean/requirements.txt`** — its flash-attn/torch pins are pre-Blackwell. Use `install_drex.sh`.
- **`fi_vattn` backend is currently broken** at KV-cache allocation (vAttention page-alignment), independent of the flashinfer port. Use `fa_vattn`.
- `run_ee.py` hides subprocess errors; judge success from `main.py` output, not its exit code.
