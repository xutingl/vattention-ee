# DREX / FLExit: Efficient Early-Exit LLM Inference with Dynamic Rebatching

> **DREX** and **FLExit** are the same system — the codebase uses the name `DREX`;
> the paper names it `FLExit`.

DREX is an LLM serving system that makes **early-exit (EE)** inference compatible
with the batching and KV-cache optimizations production serving depends on.

With EE, an "easy" token skips the deeper transformer layers once a shallow **exit
ramp** is confident enough. In a batched server, requests rarely want to exit at
the same point, so existing systems force a **batch-wide** decision — either making
confident tokens stay (*involuntary stays* → wasted compute) or forcing unconfident
tokens out (*involuntary exits* → corrupted output). DREX instead lets **each
request decide independently** and resolves the resulting batch fragmentation with
**Dynamic Rebatching**: exited requests leave the pipeline immediately, while the
non-exited ones are held in a per-model buffer and regrouped into a dense batch for
the deep layers. Two mechanisms keep this profitable:

- **Adaptive Rebatching Threshold (ART)** — only early-exit a split batch when the
  predicted savings exceed the rebatching overhead (`b' > (c/t_d)·b`), re-profiled
  periodically from measured iteration times.
- **SLA-aware forced flushing** — flush the buffer early as a buffered request's
  deadline approaches, so rebatching never starves latency-sensitive requests.

The KV-cache reorganization is **copy-free**: which cache entry belongs to which
request is selected/reordered through virtual tensor indexing (FlashAttention's
`cache_batch_idx` + [vAttention](https://arxiv.org/abs/2405.04437) virtual memory)
instead of moving data; the buffer only holds each left-behind request's
exit-layer hidden state. When buffered requests finally traverse the deep layers,
DREX fills the **missing KV cache** for their skipped layers (`kv_method`: `copy` =
state-copy from the last computed layer, `postfill` = recompute). It is built on
[Sarathi-Serve](https://github.com/microsoft/sarathi-serve) (chunked-prefill
scheduler), forked here as `sarathi-lean`.

**Paper:** *Efficient Early-Exit Inference with Dynamic Rebatching* (SOSP
submission) — LaTeX sources and PDF in
[`../EarlyExit_SOSP_git/`](../EarlyExit_SOSP_git/)
([`main.pdf`](../EarlyExit_SOSP_git/main.pdf)). Reported result: eliminates
involuntary exits entirely while improving throughput by up to **34%** over EE
baselines.

> This repository is the Blackwell (B200) port of the `vattention-ee` codebase.
> For the original upstream documentation, see [`README_old.md`](README_old.md).

---

## Hardware & environment

This checkout is installed and validated on:

| | |
|---|---|
| Node | `dgx001` (shared `dgx-b200` Slurm partition) |
| GPU | **NVIDIA B200** (Blackwell, compute capability **sm_100**), 183 GB |
| Driver | 580.95.05 (CUDA 13.0 capable) |
| OS | Ubuntu 24.04, gcc 13.3 |

### Installed stack

| Component | Version | Notes |
|---|---|---|
| Python | 3.12.3 | uv venv at `.venv/` |
| PyTorch | **2.8.0+cu128** | first torch with sm_100 support |
| CUDA toolkit (nvcc) | **12.8.93** | local prefix `cuda-12.8/` (no system CUDA on this node) |
| flash-attn | **2.8.3.post1** | official prebuilt Blackwell wheel, B200-verified |
| flashinfer | **0.6.12** | `flashinfer-python` |
| transformers | 4.49.0 | |
| ray | 2.55.1 | |
| numpy | 1.26.4 | pinned `<2` (pre-numpy-2 code) |

Everything (`.venv/`, `cuda-12.8/`, the model weights, the uv cache) lives on the
`/vast` network filesystem, so a node restart does **not** require reinstalling
anything — see [Compute-node restarts](#compute-node-restarts).

---

## Quickstart

From the repo root (`/vast/projects/liuv/pennnetworks/xutingl/vattention-ee`):

```bash
# 1. Load the environment (venv + local CUDA toolkit + library paths).
source scripts/drex_env.sh

# 2. Pick a free GPU on the shared node and clear any stale Ray state.
export CUDA_VISIBLE_DEVICES=0
export RAY_ADDRESS=local
ray stop --force          # see "Troubleshooting" for why

# 3. Run the rebatching early-exit validation example.
python scripts/run_ee.py \
    --ee_policy=rebatching \
    --max_batch_size=4 \
    --num_requests=20 \
    --shallow_exit_layer=32 \
    --conf_threshold=0.6
```

A successful run processes all 20 requests and writes a metrics CSV to
`outputs/req_20_batch_4_layer_32_conf_0.6_rebatching_copy.csv`, ending with a line
like:

```
Replica 0 exiting after processing 20 (605 iterations), Total time taken: ~14 s
Exited rates(#tokens generated via ee vs. non-ee): [401, 1771]
```

The default model is **Llama-2-13B-chat** (40 layers, hidden size 5120), loaded
from `/vast/projects/liuv/pennnetworks/hf_models/Llama-2-13b-chat-hf`. The
benchmark uses `load_format: dummy` by default (weights are initialized in-process,
no checkpoint read needed for a throughput/plumbing test). For real-weight
inference, pass `--model_load_format auto` through to the benchmark.

---

## Installation from scratch (new node)

If you need to rebuild on a fresh node, the whole process is scripted:

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vattention-ee
bash scripts/install_drex.sh
```

This: cleans the uv cache, creates the `.venv`, installs torch 2.8+cu128, downloads
and installs the CUDA 12.8 toolkit to `cuda-12.8/`, installs the Python deps,
builds the sarathi-lean and vAttention CUDA extensions for sm_100, and installs the
Blackwell flash-attn wheel + flashinfer. See the script for the exact steps and the
rationale behind each (especially why we do **not** use
`sarathi-lean/requirements.txt` directly — its flash-attn/torch pins predate
Blackwell).

---

## Compute-node restarts

This is a **shared, restartable compute node**. Because all build artifacts live on
the `/vast` network filesystem, a restart wipes only node-local, in-memory state —
not your installation.

**After a node restart you do NOT need to reinstall anything.** The `.venv/`, the
`cuda-12.8/` toolkit, the compiled CUDA extensions (the `.so` files in the venv),
and the model weights are all persisted on `/vast`.

What you DO need to do in each new shell / after a restart:

1. **Re-source the environment** (env vars do not persist across shells):
   ```bash
   source scripts/drex_env.sh
   ```
2. **Confirm you have a GPU.** You may land on a different B200 node or need a fresh
   Slurm allocation. Check `nvidia-smi` shows a B200 and pick a free index for
   `CUDA_VISIBLE_DEVICES`.
3. **Clear stale Ray state before the first run:**
   ```bash
   export RAY_ADDRESS=local
   ray stop --force
   ```
   (`/tmp/ray` sessions can linger and confuse `ray.init`; this is cheap insurance.)

You would only need to **reinstall** if: the `/vast` checkout is deleted, the node
is reimaged with a GPU driver too old for CUDA 12.8 (needs ≥ ~525; this node has
580), or you change the Python/torch/CUDA versions (which would require rebuilding
the CUDA extensions). In those cases, re-run `scripts/install_drex.sh`.

---

## Running experiments

`scripts/run_ee.py` is the early-exit benchmark driver; it wraps the Sarathi
benchmark entry point (`sarathi-lean/sarathi/benchmark/main.py`). Key flags:

| Flag | Meaning |
|---|---|
| `--ee_policy` | `off` (baseline) or `rebatching` (DREX) |
| `--shallow_exit_layer` | layer at which sequences may early-exit (Llama-2-13B has 40) |
| `--conf_threshold` | confidence threshold to trigger an early exit |
| `--num_ee_threshold` | min #requests wanting to EE before a partial exit fires (`-1` = off) |
| `--max_batch_size` | scheduler max batch size |
| `--num_requests` | number of requests to replay |
| `--kv_method` | how missing KV entries are filled after an exit (`copy`) |
| `--buffer_age_factor` | aging factor for the rebatching buffer |
| `--model` | model key from `scripts/utils.py` (default `llama-2-13b`) |

Results (CSV metrics, Chrome traces, JSON request logs) land under
`experiments/e2e_dynamic_eval/` and the CSV summary under `outputs/`.

Other drivers: `scripts/benchmark_e2e_static_trace.py`,
`scripts/benchmark_e2e_dynamic_trace.py` (both accept `--test`).

---

## Experiments

New experiments live under [`experiments/`](experiments/), organized as
`experiments/<experiment_name>/<environment_model>/` (e.g.
`experiments/sanity_check/b200_llama_13b/`). Each instance is self-contained — a README,
a run script (`run.sh`, launched on a compute node with `nohup`; or `run.sbatch` from a
login node), and a `results/` dir. See [`experiments/README.md`](experiments/README.md)
for the conventions and how to add one;
[`experiments/sanity_check/`](experiments/sanity_check/) is the reference example (decode
throughput for every EE policy + the non-EE baseline on Llama-2-13B).

The `outputs*/` dirs and the older `experiments/e2e_dynamic_eval/` are pre-existing
results that predate this convention.

---

## What changed for the B200 port

The original code targeted an A100 with CUDA 12.1 / torch 2.3 / flash-attn 2.5.9 —
none of which support Blackwell. Beyond the new dependency stack, these source
changes were required (all derived from the actual failures hit while bringing the
validation run up):

- **`scripts/utils.py`** — `llama-2-13b` now points at the local weights
  `/vast/.../hf_models/Llama-2-13b-chat-hf`; the `--csv_path` default is derived
  from the repo root instead of the hard-coded `/workspace/...`.
- **`scripts/drex_env.sh`** *(new)* — single source of truth for the environment
  (venv, `CUDA_HOME`, `LIBTORCH_PATH`, `LD_LIBRARY_PATH`, `UV_CACHE_DIR`,
  `TORCH_CUDA_ARCH_LIST=10.0`).
- **`sarathi-lean/.../benchmark_runner.py`** — `ray.init` now passes `num_gpus`
  explicitly. Importing the serving stack initializes CUDA before `ray.init`, which
  breaks Ray's GPU autodetection (it registered 0 GPUs → `KeyError: 'GPU'`). Also
  added a short poll for the GPU resource to avoid a registration race.
- **`sarathi-lean/.../models/llama.py`** — the `HiddenStatesBuffer` width was
  hard-coded to `8192` (Llama-2-70B); it now derives from `config.hidden_size`
  (5120 for Llama-2-13B), fixing a shape-mismatch crash in the rebatching path.
- **`sarathi-lean/.../config/default.yml`** — stale `/workspace/...` `csv_path`
  replaced with a repo-relative path.

---

## Known issues / limitations

- **Validated path:** `--ee_policy=rebatching` with the `fa_vattn` (FlashAttention +
  vAttention megacache) backend, which is what `run_ee.py` uses. This is the
  end-to-end-verified configuration on the B200.
- **`fi_vattn` (FlashInfer + vAttention) backend** currently fails during KV-cache
  allocation with `size_bytes is not a multiple of page_size * shape[0]` — a
  vAttention page-alignment constraint in the non-megacache layout. This is
  independent of the flash-attn/flashinfer port (the FlashInfer APIs import and
  resolve fine); it is a pre-existing vAttention geometry issue and has not been
  chased down. Use the `fa_vattn` backends.
- The `pod_attn` import warning at startup is expected — the optional POD-Attention
  module is not built here and the `FA_POD*` backends are unused by the validation.

---

## DREX/FLExit implementation map

The EE + Dynamic Rebatching logic that *defines* DREX (i.e. what was added on top
of stock Sarathi-Serve / vAttention) lives in these files. All paths are under
`sarathi-lean/sarathi/` unless noted.

| File | DREX/FLExit role (paper section) |
|---|---|
| `model_executor/models/ee_utils.py` | Shared EE utilities (~the paper's "early-exit utilities"): `HiddenStatesBuffer` (the left-behind rebatching buffer), `softmax_confidence`, `get_skip_mask` (per-request EE decision → `EEMask`), `get_adaptive_rebatching_threshold` (ART, §To-EE-or-Not), and the missing-KV-fill helper. |
| `model_executor/models/llama.py`, `qwen.py` | The EE models (Llama-EE / Qwen-EE, Apparate-style ramps). EE forward path: exit-ramp logits, all-exit vs. split handling, buffer add/take, deep-layer rebatch merge, and `kv_method` dispatch. |
| `core/scheduler/vllm_scheduler.py` | EE-aware scheduler: vLLM continuous batching + the `rebatching_buffer`; `on_rebatching()` moves sequences between running/buffer; buffer-flush conditions by size / age / SLA (the `α` term = `--buffer_age_factor`, §Flushing the Buffer). |
| `engine/base_llm_engine.py` | Engine step loop: drives EE iterations, reconciles (re-batched) outputs via `on_rebatching`, and re-profiles the ART (`rebatching_ee_factor`) periodically from measured full/shallow/deep iteration times. |
| `worker/cache_engine/vATTN_cache_engine.py` | vAttention KV management: seq→batch-idx mapping (`cache_batch_idx` — the copy-free reorder), `step()`, and `copy_kv_cache_starting_at_layer` (missing-KV state-copy for skipped layers, §Filling In the Missing State). |
| `worker/base_worker.py`, `model_executor/model_runner.py` | Wire `execute_model` to the EE forward and reconcile the possibly reordered/rebatched output sequences back with the scheduler. |
| `model_executor/layers/sampler.py` | Samples from the exit-ramp logits on EE steps; computes the softmax confidence score reported as a quality metric. |
| `benchmark/benchmark_runner.py`, `benchmark/main.py`, `scripts/run_ee.py` | Benchmark driver + metrics (throughput, RCT, BERT score, EE proportion, involuntary stays/exits) and the `run_ee.py` launcher. |

## Repository layout

```
vattention/        vAttention CUDA extension (vattention.cu): virtual<->physical
                   KV-cache mapping via CUDA VMM. Built into site-packages.
sarathi-lean/      Forked Sarathi-Serve serving engine + benchmark suite.
  sarathi/engine/              orchestration, EE logic
  sarathi/core/scheduler/      rebatching_scheduler.py (the DREX scheduler)
  sarathi/worker/cache_engine/ vATTN_cache_engine.py (vAttention KV management)
  sarathi/model_executor/      models (llama.py = primary EE model) + attention backends
  sarathi/benchmark/           main.py entry point, config/default.yml
scripts/           run_ee.py (EE driver), utils.py (model registry), drex_env.sh,
                   install_drex.sh, install_vattn_old.sh (legacy A100 reference)
nvidia-vattn-uvm-driver/  custom UVM driver for sub-2MB pages (not needed for 2MB)
outputs/, experiments/    results
```

See [`CLAUDE.md`](CLAUDE.md) for a deeper architecture map and build/run guidance.
