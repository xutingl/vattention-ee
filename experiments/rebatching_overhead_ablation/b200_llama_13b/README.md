# `rebatching_overhead_ablation` on `b200_llama_13b`

Instance of [`rebatching_overhead_ablation`](../README.md). See the parent for the design
and how to read results. Environment: NVIDIA B200, Llama-2-13B-chat, real weights, batch 16.

## Execution checklist (run AFTER the 70B sweep — do not edit engine while it runs)

The two levers are hardcoded constants; expose them via env vars (**default-preserving**,
so normal runs and the 70B sweep are unaffected). Both processes (scheduler in the engine,
forward in the Ray worker) inherit these via `os.environ`, same as the existing
`DREX_COLLECT_CONF`.

**1. Inline draining** — `sarathi/model_executor/models/llama.py`
- In the model `__init__` (near where `self.num_ee_threshold` is set, ~L423), add:
  ```python
  self.inline_drain_min = int(os.environ.get("DREX_INLINE_DRAIN_MIN", "2"))
  ```
- At the piggyback merge (~L1010), change `if len(self.deep_buffer) >= 2:` to:
  ```python
  if len(self.deep_buffer) >= self.inline_drain_min:
  ```
  (confirm `import os` exists in the file).

**2. min_flush_size** — `sarathi/core/scheduler/vllm_scheduler.py` (~L45)
- Change `self.min_flush_size = 8` to:
  ```python
  self.min_flush_size = int(os.environ.get("DREX_MIN_FLUSH_SIZE", "8"))
  ```
  **NOTE:** `vllm_scheduler.py` does **not** import `os` yet — add `import os` at the top.
  (`llama.py` already imports `os`.)

**3. Verify** (1 quick cell, ~2 min): with the hooks in, run `base` (inline=2, flush=8) and
confirm it reproduces the known ≈ −7% vs off — i.e. the defaults are preserved.

**4. Run the sweep:**
```bash
cd experiments/rebatching_overhead_ablation/b200_llama_13b
export CUDA_VISIBLE_DEVICES=0
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

## Results

Per-cell subdirs under `results/` (`off_b16/`, `base/`, `a_inline1/`, `b_flush4/`,
`b_flush12/`, `b_flush16/`, `ab_combined/`), each with its `req_*.csv`. Then:

```bash
python summarize.py --results-dir results   # -> results/summary.txt
```

`run.sh` sets `DREX_INLINE_DRAIN_MIN` / `DREX_MIN_FLUSH_SIZE` per cell and exports
`DREX_RAY_NUM_CPUS=16` (the shared-node Ray-hang fix).
