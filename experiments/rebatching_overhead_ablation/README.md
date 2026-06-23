# Experiment: `rebatching_overhead_ablation`

**Question:** do the two cheapest overhead-reduction levers — (a) stronger **inline
draining** and (b) raised **`min_flush_size`** — recover the throughput rebatching loses
to flush overhead?

Motivation (from [`num_ee_threshold_sweep`](../num_ee_threshold_sweep/)): on 13B, rebatching
never beats the no-EE baseline, and at **batch 16 with `num_ee_threshold=2`** doing real EE
*costs* −7.2% — because each partial exit fragments the batch into extra deep-flush
iterations. The flush machinery exists but is governed by **hardcoded constants**:

- **(a) inline draining** — [llama.py](../../sarathi-lean/sarathi/model_executor/models/llama.py)
  drains buffered survivors onto any in-flight deep pass only when `len(deep_buffer) >= 2`.
  Lowering that threshold drains more eagerly → fewer *dedicated* flush iterations.
- **(b) `min_flush_size`** — [vllm_scheduler.py](../../sarathi-lean/sarathi/core/scheduler/vllm_scheduler.py)
  triggers a dedicated flush at `len(buffer) >= 8`. Raising it waits for a denser flush →
  fewer, better-amortized flush iterations (effective range [1, 16] at batch 16).

## Design

Fixed at the **worst-case regime** so the levers have something to fix: 13B, layer 20,
conf 0.2, `num_requests=100`, kv copy, **real weights**, **batch 16**, **`num_ee_threshold=2`**
(ee_rate ≈ 0.18 → heavy flushing). Each lever is exposed via an env var (default-preserving)
and swept; cells isolate (a), (b), their direction, and the combination.

| cell | inline_drain_min | min_flush_size | isolates |
|---|---|---|---|
| `off_b16` | – | – | no-EE baseline (target throughput) |
| `base` | 2 | 8 | current defaults (the ≈ −7% point) |
| `a_inline1` | **1** | 8 | (a) stronger inline draining |
| `b_flush4` | 2 | **4** | (b) lower (more flushes — direction check) |
| `b_flush12` | 2 | **12** | (b) raise |
| `b_flush16` | 2 | **16** | (b) raise to batch size |
| `ab_combined` | **1** | **16** | (a)+(b) together |

7 cells, ~15 min on 13B.

## Reading it

`summarize.py` reports per cell: `decode_tok/s`, **`vs_off%`** (vs no-EE — goal is to reach
0/positive), **`vs_base%`** (vs current defaults — **>0 means the lever helped**), plus
`ee_rate`, `ee_iter`, and `avg_deep_iter_ms` (the per-flush cost). Expected reading:

- If (a)/(b) work, `a_inline1` / `b_flush12` / `b_flush16` show `vs_base% > 0` (recovering
  toward `off`), and `b_flush4` is *worse* (confirming the flush-frequency mechanism).
- If nothing moves, the overhead isn't flush-frequency but the per-exit split cost itself
  (KV copy / doubled fixed iteration cost) → points to the CUDA-graph/fusion levers instead.

Follow-ups if promising: extend to `num_ee_threshold=0` (heaviest EE) and to batch 8, and
then 70B.

## Instances

- [`b200_llama_13b/`](b200_llama_13b/) — NVIDIA B200, Llama-2-13B-chat. **Requires the
  engine env hooks** — see its README "Execution checklist".
