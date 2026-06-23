# Overhead-Scaling Experiment (`overhead_scaling`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new DREX experiment, `overhead_scaling/b200_llama_13b`, that measures how the rebatching overhead `c` and the ratio `c/t_d` scale with batch size {8,16,32,64,128} on Llama-2-13B, and decomposes `c` into its control-plane vs data-plane components — to test (and correct) the claim that rebatching's data-plane is copy-free/O(1) and the overhead therefore does not blow up with batch size.

**Architecture:** Per batch size we run four serving configs (no-EE baseline, median-EE baseline, rebatching at a fixed low threshold, rebatching auto-ART) plus one profiling pass. `c`, `t_f`, `t_s`, `t_d`, and decode throughput come from CSV columns that **already exist** (`avg_normal/ee/deep_iter_time`, `decode_time`, `num_output_tokens`). The `c` breakdown comes from per-component timers that **already exist** in the engine but are currently dead (feed only commented-out prints); the only engine change is a small shim that writes those aggregates to a parseable JSON sidecar when `--ee_profile` is on, plus an optional `torch.cuda.synchronize()` mode for accurate GPU-side data-plane timing. A `summarize.py` joins everything into one table.

**Tech Stack:** Python 3.12 / uv venv, PyTorch 2.8+cu128, Ray (local), Sarathi-Serve engine, vAttention KV cache, B200 (sm_100). Runs launched via `scripts/run_ee.py` on a `dgx-b200` compute node (no sbatch — Ray-under-SLURM is broken; see CLAUDE.md).

---

## Part 0 — Accuracy verdict on the stated claim (read before building)

The user's description, checked against the code, is **mostly accurate in spirit but two specifics are overstated.** The experiment is designed to *measure* this rather than assert it.

**What is TRUE:**
- **Regrouping survivors into a deep batch is copy-free virtual indexing.** When buffered survivors are flushed/merged into a deep pass, vAttention does *not* move their KV. `vATTNCacheEngine.step()` writes `curr_seq_lens` and sets an int32 `curr_batch_idx` index tensor ([vATTN_cache_engine.py:248-251](../../../sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L248)); FlashAttention then reads each sequence's KV in place via that `cache_batch_idx` (the vMemMap virtual→physical mapping). No per-request KV gather/scatter to re-form the batch. ✓
- **Control plane = buffer-list bookkeeping + one CPU↔GPU sync per rebatch.** `VLLMScheduler.on_rebatching` ([vllm_scheduler.py:213-247](../../../sarathi-lean/sarathi/core/scheduler/vllm_scheduler.py#L213)) is pure Python list `append`/`remove`/`insert` + `set_status` — per-request pointer ops, off the GPU critical path. The per-rebatch sync is the single `vattention.step(curr_seq_lens)` call ([vATTN_cache_engine.py:241-244](../../../sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L241)), per-operation not per-request. ✓

**What is OVERSTATED ("no hidden-state or KV copies"):**
- **There IS a hidden-state copy.** Survivors that do not early-exit are staged into `HiddenStatesBuffer` (`deep_buffer`): `add_hidden_states` scatters their activations into a buffer tensor ([ee_utils.py:38-50](../../../sarathi-lean/sarathi/model_executor/models/ee_utils.py#L38)), `take_hidden_states` gathers them back ([ee_utils.py:63-85](../../../sarathi-lean/sarathi/model_executor/models/ee_utils.py#L63)), and the inline-drain merge `torch.cat`s them onto the live batch ([llama.py:1019-1021](../../../sarathi-lean/sarathi/model_executor/models/llama.py#L1019)). This is `O(num_survivors × hidden_size)` data movement — small (one activation row per request, ~10 KB at hidden=5120/bf16) but **not zero and it scales with b.**
- **There IS a KV copy, under `kv_method=copy`.** Early-exited tokens never compute deep-layer K/V, so `copy_kv_cache_starting_at_layer` ([vATTN_cache_engine.py:133-189](../../../sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L133)) fills the missing deep layers by broadcasting the last computed layer's K/V into all deeper layers at the exited `(batch_idx, position)` slots. In megacache mode this is a single broadcast scatter (2 kernels), but it touches `O(num_exited × deep_layers × heads × head_dim)` — the largest data-plane term, and it scales with b.

**The corrected, provable statement (what the experiment shows):**
> Rebatching performs *no* KV/hidden-state copying to **regroup** sequences (that is pure virtual re-indexing). The data-plane copies that *do* exist (deep-layer KV fill for exited tokens + hidden-state staging) are small per request and are dominated by the `O(b)` deep-layer forward `t_d` (memory-bandwidth-bound weight streaming). The control plane is `O(1)` per rebatch (one sync) plus cheap per-request pointer bookkeeping off the GPU path. Therefore both numerator pieces of `c` grow no faster than `t_d`, so **`c/t_d` stays bounded as b grows**, and the adaptive threshold `ART = b·(c/t_d)` scales gracefully rather than blowing up.

This is the thesis the table will support or refute with measured numbers.

## Part 0.1 — Does current `--ee_profile` give enough information?

**Almost — every timer needed already exists, but nothing emits them, and there is one accuracy caveat.**

`--ee_profile` sets `DREX_EE_PROFILE=1` ([ee_utils.py:13](../../../sarathi-lean/sarathi/model_executor/models/ee_utils.py#L13), [llama.py:279](../../../sarathi-lean/sarathi/model_executor/models/llama.py#L279)), which already records, per component:

| Component (maps to part of `c`) | Timer that already exists | Plane |
|---|---|---|
| EE head + `get_skip_mask` | `self.ee_overhead_time_lst` ([llama.py:921-922](../../../sarathi-lean/sarathi/model_executor/models/llama.py#L921)) | control |
| Deep-layer KV fill (`kv_method=copy`) | `self.fill_kvcache_time_lst` ([llama.py:663-664](../../../sarathi-lean/sarathi/model_executor/models/llama.py#L663)) | **data** |
| `update_seqs_in_kvcache` (step + begin_forward) | `self.update_kvcache_time_lst` ([llama.py:646-647](../../../sarathi-lean/sarathi/model_executor/models/llama.py#L646)) | control |
| Hidden-state buffer staging in/out | `deep_buffer.time_spent_adding` / `time_spent_taking` ([ee_utils.py:34,50,84](../../../sarathi-lean/sarathi/model_executor/models/ee_utils.py#L34)) | **data** |
| `vattention.step` CPU↔GPU sync | `cache_engine.step_times` ([vATTN_cache_engine.py:253-254](../../../sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L253)) | control (sync) |
| Coarse total of the whole rebatching block | `self.rebatching_time` ([llama.py:1008](../../../sarathi-lean/sarathi/model_executor/models/llama.py#L1008)) | both |

**Two gaps to close (these are the only engine changes):**
1. **Emission.** All of the above feed only commented-out debug prints ([llama.py:832-836,1044](../../../sarathi-lean/sarathi/model_executor/models/llama.py#L832)); none reach the CSV or any parseable output, and they live in the Ray **worker** process (not the benchmark driver), so we cannot read them from `benchmark_runner`. Fix: the model writes a JSON sidecar of the cumulative aggregates when `DREX_EE_PROFILE_OUT` is set (Task 1).
2. **Accuracy.** The timers are plain CPU `perf_counter()` with no `torch.cuda.synchronize()`, so for the **data-plane** ops (KV fill, buffer copies) they capture kernel-*launch* time, not GPU-*execution* time — they under-measure data movement. For the **control-plane** ops (Python bookkeeping, the `vattention.step` sync which itself blocks) they are already meaningful. Fix: an opt-in `DREX_EE_PROFILE_SYNC=1` adds a `synchronize()` around the two data-plane timers, used only in the dedicated profiling pass (Task 2) so the throughput runs stay unperturbed.

`c`, `t_f`, `t_s`, `t_d`, decode throughput, and `ART` need **no** engine change — they come from existing CSV columns + the engine's existing ART log line (Tasks 4-6).

## Part 0.2 — Fixed config and the run matrix

Fixed (consistent with prior DREX experiments): `--model llama-2-13b` (real weights, `--model_load_format auto`), `--shallow_exit_layer 20` (50% of 40 layers), `--conf_threshold 0.2`, `--kv_method copy`, `--num_requests 100`, `--collect_conf false`, `DREX_RAY_NUM_CPUS=16`. Engine defaults are now the `ab_combined` config (inline_drain_min=1, min_flush_size=batch) — that is fine and is what we want to characterize; the experiment does **not** override them except where noted.

Per batch size `b ∈ {8,16,32,64,128}`, five runs:

| label | policy | `--num_ee_threshold` | `--ee_profile` | provides |
|---|---|---|---|---|
| `off` | off | (n/a) | no | `t_f` (clean full pass), no-EE decode-throughput baseline |
| `median` | median | (n/a) | no | median-EE decode-throughput baseline |
| `rebatch` | rebatching | `2` (fixed, forces real flushing) | no | `t_s`, `t_d`, `c`, `c/t_d`, `ART_derived`, rebatching decode throughput |
| `rebatch_auto` | rebatching | `-1` (auto) | no | `ART_auto` (the value the engine actually configures; parsed from log) |
| `rebatch_prof` | rebatching | `2` | yes (`--ee_profile`, sync on) | the `c` breakdown JSON sidecar |

Why a **fixed** threshold (`2`) for the `c`/`c/t_d` cell: with auto-ART the threshold degenerates to `b//2` (documented: the negative-overhead artifact when EE≈0 leaves `rebatching_ee_factor=0`), so few flushes happen, `t_d→0`, and `c` becomes a meaningless negative artifact. A fixed low threshold forces real EE + flushing so `t_d` and `c` are well-defined and positive — the regime where `c/t_d` is meaningful. `rebatch_auto` is kept *separately* to report the genuine auto value the user asked for (and to document the degeneracy).

`ART_derived = b · (c / t_d)` is the threshold the adaptive formula *prescribes* given the measured overhead — that is the quantity that demonstrates the scaling thesis. `ART_auto` is what the runtime currently sets.

## Part 0.3 — The summary table (deliverable)

`summarize.py` emits one row per batch size:

```
b | dec_tput | ART_auto | ART_deriv | c(ms) | t_f(ms) | t_s(ms) | t_d(ms) | c/t_d | kvfill | bufstg | upd_kv | stepsync | skipmask | resid | Δoff% | Δmedian%
```

- `dec_tput` = `num_output_tokens / decode_time` of `rebatch` (≡ 1/tpot).
- `c = t_s + t_d − t_f` where `t_s=avg_ee_iter_time`, `t_d=avg_deep_iter_time` (from `rebatch`), `t_f=avg_normal_iter_time` from `off` (clean full pass at matched b).
- `c/t_d` — the central ratio (thesis: ~flat / bounded in b).
- breakdown (ms, mean per occurrence, from `rebatch_prof` sidecar): `kvfill` (data), `bufstg` = buffer add+take (data), `upd_kv` (control), `stepsync` (control), `skipmask` (control), `resid = c − (kvfill+bufstg+upd_kv+stepsync+skipmask)` (two-pass fixed/launch + unattributed). The breakdown is an attribution, not an exact partition (CPU-launch timers + async GPU); this caveat is printed under the table.
- `Δoff%` = `100·(dec_tput_rebatch − dec_tput_off)/dec_tput_off`; `Δmedian%` likewise vs `median`. (Expected to be ≤0 at layer 20 / conf 0.2 — the point of *this* experiment is the overhead-scaling story, not a net throughput win; report honestly.)

---

## File Structure

**Engine changes (2 files, both default-preserving — no behavior change unless the new env vars are set):**
- `sarathi-lean/sarathi/model_executor/models/llama.py` — add a JSON-sidecar dump of the existing profile aggregates (Task 1); add optional `synchronize()` to the two data-plane timers (Task 2).
- `sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py` — expose `step_times` mean for the sidecar (read-only helper; Task 1).

**New experiment tree (no engine logic):**
```
experiments/overhead_scaling/
  README.md                      # experiment description (what/why/metric/thesis)
  b200_llama_13b/
    README.md                    # env + exact launch command + how to re-summarize
    run.sh                       # sweep: 5 runs × 5 batch sizes, then summarize
    summarize.py                 # results/*.csv + *.json sidecars -> results/summary.txt
    results/                     # created at runtime
```

**Docs:**
- `experiments/README.md` — add the row for `overhead_scaling/` to the index table.

---

## Task 1: Emit existing profile aggregates to a JSON sidecar

**Files:**
- Modify: `sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py` (add a read helper near `step_times`)
- Modify: `sarathi-lean/sarathi/model_executor/models/llama.py:836` (forward, after the commented profile prints)

- [ ] **Step 1: Add a `step_times` accessor to the cache engine**

In `vATTN_cache_engine.py`, the `step_times` list is populated at line 253-254 but only `__init__`-ed implicitly. Confirm it is initialized in `__init__` (search `self.step_times`); if it is only referenced under `EE_PROFILE`, add `self.step_times = []` in `__init__` next to the other instance attrs. Then add this method to the class (anywhere after `step`):

```python
    def mean_step_time(self) -> float:
        """Mean CPU-side time of one vattention.step() sync (s). 0 if unprofiled."""
        return (sum(self.step_times) / len(self.step_times)) if self.step_times else 0.0
```

- [ ] **Step 2: Add the sidecar dump in `LlamaModel.forward`**

In `llama.py`, immediately after the block of commented-out profile prints (the line `# print(f"[LlamaModel.forward] hidden states buffer spent time ...")` at ~1044 is too late because of early returns; instead place this near the **top** of `forward`, right after the two commented `time adding/taking` prints at ~832-836, so it runs on every forward regardless of return path). Insert:

```python
        # --- EE profiling sidecar (DREX_EE_PROFILE=1 + DREX_EE_PROFILE_OUT=<path>) ---
        # All timers below already exist; this only serializes their running
        # aggregates so an offline summarizer can read them. No-op unless both
        # env vars are set. Writes cumulative means every 50 forwards (last write
        # == final aggregates). Off the hot path otherwise.
        if EE_PROFILE:
            self._prof_fwd_count = getattr(self, "_prof_fwd_count", 0) + 1
            _prof_out = os.environ.get("DREX_EE_PROFILE_OUT", "")
            if _prof_out and self._prof_fwd_count % 50 == 0:
                def _mean(lst):
                    return (sum(lst) / len(lst)) if lst else 0.0
                _step_mean = cache_engine.mean_step_time() if cache_engine is not None else 0.0
                _prof = {
                    "forwards": self._prof_fwd_count,
                    "ee_overhead_ms": _mean(self.ee_overhead_time_lst) * 1e3,
                    "fill_kvcache_ms": _mean(self.fill_kvcache_time_lst) * 1e3,
                    "update_kvcache_ms": _mean(self.update_kvcache_time_lst) * 1e3,
                    "buf_add_ms": _mean(self.deep_buffer.time_spent_adding) * 1e3,
                    "buf_take_ms": _mean(self.deep_buffer.time_spent_taking) * 1e3,
                    "step_sync_ms": _step_mean * 1e3,
                    "rebatching_total_ms": self.rebatching_time * 1e3,
                    "n_ee_overhead": len(self.ee_overhead_time_lst),
                    "n_fill_kvcache": len(self.fill_kvcache_time_lst),
                    "n_buf_add": len(self.deep_buffer.time_spent_adding),
                    "n_buf_take": len(self.deep_buffer.time_spent_taking),
                }
                try:
                    with open(_prof_out, "w") as _f:
                        json.dump(_prof, _f)
                except OSError:
                    pass
```

- [ ] **Step 3: Ensure `json` is imported in llama.py**

Run: `grep -n "^import json\|^import os" sarathi-lean/sarathi/model_executor/models/llama.py`
Expected: `os` is present. If `json` is absent, add `import json` next to `import os` at the top of the file.

- [ ] **Step 4: Syntax-check**

Run: `cd /vast/projects/liuv/pennnetworks/xutingl/vattention-ee && python -m py_compile sarathi-lean/sarathi/model_executor/models/llama.py sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py && echo PARSE_OK`
Expected: `PARSE_OK`

- [ ] **Step 5: Smoke that the sidecar is written (GPU, ~2 min)**

Run (on the B200 node):
```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vattention-ee
source scripts/drex_env.sh
export CUDA_VISIBLE_DEVICES=0 RAY_ADDRESS=local DREX_RAY_NUM_CPUS=16
export DREX_EE_PROFILE_OUT=/tmp/drex_prof_smoke.json
ray stop --force
python scripts/run_ee.py --model llama-2-13b --model_load_format auto \
  --ee_policy rebatching --shallow_exit_layer 20 --conf_threshold 0.2 \
  --max_batch_size 16 --num_requests 30 --num_ee_threshold 2 \
  --kv_method copy --collect_conf false --ee_profile \
  --csv_path /tmp/drex_prof_smoke_csv/
cat /tmp/drex_prof_smoke.json
```
Expected: the run ends with "Replica 0 exiting", and `/tmp/drex_prof_smoke.json` contains a JSON object with non-zero `fill_kvcache_ms` and `ee_overhead_ms` and `n_fill_kvcache > 0`. If the file is empty/missing, the worker may not share the driver's filesystem view — it does here (single node, `/tmp` local), so a missing file means the dump block was not reached: re-check it is above the flush early-return.

- [ ] **Step 6: Commit**

```bash
git add sarathi-lean/sarathi/model_executor/models/llama.py sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py
git commit -m "feat(drex): emit existing EE profile aggregates to JSON sidecar under --ee_profile

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(Note: repo is currently not a git repo per environment; if `git` errors, skip commit and proceed — the user manages commits per their operational constraints.)

## Task 2: Optional `synchronize()` for accurate data-plane GPU timing

**Files:**
- Modify: `sarathi-lean/sarathi/model_executor/models/llama.py` (the `fill_missing_kvcache_with_copy` timer ~650-664)
- Modify: `sarathi-lean/sarathi/model_executor/models/ee_utils.py` (the buffer add/take timers ~38-50, 63-85)

- [ ] **Step 1: Add the sync flag to ee_utils.py**

In `ee_utils.py`, next to the existing `EE_PROFILE` definition (line 13), add:

```python
EE_PROFILE_SYNC = os.environ.get("DREX_EE_PROFILE_SYNC", "0") == "1"
```

- [ ] **Step 2: Synchronize around the buffer copies (data-plane)**

In `ee_utils.py` `add_hidden_states`, change the timing tail (currently `if EE_PROFILE: self.time_spent_adding.append(...)`) to synchronize first when requested:

```python
        if EE_PROFILE:
            if EE_PROFILE_SYNC:
                torch.cuda.synchronize()
            self.time_spent_adding.append(time.perf_counter() - start_time)
```

Apply the identical change to `take_hidden_states` (`self.time_spent_taking`). `torch` is already imported in `ee_utils.py`.

- [ ] **Step 3: Synchronize around the KV fill (data-plane)**

In `llama.py`, add the same flag import usage. At the top of the file where `EE_PROFILE` is read (line 279), add:

```python
EE_PROFILE_SYNC = os.environ.get("DREX_EE_PROFILE_SYNC", "0") == "1"
```

Then in `fill_missing_kvcache_with_copy`, change the tail (line 663-664) to:

```python
        if EE_PROFILE:
            if EE_PROFILE_SYNC:
                torch.cuda.synchronize()
            self.fill_kvcache_time_lst.append(time.perf_counter() - start_time)
```

- [ ] **Step 4: Syntax-check**

Run: `cd /vast/projects/liuv/pennnetworks/xutingl/vattention-ee && python -m py_compile sarathi-lean/sarathi/model_executor/models/llama.py sarathi-lean/sarathi/model_executor/models/ee_utils.py && echo PARSE_OK`
Expected: `PARSE_OK`

- [ ] **Step 5: Smoke that sync mode raises the data-plane timings (GPU, ~2 min)**

Run the same command as Task 1 Step 5 but add `export DREX_EE_PROFILE_SYNC=1` before it and point `DREX_EE_PROFILE_OUT=/tmp/drex_prof_sync.json`. Then:
```bash
echo "no-sync:"; cat /tmp/drex_prof_smoke.json
echo; echo "sync:"; cat /tmp/drex_prof_sync.json
```
Expected: `fill_kvcache_ms` and `buf_add_ms` are **larger** under sync (true GPU time) than without; control-plane numbers (`step_sync_ms`, `ee_overhead_ms`) are similar. This confirms sync is doing its job. (If they are equal, the ops were already effectively synchronous — note it, not a failure.)

- [ ] **Step 6: Commit** (same caveat as Task 1 Step 6)

```bash
git add sarathi-lean/sarathi/model_executor/models/llama.py sarathi-lean/sarathi/model_executor/models/ee_utils.py
git commit -m "feat(drex): DREX_EE_PROFILE_SYNC for accurate data-plane GPU timing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Task 3: Experiment scaffolding — directory + READMEs

**Files:**
- Create: `experiments/overhead_scaling/README.md`
- Create: `experiments/overhead_scaling/b200_llama_13b/README.md`
- Modify: `experiments/README.md` (index table)

- [ ] **Step 1: Write `experiments/overhead_scaling/README.md`**

```markdown
# `overhead_scaling`

**Question.** How does the rebatching overhead `c` (and the ratio `c/t_d`) scale with
batch size, and what is `c` made of? Thesis: rebatching regroups sequences by virtual
KV indexing (no copy to regroup); the data-plane copies that exist (deep-layer KV fill
for exited tokens + hidden-state staging) are small and `O(b)` but dominated by the
`O(b)` deep forward `t_d`; the control plane is `O(1)` per rebatch (one CPU↔GPU sync) +
cheap per-request pointer bookkeeping. So `c/t_d` stays bounded as b grows and the
adaptive threshold `ART = b·(c/t_d)` scales gracefully.

**Definitions.** `t_f` = full-pass iteration time (no-EE baseline). `t_s` =
shallow/EE iteration time (`avg_ee_iter_time`). `t_d` = deep/flush iteration time
(`avg_deep_iter_time`). `c = t_s + t_d − t_f`. Decode throughput =
`num_output_tokens / decode_time` (≡ 1/tpot). `ART_auto` = the threshold the engine
auto-configures at runtime; `ART_derived = b·(c/t_d)` = the value the adaptive formula
prescribes given the measured overhead.

**Sweep.** Batch {8,16,32,64,128} on Llama-2-13B, exit layer 20, conf 0.2,
`kv_method=copy`, real weights, 100 requests. Per batch: `off`, `median`,
`rebatching@thr=2`, `rebatching@auto`, and `rebatching@thr=2 --ee_profile` (for the
`c` breakdown). Threshold is fixed at 2 for the overhead cells because auto-ART
degenerates to `b//2` (negative-overhead artifact when EE≈0) and would leave `t_d≈0`.

**Caveat.** The `c` breakdown is an attribution from CPU-launch-time timers (+ optional
`synchronize()` for data-plane), not an exact partition of `c`. Throughput numbers come
from non-profiled runs so they are unperturbed. Instances: `b200_llama_13b`.
```

- [ ] **Step 2: Write `experiments/overhead_scaling/b200_llama_13b/README.md`**

```markdown
# `overhead_scaling` on `b200_llama_13b`

Instance of [`overhead_scaling`](../README.md). Environment: NVIDIA B200 (sm_100),
Llama-2-13B-chat (40 layers), `fa_vattn_2mb` backend, real weights.

## Launch (on a B200 compute node)

```bash
cd experiments/overhead_scaling/b200_llama_13b
export CUDA_VISIBLE_DEVICES=0          # a free GPU on the shared node
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

`run.sh` sources `scripts/drex_env.sh`, sets `RAY_ADDRESS=local` and
`DREX_RAY_NUM_CPUS=16`, runs `ray stop --force` before each cell, and for each batch in
{8,16,32,64,128} runs `off`, `median`, `rebatch@thr=2`, `rebatch@auto`, and a
`--ee_profile` (sync) pass writing a JSON sidecar. It then runs `summarize.py`.

## Results

`results/<label>_b<bs>/req_*.csv` per cell, `results/prof_b<bs>.json` per profile pass,
per-cell logs `results/log_<label>_b<bs>.txt`, and `results/summary.txt` (the table).
Re-summarize without re-running: `python summarize.py --results-dir results`.

## Feasibility note

Batch 128 may OOM depending on `max_model_len`/KV-pool size (vAttention provisions
`2*batch+1` slots). `run.sh` smoke-checks b=128 first; if it fails, the cell is skipped
and noted in `summary.txt` rather than aborting the sweep.
```

- [ ] **Step 3: Add the index row to `experiments/README.md`**

In the "Experiments in this repo" table, after the `rebatching_overhead_ablation` row, add:

```markdown
| [`overhead_scaling/`](overhead_scaling/) | How rebatching overhead `c` and `c/t_d` scale with batch size {8..128}, and `c`'s control-plane vs data-plane breakdown. 13B |
```

- [ ] **Step 4: Verify the tree + links**

Run: `ls experiments/overhead_scaling experiments/overhead_scaling/b200_llama_13b && grep -n overhead_scaling experiments/README.md`
Expected: both READMEs listed; one new index line printed.

## Task 4: Write `summarize.py`

**Files:**
- Create: `experiments/overhead_scaling/b200_llama_13b/summarize.py`

- [ ] **Step 1: Write the summarizer**

It reads, per batch size, the four CSVs (`off`, `median`, `rebatch`, `rebatch_auto`) and the `prof_b<bs>.json` sidecar; parses `ART_auto` from `log_rebatch_auto_b<bs>.txt`; computes the table. Pure stdlib + pandas.

```python
#!/usr/bin/env python3
"""Aggregate overhead_scaling results into one table.

For each batch size b, expects under --results-dir:
  off_b<b>/req_*.csv          median_b<b>/req_*.csv
  rebatch_b<b>/req_*.csv       rebatch_auto_b<b>/req_*.csv   (optional)
  prof_b<b>.json               (optional; the c breakdown)
  log_rebatch_auto_b<b>.txt    (optional; parsed for ART_auto)
"""
import argparse, glob, json, os, re
import pandas as pd

BATCHES = [8, 16, 32, 64, 128]

def _scalar(csv_glob, col):
    files = glob.glob(csv_glob)
    if not files:
        return None
    df = pd.read_csv(files[0])
    if col not in df.columns or len(df) == 0:
        return None
    return float(df[col].iloc[0])

def _decode_tput(csv_glob):
    files = glob.glob(csv_glob)
    if not files:
        return None
    df = pd.read_csv(files[0])
    if "num_output_tokens" not in df.columns or "decode_time" not in df.columns:
        return None
    nout = float(df["num_output_tokens"].iloc[0])
    dt = float(df["decode_time"].iloc[0])
    return nout / dt if dt > 0 else None

def _parse_art_auto(log_path, batch):
    """Last 'rebatching_ee_factor updated to: X' -> X*batch; else 'not updated' -> batch//2."""
    if not os.path.exists(log_path):
        return None
    factor = None
    pat = re.compile(r"rebatching_ee_factor updated to:\s*([0-9.eE+-]+)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                factor = float(m.group(1))
    if factor is None:
        return batch // 2  # never updated -> the b//2 fallback
    return factor * batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    R = args.results_dir
    rows = []
    for b in BATCHES:
        t_f = _scalar(f"{R}/off_b{b}/req_*.csv", "avg_normal_iter_time")
        if t_f is None:
            rows.append({"b": b, "note": "missing (off run absent / OOM)"})
            continue
        t_s = _scalar(f"{R}/rebatch_b{b}/req_*.csv", "avg_ee_iter_time")
        t_d = _scalar(f"{R}/rebatch_b{b}/req_*.csv", "avg_deep_iter_time")
        dtput = _decode_tput(f"{R}/rebatch_b{b}/req_*.csv")
        dtput_off = _decode_tput(f"{R}/off_b{b}/req_*.csv")
        dtput_med = _decode_tput(f"{R}/median_b{b}/req_*.csv")
        c = (t_s + t_d - t_f) if None not in (t_s, t_d) else None
        c_over_td = (c / t_d) if (c is not None and t_d) else None
        art_deriv = (b * c_over_td) if c_over_td is not None else None
        art_auto = _parse_art_auto(f"{R}/log_rebatch_auto_b{b}.txt", b)

        prof = {}
        pj = f"{R}/prof_b{b}.json"
        if os.path.exists(pj):
            with open(pj) as f:
                prof = json.load(f)
        ms = lambda k: prof.get(k, 0.0)
        kvfill = ms("fill_kvcache_ms")
        bufstg = ms("buf_add_ms") + ms("buf_take_ms")
        upd_kv = ms("update_kvcache_ms")
        stepsync = ms("step_sync_ms")
        skipmask = ms("ee_overhead_ms")
        c_ms = (c * 1e3) if c is not None else None
        resid = (c_ms - (kvfill + bufstg + upd_kv + stepsync + skipmask)) if c_ms is not None else None

        def pct(a, base):
            return 100.0 * (a - base) / base if (a is not None and base) else None
        rows.append({
            "b": b,
            "dec_tput": dtput,
            "ART_auto": art_auto,
            "ART_deriv": art_deriv,
            "c_ms": c_ms,
            "t_f_ms": t_f * 1e3 if t_f else None,
            "t_s_ms": t_s * 1e3 if t_s else None,
            "t_d_ms": t_d * 1e3 if t_d else None,
            "c_over_td": c_over_td,
            "kvfill": kvfill, "bufstg": bufstg, "upd_kv": upd_kv,
            "stepsync": stepsync, "skipmask": skipmask, "resid": resid,
            "d_off%": pct(dtput, dtput_off),
            "d_med%": pct(dtput, dtput_med),
        })

    df = pd.DataFrame(rows)
    cols = ["b","dec_tput","ART_auto","ART_deriv","c_ms","t_f_ms","t_s_ms","t_d_ms",
            "c_over_td","kvfill","bufstg","upd_kv","stepsync","skipmask","resid",
            "d_off%","d_med%"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(df[cols].to_string(index=False))
    print()
    print("c_ms = t_s+t_d-t_f (t_f from off baseline). c/t_d is the scaling ratio (thesis: ~flat).")
    print("ART_auto = engine runtime value (parsed from log; b//2 if never updated).")
    print("ART_deriv = b*(c/t_d), the value the adaptive formula prescribes.")
    print("breakdown ms = mean per occurrence from --ee_profile sidecar (data: kvfill,bufstg; "
          "control: upd_kv,stepsync,skipmask; resid = c - sum, the two-pass/launch fixed cost).")
    print("Breakdown is an attribution (CPU-launch timers + optional sync), not an exact partition.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Static-parse check (no GPU)**

Run: `cd experiments/overhead_scaling/b200_llama_13b && python -c "import ast,pathlib; ast.parse(pathlib.Path('summarize.py').read_text()); print('AST_OK')"`
Expected: `AST_OK`

- [ ] **Step 3: Sanity-run on a legacy CSV (no GPU)**

Run: copy one existing rebatching CSV into a temp layout and confirm the decode-throughput math and ART parse don't crash:
```bash
cd experiments/overhead_scaling/b200_llama_13b
mkdir -p /tmp/ov_test/off_b16 /tmp/ov_test/rebatch_b16
cp ../../num_ee_threshold_sweep/b200_llama_13b/results/*off*.csv /tmp/ov_test/off_b16/ 2>/dev/null || true
cp ../../num_ee_threshold_sweep/b200_llama_13b/results/*thr2*rebatching*.csv /tmp/ov_test/rebatch_b16/ 2>/dev/null || \
  cp ../../num_ee_threshold_sweep/b200_llama_13b/results/*rebatching*.csv /tmp/ov_test/rebatch_b16/ 2>/dev/null || true
python summarize.py --results-dir /tmp/ov_test
```
Expected: a one-row (b=16) table prints with finite `c_ms`, `t_d_ms`, `c_over_td`; other batch rows show the "missing" note. (Exact legacy filenames vary; if no CSV matched, the table is all-missing — that still proves the script runs. The real check is Task 6.)

- [ ] **Step 4: Commit** (same git caveat)

```bash
git add experiments/overhead_scaling/b200_llama_13b/summarize.py
git commit -m "feat(drex): overhead_scaling summarizer"
```

## Task 5: Write `run.sh`

**Files:**
- Create: `experiments/overhead_scaling/b200_llama_13b/run.sh`

- [ ] **Step 1: Write the run script**

```bash
#!/usr/bin/env bash
#
# overhead_scaling / b200_llama_13b
# ---------------------------------
# Per batch in {8,16,32,64,128}: off / median / rebatch@thr2 / rebatch@auto /
# rebatch@thr2+profile. Then summarize. Shows how c and c/t_d scale with batch.
#
# Launch (on a B200 compute node):
#     cd experiments/overhead_scaling/b200_llama_13b
#     export CUDA_VISIBLE_DEVICES=0
#     nohup bash run.sh > run.log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
RESULTS_DIR="$HERE/results"
mkdir -p "$RESULTS_DIR"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/drex_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_ADDRESS=local
export DREX_RAY_NUM_CPUS="${DREX_RAY_NUM_CPUS:-16}"

MODEL="llama-2-13b"
EXIT_LAYER=20
CONF=0.2
NUM_REQUESTS=100
KV_METHOD="copy"
LOAD_FORMAT="auto"
BATCHES=(8 16 32 64 128)

# run_cell <label> <bs> <policy> <thr> <profile:0|1>
run_cell() {
  local label="$1" bs="$2" policy="$3" thr="$4" prof="$5"
  local out="$RESULTS_DIR/${label}_b${bs}"
  mkdir -p "$out"
  local extra=() 
  unset DREX_EE_PROFILE_OUT DREX_EE_PROFILE_SYNC
  if [[ "$prof" == "1" ]]; then
    extra+=(--ee_profile)
    export DREX_EE_PROFILE_OUT="$RESULTS_DIR/prof_b${bs}.json"
    export DREX_EE_PROFILE_SYNC=1
  fi
  echo "=== ${label} b=${bs} (policy=${policy} thr=${thr} prof=${prof})  $(date) ==="
  ray stop --force >/dev/null 2>&1 || true
  python -u "$REPO_ROOT/scripts/run_ee.py" \
    --model "$MODEL" --model_load_format "$LOAD_FORMAT" \
    --ee_policy "$policy" \
    --shallow_exit_layer "$EXIT_LAYER" --conf_threshold "$CONF" \
    --max_batch_size "$bs" --num_requests "$NUM_REQUESTS" \
    --num_ee_threshold "$thr" \
    --kv_method "$KV_METHOD" --collect_conf false \
    "${extra[@]}" \
    --csv_path "$out/" 2>&1 | tee "$RESULTS_DIR/log_${label}_b${bs}.txt"
  ls "$out"/req_*.csv >/dev/null 2>&1 && echo "OK ${label} b=${bs}" || echo "FAIL ${label} b=${bs}"
  echo
}

echo "=== overhead_scaling / b200_llama_13b  $(date) ==="

# Feasibility gate: smoke b=128 off with few requests first; skip b=128 if it fails.
SKIP_128=0
echo "=== feasibility smoke: off b=128 (10 req) ==="
ray stop --force >/dev/null 2>&1 || true
if ! python -u "$REPO_ROOT/scripts/run_ee.py" --model "$MODEL" --model_load_format "$LOAD_FORMAT" \
      --ee_policy off --shallow_exit_layer "$EXIT_LAYER" --conf_threshold "$CONF" \
      --max_batch_size 128 --num_requests 10 --num_ee_threshold -1 \
      --kv_method "$KV_METHOD" --collect_conf false \
      --csv_path "$RESULTS_DIR/smoke_b128/" > "$RESULTS_DIR/log_smoke_b128.txt" 2>&1 \
   || ! ls "$RESULTS_DIR"/smoke_b128/req_*.csv >/dev/null 2>&1; then
  echo "WARN: b=128 smoke failed (likely OOM) -> skipping b=128 cells. See log_smoke_b128.txt"
  SKIP_128=1
fi

for bs in "${BATCHES[@]}"; do
  if [[ "$bs" == "128" && "$SKIP_128" == "1" ]]; then
    echo "=== SKIP b=128 (feasibility) ==="; continue
  fi
  run_cell off          "$bs" off         -1 0
  run_cell median       "$bs" median      -1 0
  run_cell rebatch      "$bs" rebatching   2 0
  run_cell rebatch_auto "$bs" rebatching  -1 0
  run_cell prof         "$bs" rebatching   2 1   # writes prof_b<bs>.json
done

echo "=== summarize  $(date) ==="
python -u "$HERE/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"
echo "=== done  $(date) ==="
```

Note: the `prof` cell's CSV is *not* used for throughput (sync perturbs timing) — only its sidecar `prof_b<bs>.json`. The `rebatch` cell (no profile) supplies `dec_tput`, `t_s`, `t_d`, `c`.

- [ ] **Step 2: Bash syntax check**

Run: `bash -n experiments/overhead_scaling/b200_llama_13b/run.sh && echo BASH_OK`
Expected: `BASH_OK`

- [ ] **Step 3: Commit** (same git caveat)

```bash
git add experiments/overhead_scaling/b200_llama_13b/run.sh
git commit -m "feat(drex): overhead_scaling run script"
```

## Task 6: Execute the sweep and validate the table

**Files:** none (produces `results/`)

- [ ] **Step 1: Confirm shell is on a B200 compute node with a free GPU**

Run: `nvidia-smi --query-gpu=index,memory.used --format=csv` and pick a free index for `CUDA_VISIBLE_DEVICES`.
Expected: at least one GPU with low memory used. (If on a login node, get a compute shell first — sbatch is broken for DREX per CLAUDE.md.)

- [ ] **Step 2: Launch the sweep**

```bash
cd /vast/projects/liuv/pennnetworks/xutingl/vattention-ee/experiments/overhead_scaling/b200_llama_13b
export CUDA_VISIBLE_DEVICES=0   # or the free index
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```
Expected: each cell ends with "Replica 0 exiting" then "OK <label> b=<bs>"; ~21-25 cells total (5 batch × ~5 runs, minus b=128 if skipped). Wall time ~45-90 min.

- [ ] **Step 3: Validate `summary.txt` against the thesis**

Run: `cat results/summary.txt`
Expected and to check:
- `c_ms > 0` for every batch where rebatching does real flushing (thr=2) — `c` should be tens of ms, not negative (negative would mean the thr=2 cell still did ~no flushing; check `num_no_ee_iter` vs `num_ee_iter` in the rebatch CSV).
- `c_over_td` stays in a **bounded band** across b (does not grow ~linearly). This is the headline result. Note its trend in the README findings.
- `ART_deriv = b·(c/t_d)` grows ~linearly with b (because `c/t_d` is ~flat); `ART_auto` likely sits at `b//2` (degenerate) — contrast the two.
- breakdown: control-plane (`stepsync+skipmask+upd_kv`) vs data-plane (`kvfill+bufstg`) shares, and how each scales with b. Thesis predicts data-plane small and not exploding relative to `t_d`.
- `d_off%` / `d_med%`: report honestly (expected ≤0 at this operating point).

- [ ] **Step 4: If `c_over_td` is NOT bounded (grows with b)**

Then the thesis is partially refuted — inspect which breakdown term grows: if `kvfill` or `bufstg` grows ~linearly and is a large share, the data-plane is the culprit (the `O(b)` copies are not negligible at large b after all). Record this as the finding; it would mean "copy-free regrouping" is true but the `kv_method=copy` fill is the scaling liability → suggests testing `kv_method` alternatives or a no-fill variant. Do not massage the result.

- [ ] **Step 5: Commit results + a findings note** (same git caveat)

```bash
git add experiments/overhead_scaling/b200_llama_13b/results/summary.txt
# (results/ may be gitignored; if so, keep summary.txt by force-add or note in README)
git commit -m "results(drex): overhead_scaling 13B summary"
```

---

## Self-Review

**Spec coverage** (user's requested columns): batch ✓ (`b`); decode throughput ✓ (`dec_tput`); ART auto ✓ (`ART_auto`, plus `ART_deriv`); `c` ✓ (`c_ms`); `t_f` full-pass ✓ (`t_f_ms`); `t_s` shallow ✓ (`t_s_ms`); `t_d` ✓ (`t_d_ms`); `c`'s breakdown ✓ (`kvfill/bufstg/upd_kv/stepsync/skipmask/resid` + `c_over_td`); improvement over no-EE ✓ (`d_off%`); improvement over median-EE ✓ (`d_med%`). Batch sizes {8,16,32,64,128} ✓. "Does `--ee_profile` give enough info?" answered ✓ (Part 0.1: yes for timers, no for emission+accuracy → Tasks 1-2). Claim-accuracy question answered ✓ (Part 0).

**Placeholder scan:** every code/edit step contains literal code; no TODO/TBD; commands have expected output. The one judgement step (Task 6 Step 4) is an explicit branch, not a placeholder.

**Type/name consistency:** sidecar JSON keys written in Task 1 (`fill_kvcache_ms`, `ee_overhead_ms`, `update_kvcache_ms`, `buf_add_ms`, `buf_take_ms`, `step_sync_ms`, `rebatching_total_ms`) are exactly the keys read by `summarize.py` in Task 4. `mean_step_time()` defined in Task 1 Step 1 is the method called in Task 1 Step 2. Env vars `DREX_EE_PROFILE_OUT` / `DREX_EE_PROFILE_SYNC` are set in `run.sh` (Task 5) and read in Tasks 1-2. Cell dir naming `<label>_b<bs>` in `run.sh` matches the globs in `summarize.py`.

**Assumptions made (flag if wrong):** (1) "improvement over no-ee-median" = vs the `median` EE policy. (2) exit layer 20 / conf 0.2 reused from prior experiments. (3) `t_f` taken from the `off` baseline (clean full pass), not the rebatching run's in-run normal iters. If any differs, adjust Task 5 (runs) and Task 4 (math) accordingly.
