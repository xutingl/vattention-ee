# `num_ee_threshold_sweep` on `b200_llama_13b`

Instance of the [`num_ee_threshold_sweep`](../README.md) experiment. See the parent
README for the question, the full grid, and how to read the results.

## Environment

| | |
|---|---|
| GPU | NVIDIA **B200** (Blackwell, sm_100), shared `dgx-b200` node |
| Model | **Llama-2-13B-chat** (40 layers, hidden 5120), key `llama-2-13b` |
| Backend | `fa_vattn_2mb` (forced by `run_ee.py`) |
| Weights | `load_format: auto` — **real** pretrained (fp16), ~25 GB loaded per run |

## Run it

> ⚠️ **Known issue — `sbatch` does NOT work for DREX on this cluster; use a compute node.**
> DREX runs the model inside a **Ray** worker, and under the SLURM **batch** cgroup Ray
> either hangs at *"Started a local Ray instance"* or the worker fails to acquire the GPU:
> `torch.cuda.set_device()` → *"CUDA-capable device(s) is/are busy or unavailable"* — even
> on a free, `Default`-compute-mode GPU. This was isolated to **Ray-under-SLURM, not DREX**:
> a minimal 10-line `ray.init(num_gpus=1)` + GPU actor *also* hangs via sbatch. Findings:
> `ray stop --force` fixes the *hang* but not the worker GPU handoff;
> `RAY_NOSET_CUDA_VISIBLE_DEVICES=1` did not help; the worker is local (no stray cluster).
> The sbatch artifacts (`cell.sbatch`, `submit_b16.sh`, `diag_ray.sbatch`) are kept for a
> future fix — **do not rely on them yet.** Run **directly on a compute node** instead
> (the validated path; this is exactly how the batch-8 results were produced).

**Batch 16 — direct on a B200 compute node** (not the login node, not `sbatch`):

```bash
cd experiments/num_ee_threshold_sweep/b200_llama_13b
export CUDA_VISIBLE_DEVICES=0          # a free GPU on the node
nohup bash run_b16.sh > run_b16.log 2>&1 &
tail -f run_b16.log
```

`run_b16.sh` runs all 8 batch-16 cells sequentially (3 baselines + rebatching
`{auto,0,2,4,6}`), each loading ~25 GB → **~45 min**, then summarizes. Batch 8 is already
complete under `results/b8_*` (produced the same direct way on the interactive node).

## Results

Each run writes into its own subdir under `results/` (the benchmark CSV name doesn't
encode `num_ee_threshold`):

```
results/b8_off/            results/b8_rebatching_thrauto/   ...   results/b16_rebatching_thr6/
        b8_eager/                  b8_rebatching_thr0/
        b8_average/                b8_rebatching_thr2/
```

Each subdir holds the run's `req_*.csv` and a `run.log`. After the sweep,
`results/summary.txt` has the per-batch table.

Re-aggregate any time (no GPU):

```bash
python summarize.py --results-dir results
```

It prints, per batch size, baselines then rebatching-by-threshold, with **`decode_tok/s`**
(`num_output_tokens / decode_time`), **`vs_off%`** (relative to that batch's `off`
baseline), `ee_rate`, `ee_iter`, **`forced_out`** (involuntary exits — 0 for rebatching,
nonzero for eager/average), and `would_stay`. To see what ART chose for the `auto` cells:

```bash
grep -E "rebatching_ee_factor|overhead is negative" results/b8_rebatching_thrauto/run.log
grep -E "rebatching_ee_factor|overhead is negative" results/b16_rebatching_thrauto/run.log
```
