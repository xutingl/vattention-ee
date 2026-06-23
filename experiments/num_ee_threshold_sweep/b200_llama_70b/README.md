# `num_ee_threshold_sweep` on `b200_llama_70b`

Instance of [`num_ee_threshold_sweep`](../README.md) on **Llama-2-70B-chat**. Same design
as [`b200_llama_13b`](../b200_llama_13b/) (sweep rebatching `num_ee_threshold` vs `off` /
`eager` / `average` baselines, per batch size), to see whether the rebatching/ART picture
changes on a much larger model.

## Environment

| | |
|---|---|
| GPU | NVIDIA **B200** (Blackwell, sm_100, ~179 GiB), shared `dgx-b200` node |
| Model | **Llama-2-70B-chat** (80 layers, hidden 8192, **GQA** 8 KV heads), key `llama-2-70b` |
| Weights | `load_format: auto` — real fp16, **~129 GiB** loaded per run (local path) |
| Backend | `fa_vattn_2mb` (forced by `run_ee.py`) |

> **Memory:** 70B fp16 weights (~129 GiB) fit on one B200; with `gpu_memory_utilization=0.95`
> (~170 GiB) that leaves ~40 GiB for KV + activations. `max_model_len=512` makes KV tiny
> (~320 KiB/token → batch 16 × 512 ≈ 2.6 GiB), so batch 16 is expected to fit. The exact
> `BATCH_SIZES` in `run.sh` are confirmed by a quick load+1-cell **memory probe** before the
> full sweep; if a cell OOMs, drop to the next-smaller sizes.
>
> **Ray CPU cap:** `run.sh` exports `DREX_RAY_NUM_CPUS=16` — required so Ray doesn't
> prestart ~224 workers and hang on this shared node (see `benchmark_runner.py`).

## Run it (direct + nohup, on a compute node)

```bash
cd experiments/num_ee_threshold_sweep/b200_llama_70b
export CUDA_VISIBLE_DEVICES=0
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

70B runs are slower than 13B (~129 GiB load + ~2.7× compute per token), so budget more
wall-clock per cell. Edit the config block at the top of `run.sh` to change batch sizes,
thresholds, or baselines.

## Results

Same layout/columns as the 13B instance — per-run subdirs under `results/`
(`b<bs>_<policy>/`, `b<bs>_rebatching_thr<label>/`), summarized to `results/summary.txt`:

```bash
python summarize.py --results-dir results
```

Reports per (policy, batch): `decode_tok/s`, `vs_off%`, `ee_rate`, `ee_iter`, `forced_out`
(0 for rebatching, nonzero for eager/average), `would_stay`. For the `auto` cells, grep the
subdir `run.log` for `rebatching_ee_factor` / `overhead is negative` to see ART's choice.
