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
