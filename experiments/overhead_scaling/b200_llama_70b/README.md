# `overhead_scaling` on `b200_llama_70b`

Instance of [`overhead_scaling`](../README.md). Environment: NVIDIA B200 (sm_100),
Llama-2-70B-chat (80 layers), `fa_vattn_2mb` backend, real weights.

## Why 70B (vs the 13B instance)

On 13B the deep flush `t_d` measured ~= a full pass `t_f` (a 20-layer deep flush cost
about as much as the full 40-layer forward), because 13B decode is dominated by
per-iteration fixed cost / memory bandwidth — so early exit buys nothing. 70B has a much
heavier per-layer forward, so the deep portion (50 of 80 layers) should be a real
fraction of the full pass, making `t_d` genuinely large and EE/rebatching meaningful.

Exit layer is **30** (deep = layers 30..79, i.e. 50 of 80) with **conf_threshold 0.01**
(very low margin → aggressive early exit, so many tokens exit at layer 30 and rebatching
flushes heavily). This maximizes EE activity while keeping a large deep pass.

## Launch (on a B200 compute node)

```bash
cd experiments/overhead_scaling/b200_llama_70b
export CUDA_VISIBLE_DEVICES=0          # a free GPU on the shared node
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

## Batch sizes / OOM handling

70B is ~140GB in fp16; on the 183GB B200, large batches may not fit (KV pool is
provisioned for `2*batch+1` slots). `run.sh` sweeps batches ASCENDING {8,16,32,64,128}
and stops at the first batch whose `off` cell fails to produce a CSV (OOM is monotonic
in batch size) — so the feasible range is found automatically; skipped batches appear as
`MISSING` in `summary.txt`.

## Results

`results/<label>_b<bs>/req_*.csv` per cell, `results/prof_b<bs>.json` per profile pass,
per-cell logs `results/log_<label>_b<bs>.txt`, and `results/summary.txt` (the overhead
table + the RCT table). Re-summarize without re-running:
`python summarize.py --results-dir results`.
