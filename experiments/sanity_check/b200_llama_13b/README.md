# `sanity_check` on `b200_llama_13b`

Instance of the [`sanity_check`](../README.md) experiment on this environment + model.
See the parent README for the full config, the swept grid, and the decode-throughput
metric definition.

## Environment

| | |
|---|---|
| GPU | NVIDIA **B200** (Blackwell, sm_100), shared `dgx-b200` node |
| Model | **Llama-2-13B-chat** (40 layers, hidden 5120), key `llama-2-13b` |
| Backend | `fa_vattn_2mb` (FlashAttention + vAttention megacache; forced by `run_ee.py`) |
| Weights | `load_format: auto` — **real** pretrained Llama-2-13B-chat (fp16), ~25 GB loaded from disk per run |

> **Real weights.** Early-exit decisions hinge on softmax confidence, which is only
> meaningful with real weights — so this experiment loads them (`--model_load_format auto`).
> Cost: ~25 GB read per run (adds startup time over a `dummy` plumbing run).
> The earlier `dummy`-weight run is preserved in [`results_dummy/`](results_dummy/) for
> comparison. **Caveat:** the early-exit *ramp* still uses the default (untrained) head
> unless `--early_exit_head_path` is set, so EE-token quality is limited even with real
> base weights.

## Run it (direct + nohup)

From a shell already on the B200 compute node:

```bash
cd experiments/sanity_check/b200_llama_13b
export CUDA_VISIBLE_DEVICES=0          # a free GPU on the shared node
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

`run.sh` sources `scripts/drex_env.sh`, sets `RAY_ADDRESS=local`, and loops the 14 runs
(7 policies × batch {4, 8}), running `ray stop --force` before each. Expect roughly
**~30 min** total (~0.7 s/request + model load per run).

The config lives in the block at the top of `run.sh` — edit it there to change the
layer, conf, batch sizes, policies, or request count.

## Results

Written under `results/`:

- `results/req_100_batch_{4,8}_layer_20_conf_0.2_<policy>_copy.csv` — one per run.
- `results/log_<policy>_batch{4,8}.txt` — per-run stdout (use these to debug a failed
  policy; `run_ee.py` hides subprocess errors).
- `results/summary.txt` — the decode-throughput table (written at the end of the sweep).

Re-aggregate any time without re-running (no GPU needed):

```bash
python summarize.py --results-dir results
```

`summarize.py` reports, per (policy, batch): **`decode_tok/s`**
(`num_output_tokens / decode_time`, the headline), `overall_tok/s` (the CSV `throughput`
column, includes prefill), `tpot`, `decode_time`, and `out_tokens`. It also flags any
gaps in the policy × batch grid (a run that failed at one batch size but not the other).
