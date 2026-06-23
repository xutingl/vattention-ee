# Experiment: `sanity_check`

A quick health check: confirm that **every EE policy plus the non-EE baseline** runs
end-to-end and record each one's **decode throughput** at two batch sizes. It doubles as
a throughput sanity baseline and as the reference template for new experiments.

## What it sweeps

Fixed config (Llama-2-13B):

| Param | Value |
|---|---|
| `--model` | `llama-2-13b` |
| `--shallow_exit_layer` | `20` |
| `--conf_threshold` | `0.2` |
| `--kv_method` | `copy` |
| `--num_requests` | `100` |
| `--model_load_format` | `auto` (real pretrained weights — required for meaningful EE) |
| `--collect_conf` | `false` (drops per-step host syncs → clean throughput) |

Swept:

- `--ee_policy` ∈ `{ off, eager, lazy, average, median, rebatching, latency-only }`
  (`off` = the non-EE baseline; the other six are the EE methods, from
  [`ee_utils.py`](../../sarathi-lean/sarathi/model_executor/models/ee_utils.py)).
- `--max_batch_size` ∈ `{ 4, 8 }`.

→ **7 policies × 2 batch sizes = 14 runs.**

## The metric: decode throughput

Each run writes a CSV with run-level scalar columns (see
[`benchmark_runner.py`](../../sarathi-lean/sarathi/benchmark/benchmark_runner.py)).
Note the distinction:

- The CSV `throughput` column is **overall** output throughput
  = `num_output_tokens / total_time` — it includes prefill, scheduling, BERT scoring, etc.
- **Decode throughput** (what this experiment reports) is
  `num_output_tokens / decode_time` (≡ `1 / tpot`), using the `decode_time` / `tpot`
  columns, so it isolates the decode phase.

`summarize.py` reports decode throughput as the headline and overall throughput for
context.

## Caveats

- Only `--ee_policy rebatching` with the `fa_vattn` backend is the repo's
  **fully-validated** path (per [`CLAUDE.md`](../../CLAUDE.md)). The other policies are
  being sanity-checked here; if one fails to run end-to-end, that is itself a useful
  finding. `run_ee.py` swallows subprocess errors, so judge per-run success from the
  matching `results/log_<policy>_batch<bs>.txt`, not from an exit code — a failed run
  shows up as a missing CSV / missing cell in the summary.
- Runs load **real** weights (`--model_load_format auto`) so confidence scores — and thus
  the EE decisions — are meaningful. The early-exit *ramp* still uses the default
  (untrained) head unless `--early_exit_head_path` is set, so EE-token quality is limited
  even with real base weights. An earlier `dummy`-weight (plumbing) run is kept in
  `b200_llama_13b/results_dummy/` for comparison.

## Instances

- [`b200_llama_13b/`](b200_llama_13b/) — NVIDIA B200, Llama-2-13B-chat.
