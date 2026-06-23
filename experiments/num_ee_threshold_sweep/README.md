# Experiment: `num_ee_threshold_sweep`

**Question:** does a **fixed** `num_ee_threshold` let rebatching escape the ART
auto-tuning trap and beat the no-EE baseline — and where is the sweet spot?

Background (from [`sanity_check`](../sanity_check/)): rebatching landed *on* the no-EE
baseline at batch 4 and 8. At batch 8 the cause was that measured overhead went
≈0/negative, so ART's auto threshold never updated and stayed pinned at `batch//2 = 4`
(needs ≥5 of 8 to exit) — a self-reinforcing trap. The decision rule is
`need_skip = num_ee > num_ee_threshold`, and a fixed threshold switches off ART
([ee_utils.py](../../sarathi-lean/sarathi/model_executor/models/ee_utils.py)), so it's
the quickest way to test whether breaking the trap recovers throughput.

## What it sweeps

Fixed config (Llama-2-13B, same as `sanity_check` for comparability):

| Param | Value |
|---|---|
| `--model` | `llama-2-13b` |
| `--shallow_exit_layer` | `20` |
| `--conf_threshold` | `0.2` |
| `--kv_method` | `copy` |
| `--num_requests` | `100` |
| `--model_load_format` | `auto` (real weights — required for meaningful EE) |
| `--collect_conf` | `false` |

Grid:

- **rebatching** `--num_ee_threshold` ∈ `{ auto(-1), 0, 2, 4, 6 }`
  (`num_ee > threshold` → `0`≈exit on any 1 confident seq ≈ latency-only; `2`→≥3; `4`→≥5; `6`→≥7).
- **baselines** `--ee_policy` ∈ `{ off, eager, average }` (these ignore the threshold).
- `--max_batch_size` ∈ `{ 8, 16 }`.

→ **(5 thresholds + 3 baselines) × 2 batch sizes = 16 runs.**

## What to look for

- **decode_tok/s vs the `off` baseline** — the headline. If a fixed threshold helps, the
  lower-threshold rebatching cells should climb above `off` (especially at batch 8, where
  `auto` is trapped). `eager`/`average` mark the forced-exit throughput ceiling.
- **forced_out** — involuntary exits (the quality cost). It is **always 0 for rebatching**
  (it never forces a sequence out), and **nonzero for eager/average** — that contrast is
  the point: rebatching tries to gain throughput *without* corrupting outputs.
- **ee_rate / would_stay** — how much EE each threshold actually unlocks.
- For the `auto` cell, `grep` its `run.log` for `rebatching_ee_factor` / `overhead is
  negative` to see whether ART updated or stayed trapped at this batch size.

## Caveats

- The early-exit *ramp* uses the default (untrained) head unless `--early_exit_head_path`
  is set, so EE-token *quality* is limited even with real base weights — this experiment
  measures throughput behavior, not answer quality.
- The benchmark CSV name does **not** encode `num_ee_threshold`, so each run is written to
  its own subdir (`results/b<bs>_rebatching_thr<label>/`) to avoid collisions. `summarize.py`
  keys off the subdir name.
- Only `rebatching` + `fa_vattn` is the repo's fully-validated path; judge per-run success
  from each subdir's `run.log` (`run_ee.py` swallows subprocess errors).

## Instances

- [`b200_llama_13b/`](b200_llama_13b/) — NVIDIA B200, Llama-2-13B-chat.
- [`b200_llama_70b/`](b200_llama_70b/) — NVIDIA B200, Llama-2-70B-chat (larger model; same sweep).
