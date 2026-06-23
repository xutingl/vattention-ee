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
from non-profiled runs so they are unperturbed. Instances: `b200_llama_13b`, `b200_llama_70b` (70B uses exit layer 30 / conf 0.01 and reports RCT).
