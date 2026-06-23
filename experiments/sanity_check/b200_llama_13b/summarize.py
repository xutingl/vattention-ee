#!/usr/bin/env python3
"""Summarize decode throughput from a results/ directory of run_ee.py CSVs.

Each run_ee.py run writes one CSV named
    req_<nreq>_batch_<bs>_layer_<layer>_conf_<conf>_<policy>_<kv>.csv
(see benchmark_runner.py). The throughput-relevant columns are run-level scalars
broadcast onto every row, so we read row 0:

    num_output_tokens   total decode output tokens for the run
    decode_time         total decode-phase wall time (s)
    tpot                decode_time / sum(per-req output tokens)
    throughput          OVERALL output throughput = num_output_tokens / total_time
                        (includes prefill + scheduling; NOT decode-only)

We report DECODE throughput = num_output_tokens / decode_time (== 1 / tpot), which
is what the sanity_check cares about, alongside the overall throughput for context.

No GPU needed; re-run any time to re-aggregate an existing results/ dir:
    python summarize.py --results-dir results
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

# req_100_batch_4_layer_20_conf_0.2_latency-only_copy.csv
NAME_RE = re.compile(
    r"^req_(?P<nreq>\d+)_batch_(?P<bs>\d+)_layer_(?P<layer>\d+)_"
    r"conf_(?P<conf>[\d.]+)_(?P<policy>.+)_(?P<kv>copy|postfill)\.csv$"
)


def parse_name(filename):
    m = NAME_RE.match(filename)
    if not m:
        return None
    d = m.groupdict()
    d["nreq"] = int(d["nreq"])
    d["bs"] = int(d["bs"])
    d["layer"] = int(d["layer"])
    return d


def collect(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "req_*.csv"))):
        meta = parse_name(os.path.basename(path))
        if meta is None:
            continue
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception as exc:  # unreadable / empty CSV
            print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            print(f"WARN: empty CSV {path}", file=sys.stderr)
            continue
        r = df.iloc[0]
        out_tokens = float(r.get("num_output_tokens", float("nan")))
        decode_time = float(r.get("decode_time", float("nan")))
        tpot = float(r.get("tpot", float("nan")))
        overall = float(r.get("throughput", float("nan")))
        decode_tps = out_tokens / decode_time if decode_time and decode_time > 0 else float("nan")
        rows.append({
            "policy": meta["policy"],
            "batch": meta["bs"],
            "decode_tps": decode_tps,
            "overall_tps": overall,
            "tpot": tpot,
            "decode_time": decode_time,
            "out_tokens": out_tokens,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results",
                    help="Directory of run_ee.py CSVs (default: results)")
    args = ap.parse_args()

    rows = collect(args.results_dir)
    if not rows:
        print(f"No run_ee.py CSVs (req_*.csv) found in {args.results_dir!r}.")
        return

    rows.sort(key=lambda x: (x["batch"], x["policy"]))

    header = (f"{'policy':<14} {'batch':>5} {'decode_tok/s':>13} {'overall_tok/s':>14} "
              f"{'tpot(s)':>10} {'decode_t(s)':>12} {'out_tok':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['policy']:<14} {r['batch']:>5} {r['decode_tps']:>13.2f} "
              f"{r['overall_tps']:>14.2f} {r['tpot']:>10.5f} {r['decode_time']:>12.3f} "
              f"{int(r['out_tokens']):>9}")

    # Flag gaps in the (policy x batch) grid implied by what's present (catches a
    # policy that ran at one batch size but failed at the other).
    policies = sorted({r["policy"] for r in rows})
    batches = sorted({r["batch"] for r in rows})
    present = {(r["policy"], r["batch"]) for r in rows}
    missing = [(p, b) for b in batches for p in policies if (p, b) not in present]
    print()
    print(f"{len(rows)} run(s) summarized | policies={policies} | batches={batches}")
    if missing:
        print("MISSING cells (no CSV — check the matching results/log_<policy>_batch<bs>.txt):")
        for p, b in missing:
            print(f"  - policy={p} batch={b}")
    else:
        print("No gaps in the policy x batch grid that's present.")


if __name__ == "__main__":
    main()
