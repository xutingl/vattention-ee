#!/usr/bin/env python3
"""Summarize the num_ee_threshold sweep: decode throughput vs the no-EE baseline.

Layout — each run lives in its own subdir (the benchmark CSV name does not encode
num_ee_threshold, so we key off the subdir name):
    results/b<bs>_<policy>/req_*.csv              baseline: policy in {off, eager, average}
    results/b<bs>_rebatching_thr<label>/req_*.csv rebatching at num_ee_threshold=<label>
                                                  (label in {auto,0,2,4,6}; auto = -1 = ART)

Per batch size it prints baselines first, then rebatching by threshold, with:
  decode_tok/s  = num_output_tokens / decode_time   (decode-phase throughput)
  vs_off%       = decode_tok/s relative to that batch's `off` baseline
  ee_rate       = fraction of output tokens generated via early exit
  forced_out    = involuntary exits (num_seq_would_not_ee_but_ee) — always 0 for
                  rebatching (it never forces a sequence out); nonzero for eager/average
  would_stay    = sequences that wanted to EE but were blocked by the threshold

No GPU needed; re-run any time:  python summarize.py --results-dir results
"""
import argparse
import glob
import os
import re

import pandas as pd

BASE_RE = re.compile(r"^b(?P<bs>\d+)_(?P<policy>off|eager|average)$")
REB_RE = re.compile(r"^b(?P<bs>\d+)_rebatching_thr(?P<thr>auto|\d+)$")


def parse_sub(name):
    m = REB_RE.match(name)
    if m:
        return {"bs": int(m["bs"]), "policy": "rebatching", "thr": m["thr"]}
    m = BASE_RE.match(name)
    if m:
        return {"bs": int(m["bs"]), "policy": m["policy"], "thr": None}
    return None


def collect(results_dir):
    rows = []
    for sub in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, sub)
        if not os.path.isdir(d):
            continue
        meta = parse_sub(sub)
        if meta is None:
            continue
        csvs = glob.glob(os.path.join(d, "req_*.csv"))
        if not csvs:
            rows.append({**meta, "missing": True})
            continue
        try:
            r = pd.read_csv(csvs[0], nrows=1).iloc[0]
        except Exception as exc:
            print(f"WARN: could not read {csvs[0]}: {exc}")
            rows.append({**meta, "missing": True})
            continue
        out = float(r["num_output_tokens"])
        dt = float(r["decode_time"])
        eet = int(r["num_ee_tokens"])
        noeet = int(r["num_no_ee_tokens"])
        rows.append({
            **meta,
            "missing": False,
            "decode_tps": out / dt if dt > 0 else float("nan"),
            "overall_tps": float(r["throughput"]),
            "ee_rate": eet / max(1, eet + noeet),
            "ee_iter": int(r["num_ee_iter"]),
            "forced_out": int(r["num_seq_would_not_ee_but_ee"]),
            "would_stay": int(r["num_seq_would_ee_but_stay"]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    rows = collect(args.results_dir)
    if not rows:
        print(f"No runs found under {args.results_dir!r} "
              "(expected subdirs like b8_off/ or b8_rebatching_thr2/).")
        return

    base_order = {"off": 0, "eager": 1, "average": 2}
    thr_order = {"auto": 0, "0": 1, "2": 2, "4": 3, "6": 4}

    def sort_key(r):
        if r["policy"] == "rebatching":
            return (1, thr_order.get(r["thr"], 99))
        return (0, base_order.get(r["policy"], 99))

    for bs in sorted({r["bs"] for r in rows}):
        br = sorted([r for r in rows if r["bs"] == bs], key=sort_key)
        off = next((r for r in br if r["policy"] == "off" and not r["missing"]), None)
        off_tps = off["decode_tps"] if off else float("nan")

        hdr = (f"{'config':<26}{'decode_tok/s':>13}{'vs_off%':>9}{'ee_rate':>9}"
               f"{'ee_iter':>9}{'forced_out':>11}{'would_stay':>11}")
        print(f"\n===================== BATCH {bs} =====================")
        print(hdr)
        print("-" * len(hdr))
        for r in br:
            label = (f"rebatching thr={r['thr']}" if r["policy"] == "rebatching"
                     else f"{r['policy']} (baseline)")
            if r["missing"]:
                print(f"{label:<26}{'MISSING — see subdir run.log':>13}")
                continue
            vs = ((r["decode_tps"] - off_tps) / off_tps * 100
                  if off_tps == off_tps else float("nan"))
            print(f"{label:<26}{r['decode_tps']:>13.1f}{vs:>9.1f}{r['ee_rate']:>9.3f}"
                  f"{r['ee_iter']:>9}{r['forced_out']:>11}{r['would_stay']:>11}")
    print("\nNote: for the `auto` rebatching cell, grep its subdir run.log for "
          "'rebatching_ee_factor' / 'overhead is negative' to see what ART chose.")


if __name__ == "__main__":
    main()
