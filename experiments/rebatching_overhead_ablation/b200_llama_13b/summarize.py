#!/usr/bin/env python3
"""Summarize the rebatching-overhead ablation: do (a) stronger inline draining and
(b) raised min_flush_size recover the throughput rebatching loses to flush overhead?

Each cell lives in results/<label>/req_*.csv. Labels (set by run.sh):
    off_b16       no-EE baseline (target throughput)
    base          rebatching thr=2, inline=2, flush=8   (current defaults)
    a_inline1     (a) inline=1  (strengthen inline draining)
    b_flush4      (b) flush=4   (lower; more flushes; direction check)
    b_flush12     (b) flush=12  (raise)
    b_flush16     (b) flush=16  (raise to batch size)
    ab_combined   (a)+(b): inline=1, flush=16

Reports decode_tok/s vs the off baseline, plus ee_rate / ee_iter / avg_deep_iter (the
flush cost). If a lever helps, its cell's vs_off% should rise above `base`.
"""
import argparse
import glob
import os

import pandas as pd

ORDER = ["off_b16", "base", "a_inline1", "b_flush4", "b_flush12", "b_flush16", "ab_combined"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    rows = {}
    for sub in sorted(os.listdir(args.results_dir)):
        d = os.path.join(args.results_dir, sub)
        if not os.path.isdir(d):
            continue
        csvs = glob.glob(os.path.join(d, "req_*.csv"))
        if not csvs:
            rows[sub] = None
            continue
        r = pd.read_csv(csvs[0], nrows=1).iloc[0]
        out = float(r["num_output_tokens"]); dt = float(r["decode_time"])
        eet = int(r["num_ee_tokens"]); noeet = int(r["num_no_ee_tokens"])
        rows[sub] = {
            "decode_tps": out / dt if dt > 0 else float("nan"),
            "ee_rate": eet / max(1, eet + noeet),
            "ee_iter": int(r["num_ee_iter"]),
            "deep_ms": float(r["avg_deep_iter_time"]) * 1000.0,
        }

    if not rows:
        print(f"No cells under {args.results_dir!r}.")
        return

    off = rows.get("off_b16")
    off_tps = off["decode_tps"] if off else float("nan")
    base = rows.get("base")
    base_tps = base["decode_tps"] if base else float("nan")

    ordered = [k for k in ORDER if k in rows] + [k for k in rows if k not in ORDER]
    hdr = (f"{'cell':<14}{'decode_tok/s':>13}{'vs_off%':>9}{'vs_base%':>9}"
           f"{'ee_rate':>9}{'ee_iter':>9}{'deep_iter_ms':>13}")
    print(hdr); print("-" * len(hdr))
    for k in ordered:
        v = rows[k]
        if v is None:
            print(f"{k:<14}{'MISSING — see run.log':>13}"); continue
        vo = (v["decode_tps"] - off_tps) / off_tps * 100 if off_tps == off_tps else float("nan")
        vb = (v["decode_tps"] - base_tps) / base_tps * 100 if base_tps == base_tps else float("nan")
        print(f"{k:<14}{v['decode_tps']:>13.1f}{vo:>9.1f}{vb:>9.1f}"
              f"{v['ee_rate']:>9.3f}{v['ee_iter']:>9}{v['deep_ms']:>13.2f}")
    print("\nvs_off% = vs no-EE baseline (target is to reach 0 / positive).")
    print("vs_base% = vs current rebatching defaults; >0 means the lever HELPED.")


if __name__ == "__main__":
    main()
