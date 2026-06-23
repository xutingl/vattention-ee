#!/usr/bin/env python3
"""Aggregate overhead_scaling results into one table.

For each batch size b, expects under --results-dir:
  off_b<b>/req_*.csv          median_b<b>/req_*.csv
  rebatch_b<b>/req_*.csv       rebatch_auto_b<b>/req_*.csv   (optional)
  prof_b<b>.json               (optional; the c breakdown)
  log_rebatch_auto_b<b>.txt    (optional; parsed for ART_auto)
"""
import argparse, glob, json, os, re
import pandas as pd

BATCHES = [8, 16, 32, 64, 128]

def _scalar(csv_glob, col):
    files = glob.glob(csv_glob)
    if not files:
        return None
    df = pd.read_csv(files[0])
    if col not in df.columns or len(df) == 0:
        return None
    return float(df[col].iloc[0])

def _decode_tput(csv_glob):
    files = glob.glob(csv_glob)
    if not files:
        return None
    df = pd.read_csv(files[0])
    if "num_output_tokens" not in df.columns or "decode_time" not in df.columns:
        return None
    nout = float(df["num_output_tokens"].iloc[0])
    dt = float(df["decode_time"].iloc[0])
    return nout / dt if dt > 0 else None

def _parse_art_auto(log_path, batch):
    """Last 'rebatching_ee_factor updated to: X' -> X*batch; else 'not updated' -> batch//2."""
    if not os.path.exists(log_path):
        return None
    factor = None
    pat = re.compile(r"rebatching_ee_factor updated to:\s*([0-9.eE+-]+)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                factor = float(m.group(1))
    if factor is None:
        return batch // 2  # never updated -> the b//2 fallback
    return factor * batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    R = args.results_dir
    rows = []
    missing = []
    for b in BATCHES:
        t_f = _scalar(f"{R}/off_b{b}/req_*.csv", "avg_normal_iter_time")
        if t_f is None:
            missing.append(b)
            continue
        t_s = _scalar(f"{R}/rebatch_b{b}/req_*.csv", "avg_ee_iter_time")
        t_d = _scalar(f"{R}/rebatch_b{b}/req_*.csv", "avg_deep_iter_time")
        dtput = _decode_tput(f"{R}/rebatch_b{b}/req_*.csv")
        dtput_off = _decode_tput(f"{R}/off_b{b}/req_*.csv")
        dtput_med = _decode_tput(f"{R}/median_b{b}/req_*.csv")
        c = (t_s + t_d - t_f) if None not in (t_s, t_d) else None
        c_over_td = (c / t_d) if (c is not None and t_d) else None
        art_deriv = (b * c_over_td) if c_over_td is not None else None
        art_auto = _parse_art_auto(f"{R}/log_rebatch_auto_b{b}.txt", b)

        prof = {}
        pj = f"{R}/prof_b{b}.json"
        if os.path.exists(pj):
            with open(pj) as f:
                prof = json.load(f)
        ms = lambda k: prof.get(k, 0.0)
        kvfill = ms("fill_kvcache_ms")
        bufstg = ms("buf_add_ms") + ms("buf_take_ms")
        upd_kv = ms("update_kvcache_ms")
        stepsync = ms("step_sync_ms")
        skipmask = ms("ee_overhead_ms")
        c_ms = (c * 1e3) if c is not None else None
        resid = (c_ms - (kvfill + bufstg + upd_kv + stepsync + skipmask)) if c_ms is not None else None

        def pct(a, base):
            return 100.0 * (a - base) / base if (a is not None and base) else None
        rows.append({
            "b": b,
            "dec_tput": dtput,
            "ART_auto": art_auto,
            "ART_deriv": art_deriv,
            "c_ms": c_ms,
            "t_f_ms": t_f * 1e3 if t_f is not None else None,
            "t_s_ms": t_s * 1e3 if t_s is not None else None,
            "t_d_ms": t_d * 1e3 if t_d is not None else None,
            "c_over_td": c_over_td,
            "kvfill": kvfill, "bufstg": bufstg, "upd_kv": upd_kv,
            "stepsync": stepsync, "skipmask": skipmask, "resid": resid,
            "d_off%": pct(dtput, dtput_off),
            "d_med%": pct(dtput, dtput_med),
        })

    df = pd.DataFrame(rows)
    cols = ["b","dec_tput","ART_auto","ART_deriv","c_ms","t_f_ms","t_s_ms","t_d_ms",
            "c_over_td","kvfill","bufstg","upd_kv","stepsync","skipmask","resid",
            "d_off%","d_med%"]
    cols = [col for col in cols if col in df.columns]
    if len(df):
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", lambda v: f"{v:.3f}"):
            print(df[cols].to_string(index=False))
    else:
        print("(no batches with data)")
    for b in missing:
        print(f"MISSING b={b}: off run absent / OOM (see run.log / log_*_b{b}.txt)")
    print()
    print("c_ms = t_s+t_d-t_f (t_f from off baseline). c/t_d is the scaling ratio (thesis: ~flat).")
    print("ART_auto = engine runtime value (parsed from log; b//2 if never updated).")
    print("ART_deriv = b*(c/t_d), the value the adaptive formula prescribes.")
    print("breakdown ms = mean per occurrence from --ee_profile sidecar (data: kvfill,bufstg; "
          "control: upd_kv,stepsync,skipmask; resid = c - sum, the two-pass/launch fixed cost).")
    print("c_ms is from the rebatch cell; breakdown ms are from the separate prof cell (different invocation -> expect noise).")
    print("Breakdown is an attribution (CPU-launch timers + optional sync), not an exact partition.")

if __name__ == "__main__":
    main()
