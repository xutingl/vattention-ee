#!/usr/bin/env python3
"""
Run vattention-ee on Balcony-LLaMA2-7B and surface debug logs from nested_llama.py
weight loading and generation.

This is adapted from compare_balcony.py, but it only runs the vattention-ee side
and is meant to help inspect loader/debug output.

Examples:
    python scripts/check_balcony_vattn_debug.py
    python scripts/check_balcony_vattn_debug.py --num_requests 3 --ee_policy off
    python scripts/check_balcony_vattn_debug.py --ee_policy eager --shallow_exit_layer 15 --conf_threshold 0.0
"""

import argparse
import io
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

LOAD_WEIGHTS_SUMMARY_RE = re.compile(
    r"\[load_weights\]\s+loaded=(\d+)\s+skipped=(\d+)\s+missing=(\d+)\s+remapped=(\d+)"
)
EXIT_LAYER_RE = re.compile(r"\[load_weights\]\s+exit layer\s+(\d+):\s+loaded\s+(\d+)\s+tensors")
OUTPUT_BLOCK_RE = re.compile(
    r"\[BenchmarkRunner\._run\] Output id (\d+) Finished=+\n(.*?)\n=+",
    re.DOTALL,
)


def run_vattn(args):
    run_ee = os.path.join(SCRIPTS_DIR, "run_ee.py")
    cmd = [
        sys.executable, run_ee,
        "--model", "balcony-llama-2-7b",
        "--ee_policy", args.ee_policy,
        "--num_requests", str(args.num_requests),
        "--max_batch_size", str(args.max_batch_size),
        "--qps", str(args.qps),
        "--shallow_exit_layer", str(args.shallow_exit_layer),
        "--conf_threshold", str(args.conf_threshold),
        "--kv_method", args.kv_method,
        "--dataset_name", args.dataset_name,
        "--csv_path", args.csv_path,
    ]

    if args.extra:
        cmd.extend(args.extra)

    print("[check] Running command:")
    print(" ", " ".join(cmd))
    print()

    buf = io.StringIO()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=SCRIPTS_DIR,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        buf.write(line)

    proc.wait()
    return proc.returncode, buf.getvalue()


def summarize_logs(log_text: str):
    summaries = LOAD_WEIGHTS_SUMMARY_RE.findall(log_text)
    exit_layers = EXIT_LAYER_RE.findall(log_text)
    outputs = OUTPUT_BLOCK_RE.findall(log_text)

    print("\n" + "=" * 88)
    print("DEBUG SUMMARY")
    print("=" * 88)

    if summaries:
        loaded, skipped, missing, remapped = summaries[-1]
        print(f"load_weights summary: loaded={loaded} skipped={skipped} missing={missing} remapped={remapped}")
    else:
        print("load_weights summary: NOT FOUND")

    if exit_layers:
        print("exit-layer coverage:")
        for layer, count in exit_layers:
            print(f"  layer {layer}: {count} tensors loaded")
    else:
        print("exit-layer coverage: NOT FOUND")

    print(f"captured output blocks: {len(outputs)}")

    first_missing_idx = log_text.find("[load_weights] first 30 missing:")
    if first_missing_idx != -1:
        snippet = log_text[first_missing_idx:first_missing_idx + 2500]
        print("\nfirst missing/remap snippet:")
        print(snippet.rstrip())
    else:
        print("\nfirst missing/remap snippet: NOT FOUND")

    if outputs:
        print("\nfirst 3 generated outputs:")
        for out_id, text in outputs[:3]:
            print("-" * 88)
            print(f"Output id {out_id}")
            print(text.strip()[:1200])


def main():
    parser = argparse.ArgumentParser(description="Run Balcony in vattention-ee and surface debug logs")
    parser.add_argument("--num_requests", type=int, default=5)
    parser.add_argument("--ee_policy", type=str, default="off")
    parser.add_argument("--shallow_exit_layer", type=int, default=15)
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--kv_method", type=str, default="copy")
    parser.add_argument("--dataset_name", type=str, default="xsum")
    parser.add_argument("--csv_path", type=str, default="/tmp/balcony_debug/")
    parser.add_argument("--save_log", type=str, default="")
    parser.add_argument("extra", nargs="*", help="Extra args appended to run_ee.py command")
    args = parser.parse_args()

    Path(args.csv_path).mkdir(parents=True, exist_ok=True)

    rc, log_text = run_vattn(args)
    summarize_logs(log_text)

    if args.save_log:
        out = Path(args.save_log)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(log_text)
        print(f"\nSaved full log to: {out}")

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
