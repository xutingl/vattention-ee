#!/usr/bin/env python3
"""
Run vattention-ee twice on Balcony-LLaMA2-7B:
  1) ee_policy=off
  2) ee_policy=eager at layer 15 (or a chosen layer)

Then compare the generated outputs side by side.

This is adapted from compare_balcony.py, but compares two vattention-ee runs
instead of native Balcony vs vattention-ee.

Examples:
  python scripts/compare_vattn_off_vs_eager.py
  python scripts/compare_vattn_off_vs_eager.py --num_requests 5 --exit_layer 15
  python scripts/compare_vattn_off_vs_eager.py --save_log_dir /tmp/vattn_compare_logs
"""

import argparse
import io
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

_OUTPUT_BLOCK = re.compile(
    r"\[BenchmarkRunner\._run\] Output id (\d+) Finished=+\n(.*?)\n=+",
    re.DOTALL,
)
_LOAD_SUMMARY = re.compile(
    r"\[load_weights\]\s+loaded=(\d+)\s+skipped=(\d+)\s+missing=(\d+)\s+remapped=(\d+)"
)


def run_vattn(
    *,
    num_requests: int,
    ee_policy: str,
    exit_layer: int,
    conf_threshold: float,
    max_batch_size: int,
    qps: float,
    kv_method: str,
    dataset_name: str,
    csv_path: str,
):
    run_ee = os.path.join(SCRIPTS_DIR, "run_ee.py")
    cmd = [
        sys.executable, run_ee,
        "--model", "balcony-llama-2-7b",
        "--ee_policy", ee_policy,
        "--num_requests", str(num_requests),
        "--max_batch_size", str(max_batch_size),
        "--qps", str(qps),
        "--shallow_exit_layer", str(exit_layer),
        "--conf_threshold", str(conf_threshold),
        "--kv_method", kv_method,
        "--dataset_name", dataset_name,
        "--csv_path", csv_path,
    ]

    print("[run] Command:")
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
    combined = buf.getvalue()

    outputs = {}
    for m in _OUTPUT_BLOCK.finditer(combined):
        outputs[int(m.group(1))] = m.group(2).strip()

    load_summary = None
    summaries = _LOAD_SUMMARY.findall(combined)
    if summaries:
        load_summary = summaries[-1]

    ordered_outputs = [outputs.get(i + 1, "<not captured>") for i in range(num_requests)]
    return proc.returncode, combined, ordered_outputs, load_summary


def print_comparison(off_outputs, eager_outputs, width=100):
    sep = "─" * width
    for i, (off_text, eager_text) in enumerate(zip(off_outputs, eager_outputs), start=1):
        print(f"\n{'═'*width}")
        print(f"  Sample {i}/{len(off_outputs)}")
        print(f"{'═'*width}")
        print("VATTN-EE OFF:")
        print(textwrap.fill(off_text or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
        print(sep)
        print("VATTN-EE EAGER:")
        print(textwrap.fill(eager_text or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
    print(f"\n{'═'*width}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare vattention-ee off vs eager(layer N)")
    parser.add_argument("--num_requests", type=int, default=5)
    parser.add_argument("--exit_layer", type=int, default=15)
    parser.add_argument("--eager_conf_threshold", type=float, default=0.0,
                        help="Use 0.0 to force eager exit as often as possible")
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--kv_method", type=str, default="copy")
    parser.add_argument("--dataset_name", type=str, default="xsum")
    parser.add_argument("--csv_path", type=str, default="/tmp/balcony_compare_vattn/")
    parser.add_argument("--save_log_dir", type=str, default="")
    args = parser.parse_args()

    Path(args.csv_path).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("RUN 1/2: ee_policy=off")
    print("=" * 100)
    off_rc, off_log, off_outputs, off_summary = run_vattn(
        num_requests=args.num_requests,
        ee_policy="off",
        exit_layer=args.exit_layer,
        conf_threshold=0.25,
        max_batch_size=args.max_batch_size,
        qps=args.qps,
        kv_method=args.kv_method,
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
    )

    print("\n" + "=" * 100)
    print(f"RUN 2/2: ee_policy=eager at layer {args.exit_layer}")
    print("=" * 100)
    eager_rc, eager_log, eager_outputs, eager_summary = run_vattn(
        num_requests=args.num_requests,
        ee_policy="eager",
        exit_layer=args.exit_layer,
        conf_threshold=args.eager_conf_threshold,
        max_batch_size=args.max_batch_size,
        qps=args.qps,
        kv_method=args.kv_method,
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
    )

    print("\n" + "=" * 100)
    print("LOAD WEIGHTS SUMMARIES")
    print("=" * 100)
    print("OFF  :", off_summary if off_summary else "NOT FOUND")
    print("EAGER:", eager_summary if eager_summary else "NOT FOUND")

    print("\n" + "=" * 100)
    print("OUTPUT COMPARISON")
    print("=" * 100)
    print_comparison(off_outputs, eager_outputs)

    if args.save_log_dir:
        outdir = Path(args.save_log_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "off.log").write_text(off_log)
        (outdir / "eager.log").write_text(eager_log)
        (outdir / "off_outputs.txt").write_text("\n\n".join(off_outputs))
        (outdir / "eager_outputs.txt").write_text("\n\n".join(eager_outputs))
        print(f"Saved logs and outputs to: {outdir}")

    if off_rc != 0:
        raise SystemExit(off_rc)
    if eager_rc != 0:
        raise SystemExit(eager_rc)


if __name__ == "__main__":
    main()
