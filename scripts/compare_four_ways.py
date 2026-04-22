#!/usr/bin/env python3
"""
Four-way comparison of Balcony-LLaMA2-7B inference configurations:
  1. vattention-ee  ee_policy=off       (full model, no early exit)
  2. vattention-ee  ee_policy=eager     (exit at --exit_layer every token)
  3. Native Balcony output_exit_layers=[]          (full model, no early exit)
  4. Native Balcony output_exit_layers=[exit_layer] (exit at --exit_layer)

Native Balcony runs are spawned inside the 'balcony' conda env via
`conda run -n balcony`, which requires conda to be on PATH.

Usage:
    python scripts/compare_four_ways.py
    python scripts/compare_four_ways.py --num_requests 5 --exit_layer 15 --max_new_tokens 80
    python scripts/compare_four_ways.py --skip_vattn   # skip vattn runs
    python scripts/compare_four_ways.py --skip_native  # skip native Balcony runs
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONDA_SH = "/workspace/miniconda3/etc/profile.d/conda.sh"

_OUTPUT_BLOCK = re.compile(
    r"\[BenchmarkRunner\._run\] Output id (\d+) Finished=+\n(.*?)\n=+",
    re.DOTALL,
)


# ── vattn runs ────────────────────────────────────────────────────────────────

def run_vattn(*, label: str, num_requests: int, ee_policy: str, exit_layer: int,
              conf_threshold: float, max_batch_size: int, qps: float,
              kv_method: str, dataset_name: str, csv_path: str) -> list[str]:
    run_ee = os.path.join(SCRIPTS_DIR, "run_ee.py")
    cmd = [
        sys.executable, run_ee,
        "--model",              "balcony-llama-2-7b",
        "--ee_policy",          ee_policy,
        "--num_requests",       str(num_requests),
        "--max_batch_size",     str(max_batch_size),
        "--qps",                str(qps),
        "--shallow_exit_layer", str(exit_layer),
        "--conf_threshold",     str(conf_threshold),
        "--kv_method",          kv_method,
        "--dataset_name",       dataset_name,
        "--csv_path",           csv_path,
    ]
    print(f"\n[{label}] Command: {' '.join(cmd)}\n")

    buf = io.StringIO()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=SCRIPTS_DIR, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        buf.write(line)
    proc.wait()
    if proc.returncode != 0:
        print(f"[{label}] WARNING: process exited with code {proc.returncode}")

    combined = buf.getvalue()
    outputs = {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_BLOCK.finditer(combined)}
    if not outputs:
        print(f"[{label}] WARNING: no output blocks captured.")
    return [outputs.get(i + 1, "<not captured>") for i in range(num_requests)]


# ── native Balcony runs ───────────────────────────────────────────────────────

def run_native_balcony(*, label: str, num_samples: int, exit_layer: int,
                       max_new_tokens: int, no_exit: bool, out_json: str) -> list[str]:
    run_script = os.path.join(SCRIPTS_DIR, "run_native_balcony.py")
    cmd_args = [
        "python", run_script,
        "--num_samples",    str(num_samples),
        "--exit_layer",     str(exit_layer),
        "--max_new_tokens", str(max_new_tokens),
        "--output",         out_json,
    ]
    if no_exit:
        cmd_args.append("--no_exit")

    # Wrap in `conda run -n balcony` so transformers_extra is available.
    cmd = ["conda", "run", "-n", "balcony", "--no-capture-output"] + cmd_args
    print(f"\n[{label}] Command: {' '.join(cmd)}\n")

    proc = subprocess.run(cmd, cwd=SCRIPTS_DIR)
    if proc.returncode != 0:
        print(f"[{label}] WARNING: process exited with code {proc.returncode}")

    if not os.path.exists(out_json):
        print(f"[{label}] WARNING: output JSON not found: {out_json}")
        return ["<native run failed>"] * num_samples

    with open(out_json) as f:
        results = json.load(f)
    return [r.get("output", "<empty>") for r in results]


# ── comparison printer ────────────────────────────────────────────────────────

def print_comparison(num_samples: int, exit_layer: int,
                     vattn_off: list[str],
                     vattn_eager: list[str],
                     native_full: list[str],
                     native_exit: list[str],
                     width: int = 100):
    sep = "─" * width
    labels = [
        f"VATTN-EE  off       (all layers)",
        f"VATTN-EE  eager     (exit at layer {exit_layer})",
        f"NATIVE    full      (all layers)",
        f"NATIVE    exit={exit_layer:<3}  (exit at layer {exit_layer})",
    ]
    cols = [vattn_off, vattn_eager, native_full, native_exit]

    for i in range(num_samples):
        print(f"\n{'═'*width}")
        print(f"  Sample {i+1}/{num_samples}")
        print(f"{'═'*width}")
        for label, col in zip(labels, cols):
            text = col[i] if i < len(col) else "<missing>"
            print(f"{label}:")
            print(textwrap.fill(text or "<empty>", width=width,
                                initial_indent="  ", subsequent_indent="  "))
            print(sep)
    print(f"\n{'═'*width}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Four-way Balcony inference comparison")
    parser.add_argument("--num_requests",   type=int,   default=5)
    parser.add_argument("--exit_layer",     type=int,   default=15)
    parser.add_argument("--max_new_tokens", type=int,   default=80,
                        help="Max tokens for native Balcony runs")
    parser.add_argument("--max_batch_size", type=int,   default=1)
    parser.add_argument("--qps",            type=float, default=10.0)
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                        help="Confidence threshold for vattn-ee eager (0.0 = always exit)")
    parser.add_argument("--kv_method",      type=str,   default="copy")
    parser.add_argument("--dataset_name",   type=str,   default="xsum")
    parser.add_argument("--csv_path",       type=str,   default="/tmp/four_way_compare/")
    parser.add_argument("--out_dir",        type=str,   default="/tmp/four_way_compare/")
    parser.add_argument("--skip_vattn",     action="store_true", help="Skip vattn-ee runs")
    parser.add_argument("--skip_native",    action="store_true", help="Skip native Balcony runs")
    args = parser.parse_args()

    Path(args.csv_path).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    n = args.num_requests

    # ── Run 1: vattn off ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RUN 1/4: vattention-ee  ee_policy=off  (full model)")
    print("=" * 100)
    if args.skip_vattn:
        vattn_off = ["<skipped>"] * n
    else:
        vattn_off = run_vattn(
            label="vattn-off",
            num_requests=n,
            ee_policy="off",
            exit_layer=args.exit_layer,
            conf_threshold=0.25,
            max_batch_size=args.max_batch_size,
            qps=args.qps,
            kv_method=args.kv_method,
            dataset_name=args.dataset_name,
            csv_path=args.csv_path,
        )

    # ── Run 2: vattn eager ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"RUN 2/4: vattention-ee  ee_policy=eager  (exit at layer {args.exit_layer})")
    print("=" * 100)
    if args.skip_vattn:
        vattn_eager = ["<skipped>"] * n
    else:
        vattn_eager = run_vattn(
            label="vattn-eager",
            num_requests=n,
            ee_policy="eager",
            exit_layer=args.exit_layer,
            conf_threshold=args.conf_threshold,
            max_batch_size=args.max_batch_size,
            qps=args.qps,
            kv_method=args.kv_method,
            dataset_name=args.dataset_name,
            csv_path=args.csv_path,
        )

    # ── Run 3: native Balcony full ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RUN 3/4: Native Balcony  output_exit_layers=[]  (full model)")
    print("=" * 100)
    native_full_json = os.path.join(args.out_dir, f"native_full_n{n}.json")
    if args.skip_native:
        native_full = ["<skipped>"] * n
    else:
        native_full = run_native_balcony(
            label="native-full",
            num_samples=n,
            exit_layer=args.exit_layer,
            max_new_tokens=args.max_new_tokens,
            no_exit=True,
            out_json=native_full_json,
        )

    # ── Run 4: native Balcony exit at layer ───────────────────────────────────
    print("\n" + "=" * 100)
    print(f"RUN 4/4: Native Balcony  output_exit_layers=[{args.exit_layer}]")
    print("=" * 100)
    native_exit_json = os.path.join(args.out_dir, f"native_exit{args.exit_layer}_n{n}.json")
    if args.skip_native:
        native_exit = ["<skipped>"] * n
    else:
        native_exit = run_native_balcony(
            label="native-exit",
            num_samples=n,
            exit_layer=args.exit_layer,
            max_new_tokens=args.max_new_tokens,
            no_exit=False,
            out_json=native_exit_json,
        )

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FOUR-WAY OUTPUT COMPARISON")
    print("=" * 100)
    print_comparison(n, args.exit_layer, vattn_off, vattn_eager, native_full, native_exit)


if __name__ == "__main__":
    main()
