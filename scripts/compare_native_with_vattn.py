#!/usr/bin/env python3
"""
Read native Balcony JSON output and an existing vattn-ee txt log, print side-by-side.

Usage:
    python compare_native_with_vattn.py --native OUT.json --vattn_log path/to/compare_...txt
"""

import argparse
import json
import re
import textwrap

_OUTPUT_BLOCK = re.compile(
    r"\[BenchmarkRunner\._run\] Output id (\d+) Finished=+\n(.*?)\n=+",
    re.DOTALL,
)


def parse_vattn_log(path: str) -> dict[int, str]:
    with open(path) as f:
        text = f.read()
    return {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_BLOCK.finditer(text)}


def print_comparison(native_results, vattn_map, width=88):
    sep = "─" * width
    for i, row in enumerate(native_results):
        seq_id = i + 1
        vattn_text = vattn_map.get(seq_id, "<not found in log>")
        print(f"\n{'═'*width}")
        print(f"  Sample {i+1}/{len(native_results)}")
        print(f"{'═'*width}")
        print("REFERENCE:")
        print(textwrap.fill(row["reference"], width=width, initial_indent="  ", subsequent_indent="  "))
        print(sep)
        print("NATIVE BALCONY  (transformers_extra, balcony env):")
        print(textwrap.fill(row["output"] or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
        print(sep)
        print("VATTN-EE        (ee_policy=eager, exit at layer 15):")
        print(textwrap.fill(vattn_text or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
    print(f"\n{'═'*width}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native",   required=True, help="JSON file from run_native_balcony.py")
    parser.add_argument("--vattn_log", required=True, help="txt log from compare_balcony.py / run_compare_balcony.sh")
    args = parser.parse_args()

    with open(args.native) as f:
        native_results = json.load(f)

    vattn_map = parse_vattn_log(args.vattn_log)
    if not vattn_map:
        print(f"WARNING: no Output blocks found in {args.vattn_log}")

    print_comparison(native_results, vattn_map)


if __name__ == "__main__":
    main()
