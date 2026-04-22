#!/usr/bin/env python3
"""
Run a small benchmark matrix for Balcony on vattention-ee and optionally compare
against native Balcony in the separate `balcony` conda environment.

Matrix:
- vattn off
- vattn eager @ 15
- vattn eager @ 18
- vattn eager @ 21
- optional native full
- optional native exit @ 15/18/21

This is intended for the next validation step after integration:
1) verify output semantics still match native Balcony
2) compare quality / throughput across exit depths

Examples:
  python scripts/benchmark_balcony_matrix.py
  python scripts/benchmark_balcony_matrix.py --num_requests 20 --include_native
  python scripts/benchmark_balcony_matrix.py --num_requests 50 --max_batch_size 1
  python scripts/benchmark_balcony_matrix.py --num_requests 50 --max_batch_size 4 --qps 20
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
from statistics import mean

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)

OUTPUT_BLOCK_RE = re.compile(
    r"\[BenchmarkRunner\._run\] Output id (\d+) Finished=+\n(.*?)\n=+",
    re.DOTALL,
)
LOAD_SUMMARY_RE = re.compile(
    r"\[load_weights\]\s+loaded=(\d+)\s+skipped=(\d+)\s+missing=(\d+)\s+remapped=(\d+)"
)
METRIC_RE = {
    "throughput": re.compile(r"Throughput:\s+([0-9.]+)\s+tokens/sec"),
    "rougeL": re.compile(r"RougeL:\s+([0-9.]+),\s+Bert_score:\s+([0-9.]+)"),
    "tbt": re.compile(r"TBT avg:\s+([0-9.]+),\s+TBT p95:\s+([0-9.]+),\s+TBT p99:\s+([0-9.]+)"),
    "request_duration": re.compile(r"Request duration avg:\s+([0-9.]+),\s+Request duration p95:\s+([0-9.]+),\s+Request duration p99:\s+([0-9.]+)"),
}


def run_cmd(cmd, cwd):
    print("[run]", " ".join(cmd))
    buf = io.StringIO()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        buf.write(line)
    proc.wait()
    return proc.returncode, buf.getvalue()


def parse_vattn_log(log_text, num_requests):
    outputs = {}
    for m in OUTPUT_BLOCK_RE.finditer(log_text):
        outputs[int(m.group(1))] = m.group(2).strip()

    load_summary = None
    load_matches = LOAD_SUMMARY_RE.findall(log_text)
    if load_matches:
        load_summary = tuple(int(x) for x in load_matches[-1])

    metrics = {}
    m = METRIC_RE["throughput"].search(log_text)
    if m:
        metrics["throughput"] = float(m.group(1))

    m = METRIC_RE["rougeL"].search(log_text)
    if m:
        metrics["rougeL"] = float(m.group(1))
        metrics["bert_score"] = float(m.group(2))

    m = METRIC_RE["tbt"].search(log_text)
    if m:
        metrics["tbt_avg"] = float(m.group(1))
        metrics["tbt_p95"] = float(m.group(2))
        metrics["tbt_p99"] = float(m.group(3))

    m = METRIC_RE["request_duration"].search(log_text)
    if m:
        metrics["req_avg"] = float(m.group(1))
        metrics["req_p95"] = float(m.group(2))
        metrics["req_p99"] = float(m.group(3))

    ordered_outputs = [outputs.get(i + 1, "<not captured>") for i in range(num_requests)]
    return {"outputs": ordered_outputs, "load_summary": load_summary, "metrics": metrics, "log": log_text}


def run_vattn_case(name, num_requests, ee_policy, exit_layer, conf_threshold, max_batch_size, qps, kv_method, dataset_name, csv_path):
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
    rc, log_text = run_cmd(cmd, SCRIPTS_DIR)
    if rc != 0:
        raise RuntimeError(f"{name} failed with return code {rc}")
    return parse_vattn_log(log_text, num_requests)


def run_native_case(num_samples, exit_layer, max_new_tokens, output_json, no_exit):
    native_script = os.path.join(SCRIPTS_DIR, "run_native_balcony.py")
    cmd = [
        "conda", "run", "-n", "balcony", "--no-capture-output",
        "python", native_script,
        "--num_samples", str(num_samples),
        "--exit_layer", str(exit_layer),
        "--max_new_tokens", str(max_new_tokens),
        "--output", str(output_json),
    ]
    if no_exit:
        cmd.append("--no_exit")
    rc, log_text = run_cmd(cmd, SCRIPTS_DIR)
    if rc != 0:
        raise RuntimeError(f"native run failed with return code {rc}")
    data = json.loads(Path(output_json).read_text())
    return {"outputs": [x["output"] for x in data], "log": log_text}


def print_metrics_table(results):
    print("\n" + "=" * 110)
    print("SUMMARY METRICS")
    print("=" * 110)
    header = f"{'case':<18} {'throughput':>11} {'rougeL':>9} {'bert':>9} {'tbt_avg':>10} {'req_avg':>10}"
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        m = result.get("metrics", {})
        print(
            f"{name:<18} "
            f"{m.get('throughput', float('nan')):>11.2f} "
            f"{m.get('rougeL', float('nan')):>9.4f} "
            f"{m.get('bert_score', float('nan')):>9.4f} "
            f"{m.get('tbt_avg', float('nan')):>10.4f} "
            f"{m.get('req_avg', float('nan')):>10.4f}"
        )


def print_output_comparison(results, native_results=None, width=100):
    names = list(results.keys())
    n = len(next(iter(results.values()))["outputs"])
    for i in range(n):
        print("\n" + "═" * width)
        print(f"  Sample {i+1}/{n}")
        print("═" * width)
        for name in names:
            print(f"{name}:")
            print(textwrap.fill(results[name]["outputs"][i] or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
            print("─" * width)
        if native_results:
            for name, result in native_results.items():
                print(f"{name}:")
                print(textwrap.fill(result["outputs"][i] or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
                print("─" * width)


def save_artifacts(outdir, results, native_results=None):
    outdir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        (outdir / f"{name}.log").write_text(result["log"])
        (outdir / f"{name}_outputs.json").write_text(json.dumps(result["outputs"], indent=2, ensure_ascii=False))
    if native_results:
        for name, result in native_results.items():
            (outdir / f"{name}.log").write_text(result["log"])
            (outdir / f"{name}_outputs.json").write_text(json.dumps(result["outputs"], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Balcony matrix on vattention-ee")
    parser.add_argument("--num_requests", type=int, default=20)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--kv_method", type=str, default="copy")
    parser.add_argument("--dataset_name", type=str, default="xsum")
    parser.add_argument("--csv_path", type=str, default="/tmp/balcony_matrix/")
    parser.add_argument("--include_native", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--save_dir", type=str, default="/tmp/balcony_matrix_results")
    args = parser.parse_args()

    Path(args.csv_path).mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Recommendation:
    # - keep batch=1 for semantic verification / apples-to-apples comparison
    # - use larger batch sizes later for throughput experiments

    vattn_results = {}
    vattn_results["vattn_off"] = run_vattn_case(
        name="vattn_off",
        num_requests=args.num_requests,
        ee_policy="off",
        exit_layer=15,
        conf_threshold=0.25,
        max_batch_size=args.max_batch_size,
        qps=args.qps,
        kv_method=args.kv_method,
        dataset_name=args.dataset_name,
        csv_path=args.csv_path,
    )

    for layer in [15, 18, 21]:
        vattn_results[f"vattn_eager_{layer}"] = run_vattn_case(
            name=f"vattn_eager_{layer}",
            num_requests=args.num_requests,
            ee_policy="eager",
            exit_layer=layer,
            conf_threshold=0.0,
            max_batch_size=args.max_batch_size,
            qps=args.qps,
            kv_method=args.kv_method,
            dataset_name=args.dataset_name,
            csv_path=args.csv_path,
        )

    native_results = None
    if args.include_native:
        native_results = {}
        native_results["native_full"] = run_native_case(
            num_samples=args.num_requests,
            exit_layer=15,
            max_new_tokens=args.max_new_tokens,
            output_json=save_dir / f"native_full_n{args.num_requests}.json",
            no_exit=True,
        )
        for layer in [15, 18, 21]:
            native_results[f"native_exit_{layer}"] = run_native_case(
                num_samples=args.num_requests,
                exit_layer=layer,
                max_new_tokens=args.max_new_tokens,
                output_json=save_dir / f"native_exit{layer}_n{args.num_requests}.json",
                no_exit=False,
            )

    print_metrics_table(vattn_results)
    print_output_comparison(vattn_results, native_results=native_results)
    save_artifacts(save_dir, vattn_results, native_results=native_results)
    print(f"\nSaved logs and outputs to: {save_dir}")


if __name__ == "__main__":
    main()
