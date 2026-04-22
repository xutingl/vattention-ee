#!/usr/bin/env python3
"""Parse sweep_policy_conf.sh logs and compare confidence vs throughput.

Reads every *.log under OUT_DIR, extracts key metrics from the benchmark_runner
stdout, writes summary.csv, and renders a two-panel scatter plot:
  (a) throughput vs avg confidence of early-exited tokens
  (b) throughput vs RougeL against XSUM references

Run after sweep_policy_conf.sh completes.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

OUT_DIR = Path(os.environ.get("OUT_DIR", "/workspace/vattention-ee/outputs_balcony/sweep"))

METRIC_PATTERNS: dict[str, str] = {
    "throughput": r"Throughput: ([\d.]+) tokens/sec",
    "rouge": r"RougeL: ([\d.eE+-]+)",
    "bert": r"Bert_score: ([\d.eE+-]+)",
    "avg_conf": r"Avg conf_score: ([\d.eE+-]+)",
    "avg_conf_ee": r"Avg conf_score ee: ([\d.eE+-]+)",
    "avg_conf_non_ee": r"Avg conf_score non_ee: ([\d.eE+-]+)",
    "median_conf_ee": r"Median conf_score ee: ([\d.eE+-]+|nan)",
    "tbt_avg": r"TBT avg: ([\d.eE+-]+)",
    "tbt_p99": r"TBT p99: ([\d.eE+-]+)",
    "would_ee_stay": r"Num seq would ee but stay: (\d+)",
    "would_not_ee_ee": r"Num seq would not ee but ee: (\d+)",
}
EE_RATES_RE = re.compile(r"Exited rates.*\[(\d+),\s*(\d+)\]")
TAG_RE = re.compile(r"^([a-z][-a-z]*)_layer(\d+)_conf([\d.]+)\.log$")

POLICY_COLORS = {
    "off": "black",
    "eager": "tab:blue",
    "lazy": "tab:orange",
    "median": "tab:green",
    "latency-only": "tab:red",
    "rebatching": "tab:purple",
}
LAYER_MARKERS = {15: "o", 18: "s", 21: "^"}


def _parse_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_log(path: Path) -> dict | None:
    m = TAG_RE.match(path.name)
    if not m:
        return None
    row: dict = {
        "policy": m.group(1),
        "layer": int(m.group(2)),
        "conf": float(m.group(3)),
    }
    text = path.read_text(errors="replace")
    for key, pat in METRIC_PATTERNS.items():
        mm = re.search(pat, text)
        row[key] = _parse_float(mm.group(1)) if mm else None
    ee_match = EE_RATES_RE.search(text)
    if ee_match:
        ee, non_ee = int(ee_match.group(1)), int(ee_match.group(2))
        total = ee + non_ee
        row["ee_tokens"] = ee
        row["non_ee_tokens"] = non_ee
        row["ee_rate"] = ee / total if total else 0.0
    else:
        row["ee_tokens"] = row["non_ee_tokens"] = row["ee_rate"] = None
    if row["throughput"] is None:
        row["status"] = "failed"
    else:
        row["status"] = "ok"
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict]) -> None:
    hdr = f"{'policy':<13} {'layer':>5} {'conf':>5} {'thru':>8} {'ee%':>6} {'conf_ee':>8} {'conf_neg':>9} {'rouge':>7} {'bert':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["policy"], x["layer"], x["conf"])):
        thru = r.get("throughput") or 0
        ee_rate = (r.get("ee_rate") or 0) * 100
        ce = r.get("avg_conf_ee") or 0
        cn = r.get("avg_conf_non_ee") or 0
        rouge = r.get("rouge") or 0
        bert = r.get("bert") or 0
        flag = "" if r["status"] == "ok" else " FAIL"
        print(
            f"{r['policy']:<13} {r['layer']:>5} {r['conf']:>5.2f} "
            f"{thru:>8.2f} {ee_rate:>5.1f}% {ce:>8.4f} {cn:>9.4f} "
            f"{rouge:>7.4f} {bert:>7.4f}{flag}"
        )


def make_plot(rows: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib missing; install it to render the plot", file=sys.stderr)
        return False

    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("no successful runs to plot", file=sys.stderr)
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def scatter(ax, ykey, ylabel, title):
        for r in ok:
            x = r.get("throughput")
            y = r.get(ykey)
            if x is None or y is None:
                continue
            color = POLICY_COLORS.get(r["policy"], "gray")
            marker = LAYER_MARKERS.get(r["layer"], "x")
            size = 40 + r["conf"] * 300
            ax.scatter(x, y, c=color, marker=marker, s=size,
                       alpha=0.75, edgecolors="k", linewidths=0.4)
        ax.set_xlabel("Throughput (tokens/sec)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    scatter(axes[0], "avg_conf_ee",
            "Avg confidence of EE-exited tokens",
            "Throughput vs EE-token confidence (quality proxy)")
    scatter(axes[1], "rouge",
            "RougeL vs XSUM reference",
            "Throughput vs RougeL")

    policy_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markeredgecolor="k",
               markersize=10, label=p)
        for p, c in POLICY_COLORS.items()
    ]
    layer_handles = [
        Line2D([0], [0], marker=m, color="gray", markersize=10,
               linestyle="None", label=f"layer {L}")
        for L, m in LAYER_MARKERS.items()
    ]
    conf_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=6 + i * 2,
               linestyle="None", label=f"conf {c}")
        for i, c in enumerate([0.1, 0.2, 0.3])
    ]
    axes[0].legend(handles=policy_handles, loc="lower left",
                   fontsize=8, title="policy")
    axes[1].legend(handles=layer_handles + conf_handles, loc="lower right",
                   fontsize=8, ncol=2, title="layer / conf size")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")
    return True


def main() -> int:
    log_paths = sorted(OUT_DIR.glob("*.log"))
    if not log_paths:
        print(f"no *.log files under {OUT_DIR}", file=sys.stderr)
        return 1
    rows: list[dict] = []
    for p in log_paths:
        r = parse_log(p)
        if r is not None:
            rows.append(r)
    if not rows:
        print("no parseable log files", file=sys.stderr)
        return 1

    csv_path = OUT_DIR / "summary.csv"
    write_csv(rows, csv_path)
    print(f"parsed {len(rows)} runs -> {csv_path}")
    print()
    print_table(rows)
    print()
    make_plot(rows, OUT_DIR / "sweep_plot.png")
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} run(s) did not produce throughput:")
        for r in failed:
            print(f"  - {r['policy']}/layer{r['layer']}/conf{r['conf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
