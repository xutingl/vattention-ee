#!/usr/bin/env python3
"""
Builds 10 (input prompt, output text) pairs for each EE policy.
Loads prompts from XSUM dataset (same format as RealRequestGenerator).
Reads outputs from CSVs produced by run_ee.py.
"""

import argparse
import json
from pathlib import Path


def load_xsum_prompts(num_requests: int, dataset_name: str = "xsum", max_article_length: int = 2000):
    """Load prompts using same logic as RealRequestGenerator."""
    from datasets import load_dataset

    dataset = load_dataset("EdinburghNLP/xsum", split="validation")
    filtered_indices = [
        i for i in range(len(dataset))
        if len(dataset[i]["document"]) <= max_article_length
    ]
    num_available = min(num_requests, len(filtered_indices))
    prompts = []
    for i in range(num_available):
        original_idx = filtered_indices[i]
        prompt = "Article: " + dataset[original_idx]["document"] + ". Summarize the article in one sentence. Summary:"
        prompts.append(prompt)
    return prompts


def load_references(num_requests: int, dataset_name: str = "xsum", max_article_length: int = 2000):
    """Load reference summaries."""
    from datasets import load_dataset

    dataset = load_dataset("EdinburghNLP/xsum", split="validation")
    filtered_indices = [
        i for i in range(len(dataset))
        if len(dataset[i]["document"]) <= max_article_length
    ]
    num_available = min(num_requests, len(filtered_indices))
    return [dataset[filtered_indices[i]]["summary"] for i in range(num_available)]


def _clean_output_text(text: str) -> str:
    """Remove CSV escaping from output text."""
    if not isinstance(text, str):
        return str(text)
    text = text.strip()
    # Remove surrounding escaped quotes: \"...\" or """..."
    while text.startswith('"') or text.startswith('\\"'):
        text = text[1:]
    while text.endswith('"') or text.endswith('\\"'):
        text = text[:-1]
    return text.replace('""', '"').replace('\\"', '"')


def parse_csv_output(csv_path: Path, num_requests: int):
    """Parse CSV to get seq_id -> output text mapping. Returns dict[int, str]."""
    import pandas as pd

    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, escapechar="\\")
    except Exception:
        df = pd.read_csv(csv_path)
    if "seq_id" not in df.columns or "output" not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        sid = int(row["seq_id"])
        if 1 <= sid <= num_requests:
            result[sid] = _clean_output_text(row["output"])
    return result


def find_csv_for_policy(csv_path: Path, policy: str, num_requests: int, shallow_exit_layer: int, conf_threshold: float):
    """Find CSV file for given policy. Handles conf_threshold 0 vs 0.0 naming."""
    base = f"req_{num_requests}_batch_8_layer_{shallow_exit_layer}_conf_{conf_threshold}_{policy}"
    # Try exact match first
    p = csv_path / f"{base}_copy.csv"
    if p.exists():
        return p
    p = csv_path / f"{base}.csv"
    if p.exists():
        return p
    # Try 0.0 for conf
    base0 = f"req_{num_requests}_batch_8_layer_{shallow_exit_layer}_conf_0.0_{policy}"
    p = csv_path / f"{base0}_copy.csv"
    if p.exists():
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_requests", type=int, default=10)
    ap.add_argument("--dataset_name", type=str, default="xsum")
    ap.add_argument("--shallow_exit_layer", type=int, default=25)
    ap.add_argument("--conf_threshold", type=float, default=0.7)
    ap.add_argument("--policies", type=str, default="rebatching median lazy eager latency-only off")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    policies = args.policies.split()

    # Load prompts (indices 0..num_requests-1)
    prompts = load_xsum_prompts(args.num_requests, args.dataset_name)
    references = load_references(args.num_requests, args.dataset_name)

    # Build per-policy data
    data = {
        "num_requests": args.num_requests,
        "dataset": args.dataset_name,
        "shallow_exit_layer": args.shallow_exit_layer,
        "conf_threshold": args.conf_threshold,
        "policies": {},
    }

    for policy in policies:
        csv_file = find_csv_for_policy(
            csv_path, policy, args.num_requests,
            args.shallow_exit_layer, args.conf_threshold
        )
        if not csv_file:
            data["policies"][policy] = {"error": f"CSV not found for policy {policy}"}
            continue

        outputs = parse_csv_output(csv_file, args.num_requests)
        pairs = []
        for i in range(min(args.num_requests, len(prompts))):
            seq_id = i + 1
            prompt = prompts[i]
            output = outputs.get(seq_id, "")
            ref = references[i] if i < len(references) else ""
            pairs.append({
                "index": i,
                "seq_id": seq_id,
                "prompt": prompt,
                "output": output,
                "reference": ref,
            })
        data["policies"][policy] = {"pairs": pairs}

    # Write JSON
    out_json = output_dir / f"prompt_output_pairs_{args.num_requests}.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    # Write human-readable TXT
    out_txt = output_dir / f"prompt_output_pairs_{args.num_requests}.txt"
    with open(out_txt, "w") as f:
        f.write(f"Prompt-Output pairs ({args.num_requests} per policy)\n")
        f.write(f"Dataset: {args.dataset_name}, layer={args.shallow_exit_layer}, conf={args.conf_threshold}\n")
        f.write("=" * 80 + "\n\n")
        for policy in policies:
            if policy not in data["policies"] or "error" in data["policies"][policy]:
                f.write(f"\n### Policy: {policy}\n{data['policies'].get(policy, {}).get('error', 'N/A')}\n\n")
                continue
            f.write(f"\n### Policy: {policy}\n")
            f.write("-" * 60 + "\n")
            for p in data["policies"][policy]["pairs"]:
                f.write(f"\n--- Sample {p['index'] + 1} (seq_id={p['seq_id']}) ---\n\n")
                f.write("PROMPT:\n")
                # Truncate long prompts for readability
                prompt_preview = p["prompt"][:500] + "..." if len(p["prompt"]) > 500 else p["prompt"]
                f.write(prompt_preview + "\n\n")
                f.write("OUTPUT:\n")
                f.write((p["output"][:800] + "..." if len(p["output"]) > 800 else p["output"]) + "\n\n")
                f.write("REFERENCE:\n")
                f.write(p["reference"] + "\n\n")
            f.write("\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
