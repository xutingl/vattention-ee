#!/usr/bin/env python3
"""
Run native Balcony inference via transformers_extra and write results as JSON.
Designed to run inside the 'balcony' conda env.

Usage:
    python run_native_balcony.py [--num_samples N] [--exit_layer L] [--max_new_tokens T] --output OUT.json
"""

import argparse
import json
import os
import sys

BALCONY_REPO = "/workspace/Balcony-LLaMA/finetuning"
if BALCONY_REPO not in sys.path:
    sys.path.insert(0, BALCONY_REPO)

MODEL_ID = "parsakaveh/Balcony-LLaMA2-7B"
CACHE_DIR = "/workspace/downloaded_models/"
MAX_ARTICLE_CHARS = 2000


def load_xsum_samples(n: int):
    from datasets import load_dataset
    ds = load_dataset("EdinburghNLP/xsum", split="validation")
    filtered = [i for i in range(len(ds)) if len(ds[i]["document"]) <= MAX_ARTICLE_CHARS]
    samples = []
    for raw_idx in filtered[:n]:
        doc = ds[raw_idx]["document"]
        samples.append({
            "prompt": f"Article: {doc}. Summarize the article in one sentence. Summary:",
            "reference": ds[raw_idx]["summary"],
        })
    print(f"[native] Loaded {len(samples)} XSUM samples.", flush=True)
    return samples


def run_inference(samples, exit_layer: int, max_new_tokens: int, no_exit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers_extra  # noqa: F401 — registers NestedLlamaForCausalLM

    print(f"[native] Loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, use_fast=False)

    exit_layers_arg = [] if no_exit else [exit_layer]
    print(f"[native] Loading model (output_exit_layers={exit_layers_arg}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        output_exit_layers=exit_layers_arg,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    eos_id = tok.eos_token_id

    outputs = []
    for i, s in enumerate(samples):
        enc = tok(
            s["prompt"], return_tensors="pt",
            truncation=True, max_length=400,
        ).to(model.device)

        generated = enc["input_ids"]
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(input_ids=generated)
                # Balcony returns exit logits as a tuple; handle both cases
                logits = out.logits if hasattr(out, "logits") else out[0]
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == eos_id:
                    break

        text = tok.decode(generated[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text.strip())
        print(f"[native] {i+1}/{len(samples)}: {text.strip()[:80]}", flush=True)

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples",    type=int, default=5)
    parser.add_argument("--exit_layer",     type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--no_exit",        action="store_true", help="Run all layers, no early exit")
    parser.add_argument("--output",         type=str, required=True, help="Path to write JSON results")
    args = parser.parse_args()

    samples = load_xsum_samples(args.num_samples)
    texts = run_inference(samples, args.exit_layer, args.max_new_tokens, no_exit=args.no_exit)

    results = [
        {"reference": s["reference"], "prompt": s["prompt"], "output": t}
        for s, t in zip(samples, texts)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[native] Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
