#!/usr/bin/env python3
"""
Compare native Balcony model output (via transformers_extra) against
vattention-ee running with ee_policy=off on the same XSUM articles.

Usage:
    python scripts/compare_balcony.py [--num_samples N] [--exit_layer L] [--max_new_tokens T]
    python scripts/compare_balcony.py --skip_native   # skip transformers_extra side
"""

import argparse
import io
import os
import re
import subprocess
import sys
import textwrap

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
MODEL_ID = "parsakaveh/Balcony-LLaMA2-7B"
CACHE_DIR = "/workspace/downloaded_models/"
MAX_ARTICLE_CHARS = 2000
BALCONY_REPO = "/workspace/Balcony-LLaMA/finetuning"  # contains transformers_extra/

# Make transformers_extra importable without installing anything
if BALCONY_REPO not in sys.path:
    sys.path.insert(0, BALCONY_REPO)


# ── dataset ──────────────────────────────────────────────────────────────────

def load_xsum_samples(n: int):
    from datasets import load_dataset
    ds = load_dataset("EdinburghNLP/xsum", split="validation")
    filtered = [i for i in range(len(ds)) if len(ds[i]["document"]) <= MAX_ARTICLE_CHARS]
    samples = []
    for raw_idx in filtered[:n]:
        doc = ds[raw_idx]["document"]
        prompt = f"Article: {doc}. Summarize the article in one sentence. Summary:"
        samples.append({
            "prompt": prompt,
            "reference": ds[raw_idx]["summary"],
        })
    print(f"Loaded {len(samples)} XSUM samples.")
    return samples


# ── native Balcony via transformers_extra ────────────────────────────────────

def run_native_balcony(samples, exit_layer: int, max_new_tokens: int):
    try:
        # transformers_extra requires FlashAttentionKwargs (added in 4.46+).
        # Stub it out so it works with the vattn env's older transformers.
        import transformers.modeling_flash_attention_utils as _fa_utils
        if not hasattr(_fa_utils, "FlashAttentionKwargs"):
            from typing import TypedDict
            class FlashAttentionKwargs(TypedDict, total=False):
                pass
            _fa_utils.FlashAttentionKwargs = FlashAttentionKwargs

        import importlib
        importlib.import_module("transformers_extra")  # registers NestedLlamaForCausalLM as side-effect
    except Exception as e:
        print(f"[native] ERROR: could not load transformers_extra: {e}")
        return [f"<transformers_extra load failed: {e}>"] * len(samples)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[native] Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, use_fast=False,
    )

    print(f"[native] Loading model with output_exit_layers={exit_layer} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        output_exit_layers=exit_layer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    outputs = []
    for i, s in enumerate(samples):
        enc = tok(
            s["prompt"], return_tensors="pt",
            truncation=True, max_length=400,
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
            )
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text.strip())
        print(f"[native] {i+1}/{len(samples)} done.")

    return outputs


# ── vattention-ee with ee_policy=eager at exit_layer ─────────────────────────

_OUTPUT_BLOCK = re.compile(
    r"\[BenchmarkRunner\._run\] Output id (\d+) Finished=+\n(.*?)\n=+",
    re.DOTALL,
)


def run_vattn_off(n: int, exit_layer: int):
    run_ee = os.path.join(SCRIPTS_DIR, "run_ee.py")
    cmd = [
        sys.executable, run_ee,
        "--model",              "balcony-llama-2-7b",
        "--ee_policy",          "eager",    # exit at shallow_exit_layer every token
        "--num_requests",       str(n),
        "--max_batch_size",     "1",        # sequential → IDs match article order
        "--qps",                "10",       # fast arrivals, no gaps
        "--shallow_exit_layer", str(exit_layer),
        "--conf_threshold",     "0.0",      # force exit at layer 15 for every token
        "--kv_method",          "copy",
        "--dataset_name",       "xsum",
        "--csv_path",           "/tmp/balcony_compare/",
    ]
    print("[vattn-ee] Running run_ee.py (initialises Ray + loads weights, may take a minute) ...")
    print("[vattn-ee] Command:", " ".join(cmd))

    # Stream output live to terminal while also capturing it for parsing.
    buf = io.StringIO()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=SCRIPTS_DIR,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        buf.write(line)
    proc.wait()
    combined = buf.getvalue()

    id_to_text: dict[int, str] = {}
    for m in _OUTPUT_BLOCK.finditer(combined):
        id_to_text[int(m.group(1))] = m.group(2).strip()

    if not id_to_text:
        print("[vattn-ee] WARNING: no output blocks captured — check the log above.")

    # Sequence IDs are assigned in arrival order starting at 1.
    return [id_to_text.get(i + 1, "<not captured>") for i in range(n)]


# ── comparison printer ────────────────────────────────────────────────────────

def print_comparison(samples, native_outs, vattn_outs, exit_layer: int, width=88):
    sep = "─" * width
    for i, (s, native, vattn) in enumerate(zip(samples, native_outs, vattn_outs)):
        print(f"\n{'═'*width}")
        print(f"  Sample {i+1}/{len(samples)}")
        print(f"{'═'*width}")
        print("REFERENCE:")
        print(textwrap.fill(s["reference"], width=width, initial_indent="  ", subsequent_indent="  "))
        print(sep)
        print(f"NATIVE BALCONY  (transformers_extra, exit_layer={exit_layer}):")
        print(textwrap.fill(native or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
        print(sep)
        print(f"VATTN-EE        (ee_policy=eager, exit at layer {exit_layer}):")
        print(textwrap.fill(vattn or "<empty>", width=width, initial_indent="  ", subsequent_indent="  "))
    print(f"\n{'═'*width}\n")


# ── main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Compare native Balcony vs vattention-ee off-policy")
parser.add_argument("--num_samples",    type=int,   default=5,   help="Number of XSUM articles to compare")
parser.add_argument("--exit_layer",     type=int,   default=15,  help="Balcony exit layer index (native model)")
parser.add_argument("--max_new_tokens", type=int,   default=80,  help="Max tokens generated by native model")
parser.add_argument("--skip_native",    action="store_true",     help="Skip the transformers_extra run")
args = parser.parse_args()

samples = load_xsum_samples(args.num_samples)

if args.skip_native:
    native_outputs = ["<skipped>"] * len(samples)
else:
    native_outputs = run_native_balcony(samples, args.exit_layer, args.max_new_tokens)

vattn_outputs = run_vattn_off(len(samples), args.exit_layer)

print_comparison(samples, native_outputs, vattn_outputs, args.exit_layer)
