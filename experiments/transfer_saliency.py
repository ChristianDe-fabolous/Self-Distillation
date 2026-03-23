"""
Transfer Saliency Map via KL Divergence
========================================
Compares a base model against an SDFT-trained model on the science eval set.
Both models receive identical inputs. The per-token KL between their distributions
is a map of what SDFT training changed and where in the generation process.

Token positions are classified into three categories:
  - structural : XML tags (<reasoning>, <answer>, etc.)
  - reasoning  : logical connective words (therefore, because, thus, ...)
  - content    : everything else

The KL aggregated by category reveals *what kind* of thing SDFT internalizes:
  - High KL at reasoning tokens → SDFT changed how the model reasons
  - High KL at content tokens   → SDFT changed what the model knows
  - High KL at structural tokens → SDFT changed output formatting

Additionally reports accuracy of both models to check whether KL correlates
with correctness improvement.

Usage:
    python experiments/transfer_saliency.py \
        --base_model_path  <hf_id_or_path> \
        --sdft_model_path  <path_to_trained_checkpoint> \
        --n_samples 100 \
        --output_dir experiments/results/transfer_saliency
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── token classification ──────────────────────────────────────────────────────

STRUCTURAL_TOKENS = {
    "<reasoning>", "</reasoning>", "<answer>", "</answer>",
    "<think>", "</think>", "<step>", "</step>",
}

REASONING_WORDS = {
    "therefore", "because", "thus", "hence", "since", "consequently",
    "so", "implies", "means", "conclude", "follows", "given", "if",
    "then", "however", "although", "but", "whereas", "while",
    "assuming", "suppose", "contradiction", "proof", "qed",
}


def classify_token(token_str: str) -> str:
    """Classify a decoded token into structural / reasoning / content."""
    clean = token_str.strip().lower()
    if clean in STRUCTURAL_TOKENS or re.match(r"^</?[a-z_]+>$", clean):
        return "structural"
    if clean in REASONING_WORDS:
        return "reasoning"
    return "content"


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="HuggingFace ID or path to the base (untrained) model")
    parser.add_argument("--sdft_model_path", type=str, required=True,
                        help="Path to the SDFT-trained model checkpoint")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of eval questions to run (None = all 507)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str,
                        default="experiments/results/transfer_saliency")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_answer(text: str) -> str:
    if "<answer>" in text:
        text = text.split("<answer>")[-1].split("</answer>")[0]
    return text.strip().upper()[:1]


def format_prompt(tokenizer, messages: list) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── model inference ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate_completion(model, tokenizer, input_ids, max_new_tokens):
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy — deterministic baseline
        pad_token_id=tokenizer.eos_token_id,
    )
    completion_ids = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_ids, text


@torch.no_grad()
def get_logprobs(model, input_ids, completion_ids):
    """
    Return log-softmax distributions at each completion token position.
    Shape: [len(completion_ids), vocab_size]
    """
    full_ids = torch.cat([input_ids, completion_ids.unsqueeze(0)], dim=1)
    logits = model(full_ids).logits
    prompt_len = input_ids.shape[1]
    comp_logits = logits[0, prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
    return F.log_softmax(comp_logits, dim=-1)


def per_token_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """
    KL(P || Q) at each token position.
    log_p, log_q: [seq_len, vocab_size]
    Returns: [seq_len]
    """
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # load tokenizer (assume same vocab for both models)
    print(f"Loading tokenizer from base model: {args.base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    print(f"Loading base model: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    device = next(base_model.parameters()).device

    print(f"Loading SDFT model: {args.sdft_model_path}")
    sdft_model = AutoModelForCausalLM.from_pretrained(
        args.sdft_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sdft_model.eval()

    print("Loading science eval dataset...")
    dataset = load_from_disk("data/science_data/eval_data")
    if args.n_samples:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))

    # accumulators per token category
    kl_by_category = {"structural": [], "reasoning": [], "content": []}
    results = []
    base_correct, sdft_correct = 0, 0

    for i, example in enumerate(dataset):
        prompt_str = format_prompt(tokenizer, example["prompt"])
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        gold = example["answer"].strip().upper()

        # generate one completion per model (greedy)
        base_ids,  base_text  = generate_completion(base_model,  tokenizer, input_ids, args.max_new_tokens)
        sdft_ids,  sdft_text  = generate_completion(sdft_model,  tokenizer, input_ids, args.max_new_tokens)

        base_answer = extract_answer(base_text)
        sdft_answer = extract_answer(sdft_text)
        base_correct += int(base_answer == gold)
        sdft_correct += int(sdft_answer == gold)

        # forward pass of BOTH models on BOTH completions
        # We use the base model's completion as the reference sequence
        # (the model being "nudged" is the base; we ask how SDFT would predict the same tokens)
        min_len = min(len(base_ids), len(sdft_ids))
        if min_len == 0:
            continue

        # use base completion as the shared token sequence
        shared_ids = base_ids[:min_len].to(device)

        log_p_base = get_logprobs(base_model, input_ids, shared_ids)   # [T, V]
        log_p_sdft = get_logprobs(sdft_model, input_ids, shared_ids)   # [T, V]

        # per-token KL(base || sdft): where did SDFT change predictions?
        token_kl = per_token_kl(log_p_base, log_p_sdft)  # [T]

        # classify each token and accumulate
        token_categories = []
        per_token_records = []
        for t in range(min_len):
            tok_str = tokenizer.decode([shared_ids[t].item()])
            cat = classify_token(tok_str)
            kl_val = token_kl[t].item()
            kl_by_category[cat].append(kl_val)
            token_categories.append(cat)
            per_token_records.append({
                "token": tok_str,
                "category": cat,
                "kl": kl_val,
            })

        results.append({
            "index": i,
            "gold": gold,
            "base_answer": base_answer,
            "sdft_answer": sdft_answer,
            "base_correct": base_answer == gold,
            "sdft_correct": sdft_answer == gold,
            "mean_kl": float(token_kl.mean().item()),
            "per_token": per_token_records,
        })

        if (i + 1) % 10 == 0:
            n = i + 1
            print(f"[{n}/{len(dataset)}]  "
                  f"base_acc={base_correct/n:.3f}  "
                  f"sdft_acc={sdft_correct/n:.3f}  "
                  f"mean_kl={np.mean([r['mean_kl'] for r in results]):.4f}")

    n = len(dataset)

    # aggregate KL by category
    kl_summary = {
        cat: {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std":  float(np.std(vals))  if vals else 0.0,
            "n_tokens": len(vals),
        }
        for cat, vals in kl_by_category.items()
    }

    summary = {
        "n_samples": n,
        "base_model": args.base_model_path,
        "sdft_model": args.sdft_model_path,
        "base_accuracy":  base_correct / n,
        "sdft_accuracy":  sdft_correct / n,
        "accuracy_delta": (sdft_correct - base_correct) / n,
        "kl_by_token_category": kl_summary,
        "overall_mean_kl": float(np.mean([r["mean_kl"] for r in results])),
    }

    print("\n" + "=" * 60)
    print("Transfer Saliency Results:")
    print(f"  Base accuracy : {summary['base_accuracy']:.4f}")
    print(f"  SDFT accuracy : {summary['sdft_accuracy']:.4f}")
    print(f"  Delta         : {summary['accuracy_delta']:+.4f}")
    print(f"\n  KL by token category:")
    for cat, stats in kl_summary.items():
        print(f"    {cat:12s}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  n={stats['n_tokens']}")
    print("=" * 60)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "per_sample.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
