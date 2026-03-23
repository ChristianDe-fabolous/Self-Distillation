"""
Self-Checking Diagnostic via KL Divergence
===========================================
Generates N rollouts per science question, then compares three selection strategies:
  - Random selection
  - Majority voting (most common extracted answer)
  - KL centroid (rollout with minimum average KL to all others)

Beyond selection accuracy, the script separately measures KL and entropy in two
sequence regions per rollout:
  - reasoning region : tokens inside <reasoning>...</reasoning>
  - answer region    : tokens inside <answer>...</answer>

This tests the memorization hypothesis: if the model is pattern-matching rather
than genuinely reasoning, the answer tokens will be consistent across rollouts
(low KL, majority voting works) but the reasoning tokens will be high-entropy
and inconsistent (high KL) even on correctly answered questions.

If the model is genuinely reasoning, both regions should show low KL and low
entropy on correct answers, and high KL / high entropy on wrong ones.

Usage:
    python experiments/self_checking.py \
        --model_path <path_or_hf_id> \
        --n_rollouts 8 \
        --temperature 0.7 \
        --n_samples 100 \
        --output_dir experiments/results/self_checking
"""

import argparse
import json
import os
import re
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--n_rollouts", type=int, default=8,
                        help="Number of rollouts per question")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (must be > 0 for diverse rollouts)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of eval questions to run (None = all)")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/results/self_checking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run 2 questions with 2 rollouts to verify the pipeline works")
    return parser.parse_args()


# ── token region assignment ───────────────────────────────────────────────────

def assign_token_regions(completion_ids: torch.Tensor, tokenizer) -> list[str]:
    """
    Decode token by token, track cumulative text, assign each token to a region:
      'reasoning' — inside <reasoning>...</reasoning>
      'answer'    — inside <answer>...</answer>
      'other'     — outside both tags (structural markers, preamble)

    Returns a list of region labels, one per token.
    """
    regions = []
    cumulative = ""
    for tid in completion_ids.tolist():
        tok = tokenizer.decode([tid])
        cumulative += tok

        in_reasoning = bool(
            re.search(r"<reasoning>(?!.*</reasoning>)", cumulative, re.DOTALL)
            and not re.search(r"</reasoning>.*$", cumulative.split("<reasoning>")[-1], re.DOTALL)
        )
        in_answer = bool(
            re.search(r"<answer>(?!.*</answer>)", cumulative, re.DOTALL)
            and not re.search(r"</answer>.*$", cumulative.split("<answer>")[-1], re.DOTALL)
        )

        if in_answer:
            regions.append("answer")
        elif in_reasoning:
            regions.append("reasoning")
        else:
            regions.append("other")

    return regions


def region_stats(logprobs: torch.Tensor, regions: list[str], region: str) -> dict:
    """
    Given log-prob distributions [T, V] and region labels per token,
    return mean KL-to-uniform and mean entropy for tokens in the given region.
    Entropy: H = -sum p log p
    """
    mask = [i for i, r in enumerate(regions) if r == region]
    if not mask:
        return {"mean_entropy": None, "n_tokens": 0}
    lp = logprobs[mask]          # [n_region_tokens, vocab]
    entropy = -(lp.exp() * lp).sum(dim=-1)  # [n_region_tokens]
    return {
        "mean_entropy": float(entropy.mean().item()),
        "n_tokens": len(mask),
    }


# ── answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    if "<answer>" in text:
        text = text.split("<answer>")[-1].split("</answer>")[0]
    return text.strip().upper()[:1]


def format_prompt(tokenizer, messages: list) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_rollouts(model, tokenizer, input_ids, attention_mask, n_rollouts, temperature, max_new_tokens):
    rollouts = []
    for _ in range(n_rollouts):
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion_ids = output[0][input_ids.shape[1]:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        rollouts.append((completion_ids, text))
    return rollouts


# ── logit extraction ──────────────────────────────────────────────────────────

@torch.no_grad()
def get_logprobs(model, input_ids, completion_ids):
    """
    Forward pass on [prompt + completion].
    Returns log-softmax distributions at each completion token: [T, vocab_size]
    """
    full_ids = torch.cat([input_ids, completion_ids.unsqueeze(0)], dim=1)
    logits = model(full_ids).logits
    prompt_len = input_ids.shape[1]
    comp_logits = logits[0, prompt_len - 1 : prompt_len - 1 + len(completion_ids)]
    return F.log_softmax(comp_logits, dim=-1)


# ── KL computation ────────────────────────────────────────────────────────────

def pairwise_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """
    Symmetric KL over shared token positions (minimum length).
    KL_sym = 0.5 * [KL(P||Q) + KL(Q||P)]
    Neither rollout is a reference so we treat both directions equally.
    """
    min_len = min(len(log_p), len(log_q))
    if min_len == 0:
        return 0.0
    lp = log_p[:min_len]
    lq = log_q[:min_len]
    kl_pq = (lp.exp() * (lp - lq)).sum(dim=-1).mean().item()
    kl_qp = (lq.exp() * (lq - lp)).sum(dim=-1).mean().item()
    return 0.5 * (kl_pq + kl_qp)


def region_pairwise_kl(
    log_p: torch.Tensor, regions_p: list,
    log_q: torch.Tensor, regions_q: list,
    region: str,
) -> float:
    """
    Symmetric KL restricted to tokens in a given region for both rollouts.
    Uses minimum number of region tokens across the two rollouts.
    """
    idx_p = [i for i, r in enumerate(regions_p) if r == region]
    idx_q = [i for i, r in enumerate(regions_q) if r == region]
    min_len = min(len(idx_p), len(idx_q))
    if min_len == 0:
        return float("nan")
    lp = log_p[idx_p[:min_len]]
    lq = log_q[idx_q[:min_len]]
    kl_pq = (lp.exp() * (lp - lq)).sum(dim=-1).mean().item()
    kl_qp = (lq.exp() * (lq - lp)).sum(dim=-1).mean().item()
    return 0.5 * (kl_pq + kl_qp)


def kl_centroid_index(logprobs_list: list) -> tuple[int, list[float]]:
    """Return (centroid_index, avg_kl_per_rollout)."""
    n = len(logprobs_list)
    avg_kl = []
    for i in range(n):
        kls = [pairwise_kl(logprobs_list[i], logprobs_list[j]) for j in range(n) if j != i]
        avg_kl.append(float(np.mean(kls)))
    return int(np.argmin(avg_kl)), avg_kl


# ── selection strategies ──────────────────────────────────────────────────────

def select_random(answers: list, rng: random.Random) -> str:
    return rng.choice(answers)


def select_majority(answers: list) -> str:
    valid = [a for a in answers if a in "ABCD"]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def select_kl_centroid(logprobs_list: list, answers: list) -> tuple[str, int]:
    idx, _ = kl_centroid_index(logprobs_list)
    return answers[idx], idx


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    if args.smoke:
        args.n_samples = 2
        args.n_rollouts = 2
        args.max_new_tokens = 128
        print("*** SMOKE TEST — 2 questions, 2 rollouts ***")

    print("Loading science eval dataset...")
    dataset = load_from_disk("data/science_data/eval_data")
    if args.n_samples:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))

    correct_random, correct_majority, correct_kl = 0, 0, 0

    # accumulators for reasoning vs answer KL, split by correctness
    reasoning_kl_correct, reasoning_kl_wrong = [], []
    answer_kl_correct,    answer_kl_wrong    = [], []
    reasoning_entropy_correct, reasoning_entropy_wrong = [], []

    samples_path = os.path.join(args.output_dir, "samples.jsonl")
    samples_file = open(samples_path, "w")

    for i, example in enumerate(dataset):
        prompt_str = format_prompt(tokenizer, example["prompt"])
        encoded = tokenizer(prompt_str, return_tensors="pt").to(device)
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        gold = example["answer"].strip().upper()

        # generate rollouts
        rollouts = generate_rollouts(
            model, tokenizer, input_ids, attention_mask,
            args.n_rollouts, args.temperature, args.max_new_tokens
        )

        # logprobs + region labels per rollout
        answers, logprobs_list, regions_list = [], [], []
        for comp_ids, text in rollouts:
            answers.append(extract_answer(text))
            lp = get_logprobs(model, input_ids, comp_ids.to(device))
            logprobs_list.append(lp.cpu())
            regions_list.append(assign_token_regions(comp_ids.cpu(), tokenizer))

        # overall KL centroid
        centroid_idx, avg_kls = kl_centroid_index(logprobs_list)

        # per-region pairwise KL (averaged over all pairs)
        region_kl = {"reasoning": [], "answer": []}
        n_r = len(logprobs_list)
        for a in range(n_r):
            for b in range(a + 1, n_r):
                for reg in ("reasoning", "answer"):
                    val = region_pairwise_kl(
                        logprobs_list[a], regions_list[a],
                        logprobs_list[b], regions_list[b],
                        reg,
                    )
                    if not np.isnan(val):
                        region_kl[reg].append(val)

        mean_kl_reasoning = float(np.mean(region_kl["reasoning"])) if region_kl["reasoning"] else None
        mean_kl_answer    = float(np.mean(region_kl["answer"]))    if region_kl["answer"]    else None

        # per-rollout entropy in reasoning region
        reasoning_entropies = []
        for lp, regs in zip(logprobs_list, regions_list):
            stats = region_stats(lp, regs, "reasoning")
            if stats["mean_entropy"] is not None:
                reasoning_entropies.append(stats["mean_entropy"])
        mean_reasoning_entropy = float(np.mean(reasoning_entropies)) if reasoning_entropies else None

        # selection
        sel_random   = select_random(answers, rng)
        sel_majority = select_majority(answers)
        sel_kl, _    = select_kl_centroid(logprobs_list, answers)

        is_correct_majority = sel_majority == gold
        correct_random   += int(sel_random == gold)
        correct_majority += int(is_correct_majority)
        correct_kl       += int(sel_kl == gold)

        # accumulate KL by correctness of majority answer
        # (majority is the best proxy for whether the question was "known")
        if mean_kl_reasoning is not None:
            (reasoning_kl_correct if is_correct_majority else reasoning_kl_wrong).append(mean_kl_reasoning)
        if mean_kl_answer is not None:
            (answer_kl_correct if is_correct_majority else answer_kl_wrong).append(mean_kl_answer)
        if mean_reasoning_entropy is not None:
            (reasoning_entropy_correct if is_correct_majority else reasoning_entropy_wrong).append(mean_reasoning_entropy)

        record = {
            "index": i,
            "gold": gold,
            "answers": answers,
            "selected_random":      sel_random,
            "selected_majority":    sel_majority,
            "selected_kl_centroid": sel_kl,
            "kl_centroid_index":    centroid_idx,
            "avg_kl_per_rollout":   avg_kls,
            "mean_kl_reasoning":    mean_kl_reasoning,
            "mean_kl_answer":       mean_kl_answer,
            "mean_reasoning_entropy": mean_reasoning_entropy,
            "correct_random":   sel_random == gold,
            "correct_majority": is_correct_majority,
            "correct_kl":       sel_kl == gold,
        }
        samples_file.write(json.dumps(record) + "\n")
        samples_file.flush()

        if (i + 1) % 10 == 0:
            n = i + 1
            print(f"[{n}/{len(dataset)}]  "
                  f"random={correct_random/n:.3f}  "
                  f"majority={correct_majority/n:.3f}  "
                  f"kl={correct_kl/n:.3f}")

    samples_file.close()

    n = len(dataset)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else None

    summary = {
        "n_samples": n,
        "n_rollouts": args.n_rollouts,
        "temperature": args.temperature,
        "model_path": args.model_path,
        # selection accuracy
        "accuracy_random":       correct_random / n,
        "accuracy_majority":     correct_majority / n,
        "accuracy_kl_centroid":  correct_kl / n,
        # memorization diagnostic
        # if model genuinely reasons: reasoning_kl_correct < reasoning_kl_wrong
        # if pattern-matching:        reasoning_kl_correct ≈ reasoning_kl_wrong
        #                             but answer_kl_correct < answer_kl_wrong
        "mean_kl_reasoning_when_correct": safe_mean(reasoning_kl_correct),
        "mean_kl_reasoning_when_wrong":   safe_mean(reasoning_kl_wrong),
        "mean_kl_answer_when_correct":    safe_mean(answer_kl_correct),
        "mean_kl_answer_when_wrong":      safe_mean(answer_kl_wrong),
        "mean_reasoning_entropy_when_correct": safe_mean(reasoning_entropy_correct),
        "mean_reasoning_entropy_when_wrong":   safe_mean(reasoning_entropy_wrong),
    }

    print("\n" + "=" * 60)
    print("Selection accuracy:")
    print(f"  random        : {summary['accuracy_random']:.4f}")
    print(f"  majority      : {summary['accuracy_majority']:.4f}")
    print(f"  kl_centroid   : {summary['accuracy_kl_centroid']:.4f}")
    print("\nMemorization diagnostic (reasoning region KL):")
    print(f"  correct answers — mean reasoning KL : {summary['mean_kl_reasoning_when_correct']}")
    print(f"  wrong answers   — mean reasoning KL : {summary['mean_kl_reasoning_when_wrong']}")
    print(f"  correct answers — mean answer KL    : {summary['mean_kl_answer_when_correct']}")
    print(f"  wrong answers   — mean answer KL    : {summary['mean_kl_answer_when_wrong']}")
    print(f"  correct answers — reasoning entropy : {summary['mean_reasoning_entropy_when_correct']}")
    print(f"  wrong answers   — reasoning entropy : {summary['mean_reasoning_entropy_when_wrong']}")
    print("=" * 60)
    print("\nInterpretation guide:")
    print("  reasoning_kl correct << wrong  →  model genuinely reasons on known questions")
    print("  reasoning_kl correct ≈  wrong  →  reasoning paths are noisy regardless (pattern-matching)")
    print("  answer_kl    correct << wrong  →  answer consistency is the only signal (majority voting sufficient)")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")
    print(f"  summary.json  — aggregated metrics")
    print(f"  samples.jsonl — one record per question, written incrementally")


if __name__ == "__main__":
    main()
