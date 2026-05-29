#!/usr/bin/env python3
"""
Before/after accuracy comparison for a fine-tuned model vs its base.

Runs both the base model and fine-tuned checkpoint against VTASK domains
and prints a side-by-side accuracy table.

Usage:
  python train/eval_model.py --base Qwen/Qwen2.5-1.5B-Instruct --finetuned ./output
  python train/eval_model.py --base Qwen/Qwen2.5-1.5B-Instruct --finetuned ./output --domain all
"""
from __future__ import annotations
import argparse
import sys

import VTASK
from VTASK.registry import REGISTRY
from VTASK.training import format_chat_prompt, SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Before/after VTASK accuracy comparison")
    parser.add_argument("--base", required=True,
                        help="Base model ID or path (before fine-tuning)")
    parser.add_argument("--finetuned", required=True,
                        help="Fine-tuned model checkpoint path")
    parser.add_argument("--domain", default="scheduling",
                        help="Domain name or 'all'")
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--size", type=int, default=50,
                        help="Number of tasks per domain for evaluation")
    parser.add_argument("--seed", type=int, default=99,
                        help="Seed (use different seed from training)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def make_pipeline(model_path: str, max_new_tokens: int):
    from transformers import pipeline, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        trust_remote_code=True,
        device_map="auto",
    )

    def call(question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix
        if output.startswith(prompt):
            output = output[len(prompt):]
        # Extract answer from tags
        import re
        match = re.search(r"<answer>(.*?)</answer>", output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
        return lines[-1] if lines else ""

    return call


def eval_domain(model_fn, domain_name: str, size: int, seed: int,
                difficulty: int | None) -> dict:
    generator = REGISTRY[domain_name]()
    task_ds = generator.create_dataset(size=size, seed=seed, difficulty=difficulty)
    correct = 0
    by_diff: dict[int, list[float]] = {}
    for entry in task_ds:
        try:
            raw = model_fn(entry.question)
        except Exception:
            raw = ""
        score = generator.score_answer(raw, entry)
        if score >= 0.5:
            correct += 1
        by_diff.setdefault(entry.difficulty, []).append(score)
    accuracy = correct / len(task_ds) if task_ds else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(task_ds),
        "by_difficulty": {d: sum(s)/len(s) for d, s in by_diff.items()},
    }


def main() -> None:
    args = parse_args()

    if args.domain == "all":
        domain_names = VTASK.list_domains()
    else:
        if args.domain not in REGISTRY:
            print(f"Unknown domain '{args.domain}'. Available: {VTASK.list_domains()}", file=sys.stderr)
            sys.exit(1)
        domain_names = [args.domain]

    print(f"Loading base model: {args.base}")
    base_fn = make_pipeline(args.base, args.max_new_tokens)

    print(f"Loading fine-tuned model: {args.finetuned}")
    ft_fn = make_pipeline(args.finetuned, args.max_new_tokens)

    print(f"\nEvaluating {len(domain_names)} domain(s), {args.size} tasks each (seed={args.seed})\n")

    results = {}
    for domain in domain_names:
        print(f"  Evaluating {domain}...", flush=True)
        base_result = eval_domain(base_fn, domain, args.size, args.seed, args.difficulty)
        ft_result = eval_domain(ft_fn, domain, args.size, args.seed, args.difficulty)
        results[domain] = {"base": base_result, "finetuned": ft_result}

    # Print summary table
    print(f"\n{'Domain':<20} {'Base':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 54)
    total_base_correct = total_ft_correct = total = 0
    for domain, r in results.items():
        base_acc = r["base"]["accuracy"]
        ft_acc = r["finetuned"]["accuracy"]
        delta = ft_acc - base_acc
        sign = "+" if delta >= 0 else ""
        print(f"{domain:<20} {base_acc:>9.1%} {ft_acc:>11.1%} {sign}{delta:>7.1%}")
        total_base_correct += r["base"]["correct"]
        total_ft_correct += r["finetuned"]["correct"]
        total += r["base"]["total"]

    print("-" * 54)
    overall_base = total_base_correct / total if total else 0
    overall_ft = total_ft_correct / total if total else 0
    delta = overall_ft - overall_base
    sign = "+" if delta >= 0 else ""
    print(f"{'OVERALL':<20} {overall_base:>9.1%} {overall_ft:>11.1%} {sign}{delta:>7.1%}")

    if len(domain_names) == 1:
        domain = domain_names[0]
        print(f"\nBy difficulty ({domain}):")
        base_by = results[domain]["base"]["by_difficulty"]
        ft_by = results[domain]["finetuned"]["by_difficulty"]
        for d in sorted(set(base_by) | set(ft_by)):
            b = base_by.get(d, 0)
            f = ft_by.get(d, 0)
            sign = "+" if (f - b) >= 0 else ""
            print(f"  Level {d}: {b:.1%} → {f:.1%} ({sign}{f-b:.1%})")


if __name__ == "__main__":
    main()
