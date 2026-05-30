#!/usr/bin/env python3
"""
VTASK RL Training Script — GRPO with verifiable rewards.

Usage:
  python train/train.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --domain scheduling \
    --steps 500 \
    --batch-size 8 \
    --output-dir checkpoints/scheduling

  python train/train.py --model Qwen/Qwen2.5-1.5B-Instruct --domain all
"""
import argparse
import random
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import VTASK
from VTASK.training import format_chat_prompt, SYSTEM_PROMPT


def extract_answer(completion: str, correct_answer: str) -> str:
    """
    Extract answer from model completion with multiple fallback strategies.
    For numeric answers, tries to find any matching number in the output.
    """
    completion = completion.strip()

    # Strategy 1: look for <answer> tags
    tag_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    # Strategy 2: look for "answer is X" pattern
    ans_match = re.search(r'(?:answer is|therefore|minimum is|total is)[:\s]+(\d+)', completion, re.IGNORECASE)
    if ans_match:
        return ans_match.group(1).strip()

    # Strategy 3: if correct answer is a number, find all numbers in completion
    # and return the one that matches if present
    try:
        correct_num = int(correct_answer.strip())
        numbers = re.findall(r'\b(\d+)\b', completion)
        if str(correct_num) in numbers:
            return str(correct_num)
        # Return the last number found as best guess
        if numbers:
            return numbers[-1]
    except ValueError:
        pass

    # Strategy 4: last non-empty line
    lines = [l.strip() for l in completion.split('\n') if l.strip()]
    return lines[-1] if lines else completion


def make_reward_fn(generators: dict):
    """
    Returns a TRL 1.5.1 compatible reward function.

    TRL passes extra dataset columns as kwargs.
    We store correct_answer and domain in the dataset,
    so they arrive here as lists in kwargs.
    """
    def reward_fn(completions, prompts=None, **kwargs):
        correct_answers = kwargs.get('correct_answer', [])
        scores = []

        for i, completion in enumerate(completions):
            try:
                correct = correct_answers[i] if i < len(correct_answers) else ''
                extracted = extract_answer(completion, correct)
                extracted_norm = extracted.strip().lower().replace(',', '').replace(' ', '')
                correct_norm = correct.strip().lower().replace(',', '').replace(' ', '')
                score = 1.0 if extracted_norm == correct_norm else 0.0
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return scores

    return reward_fn


def build_dataset(domains: list, size: int, seed: int, difficulty: int | None):
    """Generate training tasks and return as HuggingFace Dataset."""
    from datasets import Dataset

    all_items = []
    rng = random.Random(seed)

    for domain in domains:
        domain_size = size // len(domains)
        ds = VTASK.create_dataset(domain, size=domain_size, seed=rng.randint(0, 2**31), difficulty=difficulty)

        for entry in ds.entries:
            prompt = format_chat_prompt(entry)
            all_items.append({
                'prompt': prompt,
                'correct_answer': entry.answer,
                'domain': entry.domain,
                'difficulty': entry.difficulty,
                'task_id': entry.task_id,
            })

    rng.shuffle(all_items)

    # Sanity check
    if all_items:
        print(f"Sample task [{all_items[0]['domain']}]: {all_items[0]['prompt'][-1]['content'][:80]}...")
        print(f"Correct answer: {all_items[0]['correct_answer']}")

    return Dataset.from_list(all_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--domain', default='all')
    parser.add_argument('--difficulty', type=int, default=None)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dataset-size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='checkpoints/vtask-run')
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    domains = VTASK.list_domains() if args.domain == 'all' else [args.domain]
    print(f"Training on domain(s): {domains}")
    print(f"Model: {args.model}")
    print(f"Steps: {args.steps} | Batch size: {args.batch_size}")

    # Build dataset
    print("Generating training tasks...")
    dataset = build_dataset(
        domains=domains,
        size=args.dataset_size,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    print(f"Dataset size: {len(dataset)} tasks | Steps: {args.steps}")

    # Build reward function
    from VTASK.registry import REGISTRY
    generators = {d: REGISTRY[d]() for d in domains}
    reward_fn = make_reward_fn(generators)

    # Load model and tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOTrainer, GRPOConfig
    import torch

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        report_to='none' if args.no_wandb else 'wandb',
        run_name=f"vtask-{args.domain}-{args.model.split('/')[-1]}",
        max_completion_length=512,
        num_generations=4,
        temperature=0.9,
        seed=args.seed,
    )

    if not args.no_wandb:
        import wandb
        wandb.init(project='vtask-training', config=vars(args))

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == '__main__':
    main()
