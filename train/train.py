#!/usr/bin/env python3
"""
GRPO fine-tuning script for VTASK domains.

Uses TRL's GRPOTrainer with VTASK's RewardFunction to train a model
on procedurally generated reasoning tasks.

Usage:
  python train/train.py --model Qwen/Qwen2.5-1.5B-Instruct --domain scheduling
  python train/train.py --model Qwen/Qwen2.5-7B-Instruct --domain all --steps 500
"""
from __future__ import annotations
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import VTASK
from VTASK.registry import REGISTRY
from VTASK.training import RewardFunction, VTASKDataset, format_chat_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training on VTASK domains")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--domain", default="scheduling",
                        help="Domain name or 'all' for mixed training")
    parser.add_argument("--difficulty", type=int, default=None,
                        help="Fix difficulty (1-5). Default: mixed.")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--dataset-size", type=int, default=500,
                        help="Number of tasks to generate for training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output",
                        help="Directory to save checkpoints")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--wandb-project", default="vtask-grpo",
                        help="Weights & Biases project name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    return parser.parse_args()


def build_dataset(domain: str, size: int, seed: int, difficulty: int | None,
                  generator, reward_fn: RewardFunction) -> "datasets.Dataset":
    task_ds = generator.create_dataset(size=size, seed=seed, difficulty=difficulty)
    vtask_ds = VTASKDataset(task_ds, reward_fn)
    return vtask_ds.to_hf_dataset()


def main() -> None:
    args = parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    elif args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # Resolve domain(s)
    if args.domain == "all":
        domain_names = VTASK.list_domains()
    else:
        if args.domain not in REGISTRY:
            print(f"Unknown domain '{args.domain}'. Available: {VTASK.list_domains()}", file=sys.stderr)
            sys.exit(1)
        domain_names = [args.domain]

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Build combined dataset across all requested domains
    all_datasets = []
    reward_fns = {}
    for domain_name in domain_names:
        generator = REGISTRY[domain_name]()
        reward_fn = RewardFunction(generator)
        ds = build_dataset(
            domain=domain_name,
            size=args.dataset_size // len(domain_names),
            seed=args.seed,
            difficulty=args.difficulty,
            generator=generator,
            reward_fn=reward_fn,
        )
        all_datasets.append(ds)
        reward_fns[domain_name] = (generator, reward_fn)

    if len(all_datasets) == 1:
        train_dataset = all_datasets[0]
        primary_generator, primary_reward_fn = list(reward_fns.values())[0]
    else:
        from datasets import concatenate_datasets
        train_dataset = concatenate_datasets(all_datasets).shuffle(seed=args.seed)
        # For mixed training, use a unified reward function
        primary_generator = None
        primary_reward_fn = _MultiDomainReward(reward_fns)

    def reward_wrapper(completions, prompts=None, **kwargs):
        return primary_reward_fn(completions=completions, prompts=prompts, **kwargs)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        report_to="none" if args.no_wandb else "wandb",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_funcs=[reward_wrapper],
    )

    print(f"Training on domain(s): {domain_names}")
    print(f"Dataset size: {len(train_dataset)} tasks | Steps: {args.steps}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


class _MultiDomainReward:
    """Unified reward function for mixed-domain training."""

    def __init__(self, reward_fns: dict):
        self._fns = {k: v[1] for k, v in reward_fns.items()}

    def __call__(self, completions, prompts=None, **kwargs):
        rewards = []
        if prompts is None:
            prompts = [None] * len(completions)
        for completion, prompt in zip(completions, prompts):
            score = 0.0
            for fn in self._fns.values():
                result = fn([completion], [prompt])
                if result[0] > 0.0:
                    score = result[0]
                    break
            rewards.append(score)
        return rewards


if __name__ == "__main__":
    main()
