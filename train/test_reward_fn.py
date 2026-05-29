#!/usr/bin/env python3
"""
Directly tests the TRL 1.5.1 reward function fix.

Simulates exactly what GRPOTrainer does: builds the dataset,
then calls reward_fn(completions, correct_answer=[...], domain=[...])
as kwargs — no prompt-string lookup needed.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import VTASK
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
from train import extract_answer, make_reward_fn, build_dataset
from VTASK.registry import REGISTRY

print("=" * 60)
print("Testing TRL 1.5.1 reward function fix")
print("=" * 60)

# --- build dataset (same path as the real training script) ---
dataset = build_dataset(domains=["scheduling"], size=8, seed=42, difficulty=1)
generators = {"scheduling": REGISTRY["scheduling"]()}
reward_fn = make_reward_fn(generators)

entries = dataset[:4]
correct_answers = entries["correct_answer"]
domains = entries["domain"]

print(f"\nSample tasks:")
for i, (q, a) in enumerate(zip(entries["prompt"], correct_answers)):
    print(f"  [{i}] Q: {q[-1]['content'][:60]}... | A: {a!r}")

# --- TEST 1: correct completions ---
correct_completions = [f"<answer>{a}</answer>" for a in correct_answers]
rewards = reward_fn(correct_completions, correct_answer=correct_answers, domain=domains)
print(f"\nTest 1 — correct answers via <answer> tags:")
for a, c, r in zip(correct_answers, correct_completions, rewards):
    print(f"  expected={a!r:6}  completion={c!r:20}  reward={r}")
assert all(r == 1.0 for r in rewards), f"FAIL: expected all 1.0, got {rewards}"
print("  PASS: all rewards = 1.0")

# --- TEST 2: wrong completions ---
wrong_completions = ["I don't know <answer>999</answer>"] * 4
rewards = reward_fn(wrong_completions, correct_answer=correct_answers, domain=domains)
print(f"\nTest 2 — wrong answers:")
for a, c, r in zip(correct_answers, wrong_completions, rewards):
    print(f"  expected={a!r:6}  completion={c!r:34}  reward={r}")
assert all(r == 0.0 for r in rewards), f"FAIL: expected all 0.0, got {rewards}"
print("  PASS: all rewards = 0.0")

# --- TEST 3: mixed (simulates a real training batch) ---
mixed_completions = [
    f"Let me think... the answer is <answer>{correct_answers[0]}</answer>",
    "<answer>999</answer>",
    f"After calculation: <answer>{correct_answers[2]}</answer>",
    "I compute this as <answer>999</answer>",
]
rewards = reward_fn(mixed_completions, correct_answer=correct_answers, domain=domains)
print(f"\nTest 3 — mixed batch (like a real training step):")
for a, c, r in zip(correct_answers, mixed_completions, rewards):
    print(f"  expected={a!r:6}  reward={r}  {'✓' if r == 1.0 else '✗'}")
assert rewards[0] == 1.0 and rewards[2] == 1.0, "Correct answers should score 1.0"
assert rewards[1] == 0.0 and rewards[3] == 0.0, "Wrong answers should score 0.0"
print("  PASS: correct=1.0, wrong=0.0 — reward function works with TRL 1.5.1 kwargs")

# --- TEST 4: empty kwargs (old broken behavior simulation) ---
rewards_no_kwargs = reward_fn(mixed_completions)
print(f"\nTest 4 — missing kwargs (old broken behavior):")
print(f"  rewards = {rewards_no_kwargs}  (all 0.0 — no correct_answer in kwargs)")

print("\n" + "=" * 60)
print("RESULT: Reward function fix verified.")
print("  - kwargs['correct_answer'] is used for scoring (TRL 1.5.1 pattern)")
print("  - Correct answers → 1.0, wrong answers → 0.0")
print("  - Non-zero rewards will flow back to the GRPO loss")
print("=" * 60)
