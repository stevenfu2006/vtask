"""
Demonstrates core vtask usage without any API calls.
Run: python examples/basic_usage.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import VTASK

print("Available domains:", VTASK.list_domains())

for domain in VTASK.list_domains():
    ds = VTASK.create_dataset(domain, size=3, seed=42)
    gen = ds.generator
    entry = ds[0]
    print(f"\n[{entry.domain.upper()} | Difficulty {entry.difficulty}]")
    print(f"Q: {entry.question[:200]}...")
    print(f"A: {entry.answer}")
    print(f"Distractors: {entry.distractors}")
    score_correct = ds.score_answer(entry.answer, entry)
    score_wrong = ds.score_answer(entry.distractors[0] if entry.distractors else "999", entry)
    print(f"Correct answer scores: {score_correct}")
    print(f"Wrong answer scores:   {score_wrong}")

print("\n\nGenerating 200-task mixed dataset...")
all_tasks = []
for domain in VTASK.list_domains():
    ds = VTASK.create_dataset(domain, size=33, seed=123)
    all_tasks.extend(ds.entries)

print(f"Total tasks: {len(all_tasks)}")
print(f"Domains covered: {set(e.domain for e in all_tasks)}")
