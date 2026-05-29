"""
Evaluate Claude against all VTASK domains and print a performance report.
Run: python examples/anthropic_eval.py --api-key sk-ant-...
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import VTASK
from VTASK.harness import evaluate, make_anthropic_model

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)
parser.add_argument("--model", default="claude-sonnet-4-6")
parser.add_argument("--size", type=int, default=50,
                    help="Tasks per domain (keep low to control API cost)")
args = parser.parse_args()

model_fn = make_anthropic_model(api_key=args.api_key, model=args.model)

print(f"\nEvaluating {args.model} across {len(VTASK.list_domains())} domains")
print(f"Tasks per domain: {args.size}")
print(f"Total API calls: ~{args.size * len(VTASK.list_domains())}\n")

results = {}
for domain in VTASK.list_domains():
    result = evaluate(model_fn, domain=domain, size=args.size, seed=42)
    results[domain] = result
    print(result)

print("\n\nSUMMARY")
print(f"{'Domain':<20} {'Accuracy':>10} {'Correct':>10}")
print("-" * 42)
for domain, r in results.items():
    print(f"{domain:<20} {r.accuracy:>9.1%} {r.correct:>5}/{r.total}")

overall_acc = sum(r.correct for r in results.values()) / sum(r.total for r in results.values())
print("-" * 42)
print(f"{'OVERALL':<20} {overall_acc:>9.1%}")
