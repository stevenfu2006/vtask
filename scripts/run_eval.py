#!/usr/bin/env python3
"""
Evaluate a model against vtask domains.

Usage:
  python scripts/run_eval.py --api anthropic --api-key sk-ant-... --domain scheduling --size 50
  python scripts/run_eval.py --api anthropic --api-key sk-ant-... --all --size 100
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import VTASK
from VTASK.harness import evaluate, make_anthropic_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", choices=["anthropic"], default="anthropic")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--domain", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_fn = make_anthropic_model(api_key=args.api_key, model=args.model)
    domains = VTASK.list_domains() if args.all else [args.domain]

    for domain in domains:
        result = evaluate(
            model_fn=model_fn,
            domain=domain,
            size=args.size,
            seed=args.seed,
            difficulty=args.difficulty,
            verbose=args.verbose,
        )
        print(result)


if __name__ == "__main__":
    main()
