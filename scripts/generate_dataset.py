#!/usr/bin/env python3
"""
Generate a JSONL dataset file for one or all domains.

Usage:
  python scripts/generate_dataset.py --domain scheduling --size 1000 --output data/scheduling.jsonl
  python scripts/generate_dataset.py --all --size 500 --output-dir data/
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import VTASK


def main():
    parser = argparse.ArgumentParser(description="Generate VTASK dataset")
    parser.add_argument("--domain", type=str, help="Domain name (or use --all)")
    parser.add_argument("--all", action="store_true", help="Generate all domains")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=None,
                        help="Fix difficulty level (1-5). Default: mixed.")
    parser.add_argument("--output", type=str, help="Output JSONL path (single domain)")
    parser.add_argument("--output-dir", type=str, help="Output directory (--all mode)")
    args = parser.parse_args()

    if args.all:
        os.makedirs(args.output_dir or "data", exist_ok=True)
        for domain in VTASK.list_domains():
            path = os.path.join(args.output_dir or "data", f"{domain}.jsonl")
            print(f"Generating {args.size} tasks for domain '{domain}'...")
            ds = VTASK.create_dataset(domain, size=args.size, seed=args.seed,
                                       difficulty=args.difficulty)
            ds.to_jsonl(path)
            print(f"  → Saved to {path}")
    elif args.domain:
        path = args.output or f"{args.domain}.jsonl"
        print(f"Generating {args.size} tasks for domain '{args.domain}'...")
        ds = VTASK.create_dataset(args.domain, size=args.size, seed=args.seed,
                                   difficulty=args.difficulty)
        ds.to_jsonl(path)
        print(f"Saved to {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
