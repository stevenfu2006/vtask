"""
Model evaluation harness.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import time


@dataclass
class EvalResult:
    domain: str
    total: int
    correct: int
    accuracy: float
    by_difficulty: dict[int, dict]
    avg_latency_ms: float
    errors: list[str]

    def __str__(self):
        lines = [
            f"\n{'='*50}",
            f"Domain: {self.domain}",
            f"Accuracy: {self.accuracy:.1%} ({self.correct}/{self.total})",
            f"Avg latency: {self.avg_latency_ms:.0f}ms",
            f"\nBy difficulty:",
        ]
        for d in sorted(self.by_difficulty):
            r = self.by_difficulty[d]
            lines.append(f"  Level {d}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)} (first: {self.errors[0]})")
        lines.append("=" * 50)
        return "\n".join(lines)


def evaluate(
    model_fn: Callable[[str], str],
    domain: str,
    size: int = 100,
    seed: int = 42,
    difficulty: int | None = None,
    verbose: bool = False,
) -> EvalResult:
    import VTASK
    dataset = VTASK.create_dataset(domain, size=size, seed=seed, difficulty=difficulty)

    correct = 0
    total = 0
    errors = []
    by_difficulty: dict[int, dict] = {}
    total_latency = 0.0

    for entry in dataset:
        d = entry.difficulty
        if d not in by_difficulty:
            by_difficulty[d] = {"total": 0, "correct": 0, "accuracy": 0.0}

        try:
            t0 = time.time()
            raw_answer = model_fn(entry.question)
            elapsed = (time.time() - t0) * 1000
            total_latency += elapsed

            score = dataset.score_answer(raw_answer, entry)
            is_correct = score >= 0.5

            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"[{status}] D{d} | Q: {entry.question[:60]}...")
                print(f"      Model: {raw_answer!r} | Expected: {entry.answer!r}")

            if is_correct:
                correct += 1
                by_difficulty[d]["correct"] += 1

            total += 1
            by_difficulty[d]["total"] += 1

        except Exception as e:
            errors.append(str(e))
            total += 1
            by_difficulty[d]["total"] += 1

    for d in by_difficulty:
        r = by_difficulty[d]
        r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0.0

    return EvalResult(
        domain=domain,
        total=total,
        correct=correct,
        accuracy=correct / total if total > 0 else 0.0,
        by_difficulty=by_difficulty,
        avg_latency_ms=total_latency / total if total > 0 else 0.0,
        errors=errors,
    )


def make_anthropic_model(api_key: str, model: str = "claude-sonnet-4-6",
                          system_prompt: str | None = None) -> Callable[[str], str]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    sys = system_prompt or (
        "You are solving reasoning problems. "
        "Answer with ONLY the answer value — no explanation, no units unless asked, "
        "no punctuation. If the answer is a number, write just the number. "
        "If yes/no, write just yes or no."
    )

    def call(question: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=256,
            system=sys,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text.strip()

    return call
