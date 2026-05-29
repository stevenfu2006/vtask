"""
Base classes for all vtask domains.
Every task domain must implement TaskGenerator and its score_answer method.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import random


@dataclass
class TaskEntry:
    """A single generated task instance."""
    question: str
    answer: str
    distractors: list[str]
    difficulty: int
    domain: str
    metadata: dict
    task_id: str


class TaskGenerator:
    """
    Abstract base class. All domain generators subclass this.

    Subclasses MUST implement:
        - generate(seed, difficulty) -> TaskEntry
        - score_answer(answer, entry) -> float  (0.0 or 1.0, occasionally partial)

    Subclasses SHOULD implement:
        - generate_distractors(entry, n) -> list[str]
    """
    domain: str = "base"
    difficulty_range: tuple[int, int] = (1, 5)

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        raise NotImplementedError

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        raise NotImplementedError

    def generate_distractors(self, entry: TaskEntry, n: int = 3) -> list[str]:
        return []

    def create_dataset(self, size: int, seed: int = 42,
                       difficulty: int | None = None) -> "TaskDataset":
        rng = random.Random(seed)
        entries = []
        for i in range(size):
            s = rng.randint(0, 2**31)
            d = difficulty if difficulty is not None else rng.randint(*self.difficulty_range)
            entry = self.generate(seed=s, difficulty=d)
            entry.task_id = f"{self.domain}_{seed}_{i}"
            entries.append(entry)
        return TaskDataset(entries=entries, generator=self)


class TaskDataset:
    """Wraps a list of TaskEntry objects with scoring convenience."""

    def __init__(self, entries: list[TaskEntry], generator: TaskGenerator):
        self.entries = entries
        self.generator = generator

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        return self.generator.score_answer(answer, entry)

    def to_jsonl(self, path: str):
        import json
        with open(path, "w") as f:
            for e in self.entries:
                row = {
                    "question": e.question,
                    "answer": e.answer,
                    "difficulty": e.difficulty,
                    "domain": e.domain,
                    "task_id": e.task_id,
                    "distractors": e.distractors,
                }
                f.write(json.dumps(row) + "\n")
