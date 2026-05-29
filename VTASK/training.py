"""
RL training utilities for VTASK.

Provides:
- RewardFunction: TRL-compatible reward callable
- format_chat_prompt: converts TaskEntry to chat messages
- VTASKDataset: HuggingFace-compatible dataset wrapper
"""
from __future__ import annotations
import re
from typing import Any
from VTASK.base import TaskEntry, TaskGenerator, TaskDataset

SYSTEM_PROMPT = (
    "You are solving reasoning problems. "
    "Think through the problem carefully, then provide your final answer. "
    "Wrap your final answer in <answer> and </answer> tags. "
    "Example: <answer>42</answer>"
)


def format_chat_prompt(entry: TaskEntry) -> list[dict]:
    """Convert a TaskEntry into a chat-format prompt list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": entry.question},
    ]


class RewardFunction:
    """
    TRL GRPOTrainer-compatible reward function.

    Usage:
        reward_fn = RewardFunction(generator)
        reward_fn.register(entry, prompt_str)
        # TRL calls: reward_fn(completions=..., prompts=..., **kwargs)
    """

    def __init__(self, generator: TaskGenerator):
        self.generator = generator
        self._prompt_to_entry: dict[str, TaskEntry] = {}

    def register(self, entry: TaskEntry, prompt: str) -> None:
        """Map a formatted prompt string back to its TaskEntry for scoring."""
        self._prompt_to_entry[prompt] = entry

    def __call__(
        self,
        completions: list[str],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """
        Score a batch of model completions.

        Args:
            completions: List of raw model outputs.
            prompts: Corresponding prompts (used to look up TaskEntry).
                     Falls back to kwargs["prompt"] if not provided.

        Returns:
            List of float rewards (0.0 or 1.0) aligned with completions.
        """
        if prompts is None:
            prompts = kwargs.get("prompt", [None] * len(completions))

        rewards = []
        for completion, prompt in zip(completions, prompts):
            entry = self._prompt_to_entry.get(prompt) if prompt else None
            if entry is None:
                rewards.append(0.0)
                continue
            answer = self._extract_answer(completion)
            score = self.generator.score_answer(answer, entry)
            rewards.append(score)
        return rewards

    def _extract_answer(self, completion: str) -> str:
        """
        Extract the final answer from a model completion.

        Priority:
        1. Content inside <answer>...</answer> tags
        2. Pattern "answer is X" (case-insensitive)
        3. Last non-empty line of the completion
        """
        tag_match = re.search(r"<answer>(.*?)</answer>", completion, re.IGNORECASE | re.DOTALL)
        if tag_match:
            return tag_match.group(1).strip()

        phrase_match = re.search(r"answer is[:\s]+(.+?)[\.\n]?$", completion, re.IGNORECASE)
        if phrase_match:
            return phrase_match.group(1).strip()

        lines = [l.strip() for l in completion.strip().splitlines() if l.strip()]
        return lines[-1] if lines else ""


class VTASKDataset:
    """
    Wraps a TaskDataset for use with HuggingFace / TRL training pipelines.

    Each item is a dict with:
        - "prompt": formatted chat messages (list of dicts)
        - "prompt_str": flattened string version (for reward_fn lookup)
        - "answer": ground-truth answer string
        - "domain": domain name
        - "difficulty": difficulty level
        - "task_id": unique task identifier
    """

    def __init__(self, task_dataset: TaskDataset, reward_fn: RewardFunction):
        self.task_dataset = task_dataset
        self.reward_fn = reward_fn
        self._items: list[dict] = []
        self._build()

    def _build(self) -> None:
        for entry in self.task_dataset:
            messages = format_chat_prompt(entry)
            prompt_str = _messages_to_str(messages)
            self.reward_fn.register(entry, prompt_str)
            self._items.append({
                "prompt": messages,
                "prompt_str": prompt_str,
                "answer": entry.answer,
                "domain": entry.domain,
                "difficulty": entry.difficulty,
                "task_id": entry.task_id,
            })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]

    def to_hf_dataset(self):
        """Convert to a HuggingFace datasets.Dataset for use with TRL."""
        from datasets import Dataset
        return Dataset.from_list(self._items)


def _messages_to_str(messages: list[dict]) -> str:
    """Stable string representation of a chat message list for dict keying."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
