"""
RL training utilities for VTASK.

Provides:
- format_chat_prompt: converts TaskEntry to chat messages
- SYSTEM_PROMPT: system prompt used during training and evaluation
"""
from __future__ import annotations
from VTASK.base import TaskEntry

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
