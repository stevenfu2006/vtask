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
    "You must always end your response with your final answer on the last line "
    "wrapped in answer tags like this: <answer>42</answer>. "
    "Do not include units, just the number or value. "
    "Example: if the answer is 8 hours, write <answer>8</answer>. "
    "If the answer is yes, write <answer>yes</answer>. "
    "Always include the answer tags. Never skip them."
)


def format_chat_prompt(entry: TaskEntry) -> list[dict]:
    """Convert a TaskEntry into a chat-format prompt list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": entry.question},
    ]
