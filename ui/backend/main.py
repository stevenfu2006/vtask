"""
VTASK API server. Imports VTASK as a library; never copies its source.
Run: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import VTASK
from VTASK.registry import REGISTRY

app = FastAPI(title="VTASK API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: api_task_id -> (TaskEntry, generator_instance)
task_store: dict[str, tuple] = {}

DOMAIN_DESCRIPTIONS = {
    "scheduling": "Job scheduling — minimum makespan under worker & precedence constraints",
    "dependencies": "Project DAG — critical path, topological ordering, prerequisites",
    "inventory": "Inventory simulation — stock levels, reorder logic, lead times",
    "fsm": "Finite state machines — DFA/NFA acceptance, string generation",
    "spatial": "Grid pathfinding — BFS, Dijkstra with weights, waypoints",
    "temporal": "Temporal constraints — ordering, satisfiability, PERT/CPM",
}


# ── Models ────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    domain: str
    difficulty: int
    seed: int = 42


class VerifyRequest(BaseModel):
    task_id: str
    answer: str


class EvalRequest(BaseModel):
    domain: str
    difficulty: Optional[int] = None
    size: int = 50
    api_key: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/domains")
def get_domains():
    return [
        {"name": name, "description": DOMAIN_DESCRIPTIONS.get(name, "")}
        for name in VTASK.list_domains()
    ]


@app.post("/api/generate")
def generate_task(req: GenerateRequest):
    if req.domain not in REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {req.domain}")
    if not (1 <= req.difficulty <= 5):
        raise HTTPException(status_code=400, detail="difficulty must be 1–5")

    generator = REGISTRY[req.domain]()
    entry = generator.generate(seed=req.seed, difficulty=req.difficulty)

    api_task_id = str(uuid.uuid4())
    entry.task_id = api_task_id
    task_store[api_task_id] = (entry, generator)

    answer_hash = hashlib.sha256(entry.answer.encode()).hexdigest()

    # Build MC option list: all available distractors + the correct answer, shuffled.
    # The correct answer is mixed in so the UI can offer real multiple-choice without
    # exposing which option is correct (revealed only after /verify).
    options = list(dict.fromkeys(entry.distractors + [entry.answer]))  # dedup, preserve
    rng = random.Random(req.seed ^ (hash(api_task_id) & 0xFFFFFFFF))
    rng.shuffle(options)

    return {
        "task_id": api_task_id,
        "question": entry.question,
        "distractors": entry.distractors,
        "options": options,
        "difficulty": entry.difficulty,
        "domain": entry.domain,
        "answer_hash": answer_hash,
    }


@app.post("/api/verify")
def verify_answer(req: VerifyRequest):
    if req.task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found (server may have restarted)")

    entry, generator = task_store[req.task_id]
    score = generator.score_answer(req.answer, entry)
    is_correct = score >= 0.5

    return {
        "correct": is_correct,
        "score": score,
        "correct_answer": entry.answer,
    }


@app.post("/api/eval")
async def eval_endpoint(req: EvalRequest):
    if req.domain not in REGISTRY and req.domain != "all":
        raise HTTPException(status_code=400, detail=f"Unknown domain: {req.domain}")
    if req.size < 1 or req.size > 500:
        raise HTTPException(status_code=400, detail="size must be 1–500")

    return StreamingResponse(
        _eval_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _eval_stream(req: EvalRequest):
    import anthropic

    loop = asyncio.get_event_loop()

    domain = req.domain
    difficulty = req.difficulty
    size = req.size
    api_key = req.api_key

    dataset = VTASK.create_dataset(domain, size=size, seed=42, difficulty=difficulty)

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = (
        "Answer with only the answer value. "
        "No explanation, no units unless asked, no punctuation."
    )

    correct_count = 0
    total = 0
    by_difficulty: dict[str, dict] = {}

    for entry in dataset:
        q = entry.question

        def _call(question=q):
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": question}],
            )
            return msg.content[0].text.strip()

        try:
            answer = await loop.run_in_executor(None, _call)
            score = dataset.score_answer(answer, entry)
            is_correct = score >= 0.5
        except Exception as e:
            score = 0.0
            is_correct = False

        if is_correct:
            correct_count += 1
        total += 1

        d = str(entry.difficulty)
        if d not in by_difficulty:
            by_difficulty[d] = {"correct": 0, "total": 0}
        by_difficulty[d]["total"] += 1
        if is_correct:
            by_difficulty[d]["correct"] += 1

        event = {
            "task_id": entry.task_id,
            "difficulty": entry.difficulty,
            "correct": is_correct,
            "score": score,
            "running_accuracy": correct_count / total,
        }
        yield f"data: {json.dumps(event)}\n\n"

    final = {
        "done": True,
        "final_accuracy": correct_count / total if total > 0 else 0.0,
        "by_difficulty": by_difficulty,
    }
    yield f"data: {json.dumps(final)}\n\n"
