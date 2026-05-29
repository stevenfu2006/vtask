# Verifiable Task Engine — Build Instructions for Claude Code

## What you are building

A Python library called `vtask` — a multi-domain procedural task generator and verifier for 
reinforcement learning training data. It generates unlimited parameterized tasks across 6 
professional/workplace reasoning domains where every answer is verifiable by algorithm with 
zero human or LLM involvement.

This is designed to be handed to a frontier AI lab as a complete RL training environment.
It follows the same interface pattern as `reasoning-gym` (NeurIPS 2025) but covers domains 
that reasoning-gym explicitly does not: professional workflow reasoning, scheduling, 
resource allocation, process dependencies, and inventory/logistics.

---

## Project structure to create

```
vtask/
├── CLAUDE.md                  ← this file (already exists)
├── README.md                  ← generate this last
├── pyproject.toml
├── vtask/
│   ├── __init__.py
│   ├── base.py                ← base Task and Dataset classes
│   ├── registry.py            ← task registration and factory
│   ├── domains/
│   │   ├── __init__.py
│   │   ├── scheduling.py      ← Domain 1: job scheduling
│   │   ├── dependencies.py    ← Domain 2: project dependency resolution
│   │   ├── inventory.py       ← Domain 3: inventory/logistics reasoning
│   │   ├── fsm.py             ← Domain 4: finite state machines (extend prior work)
│   │   ├── spatial.py         ← Domain 5: grid-based spatial reasoning
│   │   └── temporal.py        ← Domain 6: temporal constraint ordering
│   └── harness.py             ← model evaluation harness
├── scripts/
│   ├── generate_dataset.py    ← CLI to generate + export JSONL dataset
│   └── run_eval.py            ← CLI to run a model against the harness
├── examples/
│   ├── basic_usage.py
│   └── anthropic_eval.py      ← eval using Anthropic API
└── tests/
    ├── test_base.py
    ├── test_scheduling.py
    ├── test_dependencies.py
    ├── test_inventory.py
    ├── test_fsm.py
    ├── test_spatial.py
    └── test_temporal.py
```

---

## Step 1: Build the base infrastructure

### `vtask/base.py`

```python
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
    question: str           # Natural language question sent to the model
    answer: str             # Ground-truth answer (always a string)
    distractors: list[str]  # Plausible wrong answers for multiple-choice mode
    difficulty: int         # 1-5 scale, set by generator
    domain: str             # e.g. "scheduling", "fsm"
    metadata: dict          # Everything needed by the verifier; do NOT leak to model
    task_id: str            # Deterministic ID: "{domain}_{seed}_{index}"


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
        """
        Returns 1.0 for correct, 0.0 for incorrect.
        May return partial credit (0.0-1.0) for complex tasks.
        The answer is always compared after normalizing whitespace and case
        where appropriate. Do NOT use an LLM here — verification must be
        purely algorithmic.
        """
        raise NotImplementedError

    def generate_distractors(self, entry: TaskEntry, n: int = 3) -> list[str]:
        """Override for domain-specific plausible wrong answers."""
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
                    # Note: distractors included, metadata excluded
                    # (metadata contains verifier state, not model input)
                    "distractors": e.distractors,
                }
                f.write(json.dumps(row) + "\n")
```

---

## Step 2: Build each domain

### Domain 1 — `vtask/domains/scheduling.py`: Job Scheduling

**Concept:** N jobs must be assigned to M workers. Each job has a duration and optional 
worker skill requirement. Find the minimum makespan (total completion time).

**Why verifiable:** The optimal makespan for simple scheduling problems is computable 
exactly via dynamic programming or brute force up to ~10 jobs.

**Difficulty ladder:**
- Level 1: 3 jobs, 2 workers, no constraints → trivial bin packing
- Level 2: 5 jobs, 2-3 workers, some jobs require specific worker type
- Level 3: 7 jobs, 3 workers, precedence constraints (job B must finish before job C starts)  
- Level 4: 10 jobs, 4 workers, precedence + skill constraints
- Level 5: 12 jobs, 4-5 workers, precedence + skill + fixed time windows

```python
"""
Implement this fully. Key implementation notes:

- Use a greedy LPT (Longest Processing Time first) scheduler for the verifier
  at difficulty 1-2, exact DP for 3-5.
- The ANSWER is always the minimum makespan as an integer (e.g. "14").
- score_answer: strip whitespace, parse as int, compare exactly.
- generate_distractors: return [makespan+1, makespan+2, makespan-1] as strings,
  filtering out any that are <= 0.
- The question should present jobs as a readable table:
  
  "You have 3 workers and the following jobs:
   Job A: 4 hours, any worker
   Job B: 6 hours, any worker  
   Job C: 3 hours, requires skilled worker (Worker 2 only)
   Job D: 5 hours, any worker
   Job B must complete before Job D can start.
   What is the minimum number of hours to complete all jobs?"

- metadata must store: jobs (list of dicts), workers (list of dicts), 
  precedences (list of (job_a, job_b) tuples), correct_makespan (int)
"""

import random
from dataclasses import dataclass
from itertools import permutations
from vtask.base import TaskEntry, TaskGenerator


class SchedulingGenerator(TaskGenerator):
    domain = "scheduling"
    difficulty_range = (1, 5)

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        # IMPLEMENT THIS
        # Use seed to create a random.Random instance for reproducibility
        # Generate jobs, workers, constraints based on difficulty
        # Compute correct_makespan using the verifier logic
        # Build natural language question string
        # Return TaskEntry
        raise NotImplementedError

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        try:
            proposed = int(answer.strip())
            return 1.0 if proposed == entry.metadata["correct_makespan"] else 0.0
        except (ValueError, KeyError):
            return 0.0

    def _compute_makespan(self, jobs, workers, precedences) -> int:
        # Implement exact makespan computation
        # For difficulty 1-2: greedy LPT
        # For difficulty 3+: branch and bound or DP
        raise NotImplementedError
```

---

### Domain 2 — `vtask/domains/dependencies.py`: Project Dependency Resolution

**Concept:** Given a set of tasks with dependency relationships (DAG), determine:
- Valid topological ordering(s)
- Critical path length
- Which tasks can run in parallel

**Why verifiable:** Topological sort is O(V+E) and deterministic. Critical path is DP on DAG.

**Difficulty ladder:**
- Level 1: 4 tasks, simple chain A→B→C→D, ask for a valid ordering
- Level 2: 6 tasks, branching DAG, ask for critical path length
- Level 3: 8 tasks, complex DAG, ask which tasks CAN'T run before task X
- Level 4: 10 tasks with durations, ask for minimum project completion time
- Level 5: 12 tasks, ask for tasks on the critical path (output as sorted list)

```python
"""
Implement this fully. Key implementation notes:

- Use networkx for graph operations: pip install networkx
- For ordering questions (difficulty 1): accept ANY valid topological ordering.
  score_answer must verify the proposed ordering satisfies all dependencies,
  NOT just check equality against one answer.
- For critical path (difficulty 2,4): answer is integer hours/days.
- For "which tasks can't run before X" (difficulty 3): answer is comma-separated
  sorted list of task names, e.g. "B, D, F"
- For critical path tasks (difficulty 5): answer is comma-separated sorted list.
- Generate questions using real-sounding project task names (not just A,B,C):
  ["Design", "Prototype", "Test", "Review", "Deploy", "Document", 
   "QA", "Integrate", "Configure", "Build", "Plan", "Validate"]

Example question (difficulty 2):
  "A software project has the following tasks and dependencies:
   - Design: 3 days (no prerequisites)
   - Build: 5 days (requires Design)
   - Test: 4 days (requires Build)
   - Document: 2 days (requires Design)
   - Deploy: 1 day (requires Test and Document)
   What is the minimum number of days to complete the entire project?"

- metadata: graph as adjacency list, task durations, correct answer
"""

import random
from vtask.base import TaskEntry, TaskGenerator


class DependencyGenerator(TaskGenerator):
    domain = "dependencies"
    difficulty_range = (1, 5)

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        raise NotImplementedError

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        # For ordering questions: verify the ordering is topologically valid
        # For numeric questions: parse and compare exactly
        # For list questions: parse comma-separated, sort, compare to metadata answer
        raise NotImplementedError

    def _critical_path(self, graph, durations) -> tuple[int, list[str]]:
        # Returns (critical_path_length, list_of_tasks_on_critical_path)
        raise NotImplementedError

    def _is_valid_topo_order(self, ordering: list[str], edges: list[tuple]) -> bool:
        # Returns True if ordering respects all dependency edges
        raise NotImplementedError
```

---

### Domain 3 — `vtask/domains/inventory.py`: Inventory & Logistics Reasoning

**Concept:** Given inventory levels, demand rates, lead times, and reorder rules, 
determine stock levels at a future point, stockout dates, or optimal reorder quantities.

**Why verifiable:** All arithmetic is deterministic. Given inputs, the answer is exact.

**Difficulty ladder:**
- Level 1: Single item, constant demand, no lead time. "You start with 100 units, sell 8/day. When do you run out?"
- Level 2: Single item, demand + lead time. "You have 50 units, sell 5/day, reorder takes 3 days, you reorder when stock hits 20. How many units do you have on day 15?"
- Level 3: Two items sharing warehouse capacity, competing reorder points
- Level 4: 3 items, seasonal demand (different rates week 1 vs week 2), ask stock on day N
- Level 5: Multi-echelon: central warehouse + 2 stores, transfers between them, ask total system stock on day N

```python
"""
Implement this fully. Key implementation notes:

- Simulate day-by-day for all difficulties. The verifier IS the simulation.
- Answers are always integers (unit counts or day numbers).
- For "when do you run out" questions, answer is a day number (integer).
- For "how many units on day N" questions, answer is an integer >= 0.
- Distractors: [correct-3, correct+5, correct+10] for stock questions;
  [correct-2, correct+1, correct+3] for day questions.
- Make questions feel real by using product names:
  ["laptop", "chair", "monitor", "keyboard", "notebook", "pen", "stapler"]
- Use a simulation class internally that advances day by day, handles
  reorders arriving after lead_time days, and clamps stock at 0 (no negative).
"""

import random
from vtask.base import TaskEntry, TaskGenerator


class InventoryGenerator(TaskGenerator):
    domain = "inventory"
    difficulty_range = (1, 5)

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        raise NotImplementedError

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        try:
            proposed = int(answer.strip().replace(",", ""))
            return 1.0 if proposed == entry.metadata["correct_answer"] else 0.0
        except (ValueError, KeyError):
            return 0.0

    def _simulate(self, params: dict) -> int:
        # Run the inventory simulation and return the answer
        raise NotImplementedError
```

---

### Domain 4 — `vtask/domains/fsm.py`: Finite State Machines

**Extend your prior work. This domain should be the most complete since you've built 
this before. The key additions are:**

**Difficulty ladder:**
- Level 1: 3-state FSM, binary alphabet, "Does string X get accepted?" → "yes" or "no"
- Level 2: 4-state FSM, alphabet {a,b,c}, "What state does the machine end in after processing string X?"
- Level 3: 5-state FSM, "Give a string of length exactly N that gets accepted"
  (verifier checks: length == N AND FSM accepts it)
- Level 4: 6-state FSM with epsilon transitions, "Does string X get accepted?"
- Level 5: Two FSMs A and B, "Give a string accepted by A but rejected by B, or say NONE if none exists of length ≤ 6"
  (verifier runs both FSMs on proposed string)

```python
"""
Implement this fully building on your Bongard/automata background.

Key implementation notes:
- Represent FSMs as: states (list), alphabet (list), transitions (dict of dict),
  start_state (str), accept_states (set)
- For level 3 (generate an accepting string): 
  Use BFS from start state to find shortest accepting string,
  then pad to length N with transitions that stay in accept states if possible.
  Verifier runs the FSM on the proposed string.
- For level 5 (symmetric difference): 
  Compute the symmetric difference automaton A⊕B.
  Use BFS to find accepting string of length ≤ 6 in the symmetric difference.
  If none exists, correct answer is "NONE".
  Verifier: run A and B on proposed string, check exactly one accepts.
- score_answer for yes/no: normalize to lowercase, accept "yes"/"no"/"true"/"false"
- score_answer for state name: strip, lowercase, exact match
- score_answer for string generation: run the actual FSM, return 1.0 if accepted
"""

import random
from collections import deque
from vtask.base import TaskEntry, TaskGenerator


class FSMGenerator(TaskGenerator):
    domain = "fsm"
    difficulty_range = (1, 5)

    def _build_fsm(self, seed: int, n_states: int, alphabet: list) -> dict:
        # Returns FSM as dict with keys: states, alphabet, transitions,
        # start_state, accept_states
        raise NotImplementedError

    def _accepts(self, fsm: dict, string: str) -> bool:
        # Run the FSM on the string, return True if accepted
        raise NotImplementedError

    def _bfs_accepting_string(self, fsm: dict, max_len: int = 10) -> str | None:
        # BFS to find shortest string accepted by FSM
        # Returns None if no accepting string exists within max_len
        raise NotImplementedError

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        raise NotImplementedError

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        raise NotImplementedError
```

---

### Domain 5 — `vtask/domains/spatial.py`: Grid-based Spatial Reasoning

**Concept:** Pathfinding and spatial reasoning on a 2D grid. The model must reason 
about positions, obstacles, and movement rules.

**Why verifiable:** Shortest path is computable exactly with BFS/Dijkstra.

**Difficulty ladder:**
- Level 1: 4x4 grid, no obstacles, "What is the minimum number of steps from (0,0) to (3,3) moving only right or down?"
- Level 2: 6x6 grid, some obstacle cells, 4-directional movement, ask min steps
- Level 3: 8x8 grid with obstacles, ask if destination is reachable at all ("yes"/"no")
- Level 4: 8x8 grid, weighted cells (moving through certain cells costs 2 steps), ask min cost
- Level 5: 10x10 grid, multiple named waypoints that must ALL be visited before reaching goal, ask min steps

```python
"""
Implement this fully.

Key implementation notes:
- Represent grids as 2D lists. '.' = passable, '#' = obstacle, 
  'S' = start, 'E' = end, 'W1/W2' = waypoints (level 5)
- For levels 1-3: use BFS. Answer is integer or "yes"/"no".
- For level 4: use Dijkstra with cell weights.
- For level 5: use BFS with state = (position, frozenset_of_visited_waypoints).
  This is NP-hard in general but tractable for ≤4 waypoints on 10x10 grid.
- Generate grids procedurally: place obstacles randomly, verify start→end 
  is still reachable before returning (regenerate if not).
- Print the grid in the question using ASCII art so the model can visualize:
  
  "Navigate this 6x6 grid from S to E. '#' cells are walls.
   You can move up, down, left, or right (not diagonally).
   
   . . # . . .
   . S . . # .
   . . . # . .
   # . . . . .
   . . # . E .
   . . . . . .
   
   What is the minimum number of steps to reach E?"

- score_answer: strip and parse as int (or yes/no), compare exactly.
- generate_distractors: [correct+1, correct+2, correct+3] for step counts.
"""

import random
from collections import deque
from vtask.base import TaskEntry, TaskGenerator


class SpatialGenerator(TaskGenerator):
    domain = "spatial"
    difficulty_range = (1, 5)

    def _generate_grid(self, rng, size: int, obstacle_density: float) -> list[list[str]]:
        # Generate grid, place obstacles, ensure start→end reachable
        raise NotImplementedError

    def _bfs_shortest_path(self, grid, start, end) -> int | None:
        # Returns min steps or None if unreachable
        raise NotImplementedError

    def _grid_to_string(self, grid) -> str:
        # Format grid as readable ASCII for the question
        raise NotImplementedError

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        raise NotImplementedError

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        raise NotImplementedError
```

---

### Domain 6 — `vtask/domains/temporal.py`: Temporal Constraint Ordering

**Concept:** Given a set of events and constraints about their relative timing 
(before/after/during/simultaneous), determine valid orderings or detect contradictions.

**Why verifiable:** Constraint satisfaction on a DAG is exactly solvable.

**Difficulty ladder:**
- Level 1: 4 events, pure "before" constraints, "Which of these is a valid ordering?" (multiple choice, pick one letter)
- Level 2: 5 events, before/after constraints, "What is the earliest position event X can occupy?"
- Level 3: 6 events, ask "Can event A and event C happen simultaneously?" ("yes"/"no")
- Level 4: 7 events with some constraints contradictory, "Are these constraints satisfiable?" ("yes"/"no")
- Level 5: 8 events, continuous time with durations, "What is the earliest time event X can finish?"

```python
"""
Implement this fully.

Key implementation notes:
- Model constraints as a directed graph (A before B = edge A→B).
- Level 1: Generate 4 valid orderings where 1 is correct, 3 violate a constraint.
  Question presents 4 options (A/B/C/D), answer is the letter.
  score_answer: normalize letter, compare to correct letter in metadata.
- Level 2: Use topological sort levels to determine earliest position.
  Answer is 1-indexed position integer.
- Level 3: "Can A and C be simultaneous?" = can they both be in the same 
  topological layer? Yes if no directed path exists between them.
- Level 4: Contradiction = cycle in the constraint graph. 
  answer is "yes" (satisfiable) or "no" (contradiction found).
- Level 5: Use PERT/CPM with event durations and dependency constraints.
  Answer is a number (float formatted to 1 decimal or integer).
- Use real event names: meeting names, project phases, historical-sounding events
  ["kickoff", "review", "approval", "testing", "launch", "handoff", "audit"]
"""

import random
from vtask.base import TaskEntry, TaskGenerator


class TemporalGenerator(TaskGenerator):
    domain = "temporal"
    difficulty_range = (1, 5)

    def _has_cycle(self, edges: list[tuple], nodes: list) -> bool:
        raise NotImplementedError

    def _topo_levels(self, edges: list[tuple], nodes: list) -> dict[str, int]:
        # Returns {node: level} where level 0 = no prerequisites
        raise NotImplementedError

    def _has_path(self, edges: list[tuple], src: str, dst: str) -> bool:
        # Returns True if directed path exists from src to dst
        raise NotImplementedError

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        raise NotImplementedError

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        raise NotImplementedError
```

---

## Step 3: Registry and `__init__.py`

### `vtask/registry.py`

```python
"""
Central registry. Maps domain name strings to generator classes.
"""
from vtask.domains.scheduling import SchedulingGenerator
from vtask.domains.dependencies import DependencyGenerator
from vtask.domains.inventory import InventoryGenerator
from vtask.domains.fsm import FSMGenerator
from vtask.domains.spatial import SpatialGenerator
from vtask.domains.temporal import TemporalGenerator

REGISTRY = {
    "scheduling": SchedulingGenerator,
    "dependencies": DependencyGenerator,
    "inventory": InventoryGenerator,
    "fsm": FSMGenerator,
    "spatial": SpatialGenerator,
    "temporal": TemporalGenerator,
}

def create_dataset(domain: str, size: int, seed: int = 42, difficulty: int | None = None):
    if domain not in REGISTRY:
        raise ValueError(f"Unknown domain '{domain}'. Available: {list(REGISTRY.keys())}")
    generator = REGISTRY[domain]()
    return generator.create_dataset(size=size, seed=seed, difficulty=difficulty)

def list_domains() -> list[str]:
    return list(REGISTRY.keys())
```

### `vtask/__init__.py`

```python
from vtask.registry import create_dataset, list_domains
from vtask.base import TaskEntry, TaskDataset, TaskGenerator

__version__ = "0.1.0"
__all__ = ["create_dataset", "list_domains", "TaskEntry", "TaskDataset", "TaskGenerator"]
```

---

## Step 4: The Evaluation Harness

### `vtask/harness.py`

```python
"""
Model evaluation harness. Given any callable model(question: str) -> str,
runs it against a dataset and returns a performance report.

Supports:
- Direct callable (for testing with rule-based solvers)
- Anthropic API via the anthropic Python SDK
- Any OpenAI-compatible API
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
    by_difficulty: dict[int, dict]  # {difficulty: {total, correct, accuracy}}
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
        lines.append('='*50)
        return "\n".join(lines)


def evaluate(
    model_fn: Callable[[str], str],
    domain: str,
    size: int = 100,
    seed: int = 42,
    difficulty: int | None = None,
    verbose: bool = False,
) -> EvalResult:
    """
    Run model_fn against a generated dataset and return EvalResult.
    
    Args:
        model_fn: A callable that takes a question string, returns answer string.
        domain: One of the registered domain names.
        size: Number of tasks to generate.
        seed: Random seed for reproducibility.
        difficulty: If set, only generate tasks at this difficulty level.
        verbose: Print each question/answer/score as it runs.
    """
    import vtask
    dataset = vtask.create_dataset(domain, size=size, seed=seed, difficulty=difficulty)

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


def make_anthropic_model(api_key: str, model: str = "claude-sonnet-4-20250514",
                          system_prompt: str | None = None) -> Callable[[str], str]:
    """
    Returns a model_fn compatible with evaluate() that calls the Anthropic API.
    Install: pip install anthropic
    """
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
```

---

## Step 5: CLI scripts

### `scripts/generate_dataset.py`

```python
#!/usr/bin/env python3
"""
Generate a JSONL dataset file for one or all domains.

Usage:
  python scripts/generate_dataset.py --domain scheduling --size 1000 --output data/scheduling.jsonl
  python scripts/generate_dataset.py --all --size 500 --output-dir data/
"""
import argparse
import os
import vtask


def main():
    parser = argparse.ArgumentParser(description="Generate vtask dataset")
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
        for domain in vtask.list_domains():
            path = os.path.join(args.output_dir or "data", f"{domain}.jsonl")
            print(f"Generating {args.size} tasks for domain '{domain}'...")
            ds = vtask.create_dataset(domain, size=args.size, seed=args.seed,
                                       difficulty=args.difficulty)
            ds.to_jsonl(path)
            print(f"  → Saved to {path}")
    elif args.domain:
        path = args.output or f"{args.domain}.jsonl"
        print(f"Generating {args.size} tasks for domain '{args.domain}'...")
        ds = vtask.create_dataset(args.domain, size=args.size, seed=args.seed,
                                   difficulty=args.difficulty)
        ds.to_jsonl(path)
        print(f"Saved to {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

### `scripts/run_eval.py`

```python
#!/usr/bin/env python3
"""
Evaluate a model against vtask domains.

Usage:
  # Anthropic API
  python scripts/run_eval.py --api anthropic --api-key sk-ant-... --domain scheduling --size 50
  
  # All domains
  python scripts/run_eval.py --api anthropic --api-key sk-ant-... --all --size 100
"""
import argparse
import vtask
from vtask.harness import evaluate, make_anthropic_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", choices=["anthropic"], default="anthropic")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--domain", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_fn = make_anthropic_model(api_key=args.api_key, model=args.model)

    domains = vtask.list_domains() if args.all else [args.domain]

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
```

---

## Step 6: `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vtask"
version = "0.1.0"
description = "Multi-domain verifiable task engine for RL training data generation"
requires-python = ">=3.10"
dependencies = [
    "networkx>=3.0",
]

[project.optional-dependencies]
eval = ["anthropic>=0.40.0"]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.hatch.build.targets.wheel]
packages = ["vtask"]
```

---

## Step 7: Example usage files

### `examples/basic_usage.py`

```python
"""
Demonstrates core vtask usage without any API calls.
Run: python examples/basic_usage.py
"""
import vtask

# List available domains
print("Available domains:", vtask.list_domains())

# Generate a small scheduling dataset
ds = vtask.create_dataset("scheduling", size=5, seed=42)
for entry in ds:
    print(f"\n[{entry.domain.upper()} | Difficulty {entry.difficulty}]")
    print(f"Q: {entry.question}")
    print(f"A: {entry.answer}")
    print(f"Distractors: {entry.distractors}")

    # Test the verifier
    score_correct = ds.score_answer(entry.answer, entry)
    score_wrong = ds.score_answer(entry.distractors[0] if entry.distractors else "999", entry)
    print(f"Correct answer scores: {score_correct}")
    print(f"Wrong answer scores:   {score_wrong}")

# Generate all domains mixed, export to JSONL
print("\n\nGenerating 200-task mixed dataset...")
all_tasks = []
for domain in vtask.list_domains():
    ds = vtask.create_dataset(domain, size=33, seed=123)
    all_tasks.extend(ds.entries)

print(f"Total tasks: {len(all_tasks)}")
print(f"Domains covered: {set(e.domain for e in all_tasks)}")
```

### `examples/anthropic_eval.py`

```python
"""
Evaluate Claude against all vtask domains and print a performance report.
Run: python examples/anthropic_eval.py --api-key sk-ant-...
"""
import argparse
import vtask
from vtask.harness import evaluate, make_anthropic_model

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)
parser.add_argument("--model", default="claude-sonnet-4-20250514")
parser.add_argument("--size", type=int, default=50,
                    help="Tasks per domain (keep low to control API cost)")
args = parser.parse_args()

model_fn = make_anthropic_model(api_key=args.api_key, model=args.model)

print(f"\nEvaluating {args.model} across {len(vtask.list_domains())} domains")
print(f"Tasks per domain: {args.size}")
print(f"Total API calls: ~{args.size * len(vtask.list_domains())}\n")

results = {}
for domain in vtask.list_domains():
    result = evaluate(model_fn, domain=domain, size=args.size, seed=42)
    results[domain] = result
    print(result)

# Summary table
print("\n\nSUMMARY")
print(f"{'Domain':<20} {'Accuracy':>10} {'Correct':>10}")
print("-" * 42)
for domain, r in results.items():
    print(f"{domain:<20} {r.accuracy:>9.1%} {r.correct:>5}/{r.total}")

overall_acc = sum(r.correct for r in results.values()) / sum(r.total for r in results.values())
print("-" * 42)
print(f"{'OVERALL':<20} {overall_acc:>9.1%}")
```

---

## Step 8: Tests

Write one test file per domain. Each test should:
1. Verify the generator produces valid TaskEntry objects
2. Verify score_answer returns 1.0 for the correct answer
3. Verify score_answer returns 0.0 for each distractor
4. Verify score_answer returns 0.0 for "garbage input"
5. Verify determinism: same seed + difficulty always produces identical question
6. Verify difficulty scaling: difficulty=5 tasks are harder to brute-force than difficulty=1

### `tests/test_scheduling.py` (write this pattern for all 6 domains)

```python
import pytest
import vtask
from vtask.domains.scheduling import SchedulingGenerator


def test_generates_valid_entry():
    gen = SchedulingGenerator()
    entry = gen.generate(seed=1, difficulty=1)
    assert entry.question
    assert entry.answer
    assert entry.domain == "scheduling"
    assert 1 <= entry.difficulty <= 5


def test_correct_answer_scores_1():
    gen = SchedulingGenerator()
    for seed in range(10):
        entry = gen.generate(seed=seed, difficulty=2)
        assert gen.score_answer(entry.answer, entry) == 1.0


def test_distractors_score_0():
    gen = SchedulingGenerator()
    entry = gen.generate(seed=42, difficulty=2)
    for d in entry.distractors:
        assert gen.score_answer(d, entry) == 0.0, f"Distractor '{d}' incorrectly scored 1.0"


def test_garbage_scores_0():
    gen = SchedulingGenerator()
    entry = gen.generate(seed=42, difficulty=1)
    assert gen.score_answer("not a number", entry) == 0.0
    assert gen.score_answer("", entry) == 0.0
    assert gen.score_answer("-999", entry) == 0.0


def test_determinism():
    gen = SchedulingGenerator()
    e1 = gen.generate(seed=77, difficulty=3)
    e2 = gen.generate(seed=77, difficulty=3)
    assert e1.question == e2.question
    assert e1.answer == e2.answer


def test_dataset_creation():
    ds = vtask.create_dataset("scheduling", size=20, seed=0)
    assert len(ds) == 20
    for entry in ds:
        assert entry.domain == "scheduling"
        assert gen.score_answer(entry.answer, entry) == 1.0  # All answers must be correct
```

---

## Implementation order for Claude Code

Build in this exact order to maximize working code at each checkpoint:

1. `vtask/base.py` — foundations, no dependencies
2. `vtask/domains/fsm.py` — you have prior work here, start where you're strong
3. `tests/test_fsm.py` — get tests passing before moving on
4. `vtask/domains/dependencies.py` — clean graph problem, networkx makes it easy
5. `tests/test_dependencies.py`
6. `vtask/domains/spatial.py` — BFS is well-understood
7. `tests/test_spatial.py`
8. `vtask/domains/temporal.py` — builds on dependency graph concepts
9. `tests/test_temporal.py`
10. `vtask/domains/scheduling.py` — hardest verifier, do last
11. `vtask/domains/inventory.py` — straightforward simulation
12. `tests/test_scheduling.py` + `tests/test_inventory.py`
13. `vtask/registry.py` + `vtask/__init__.py`
14. `vtask/harness.py`
15. `scripts/` + `examples/`
16. `pyproject.toml`
17. `README.md` (last)

Run `pytest tests/ -v` after every domain to catch regressions immediately.

---

## Quality bar

Before considering the project done, verify:

- [ ] All 6 domains generate tasks at all 5 difficulty levels without errors
- [ ] `pytest tests/ -v` passes with 0 failures
- [ ] `python examples/basic_usage.py` runs without API keys
- [ ] Every domain's `score_answer(correct_answer, entry) == 1.0` for 100 random seeds
- [ ] Every domain's `score_answer(distractor, entry) == 0.0` for all distractors
- [ ] Generating 10,000 tasks across all domains takes < 30 seconds (performance)
- [ ] JSONL output is valid JSON on every line

---

## What to tell Ambitus

This is a **domain-novel RL training environment** targeting professional workflow 
reasoning — scheduling, logistics, project dependencies, spatial navigation, temporal 
constraints, and formal computation (FSM). It fills a specific gap: reasoning-gym 
(NeurIPS 2025) covers math/logic/games but not professional/enterprise task domains.

Every task is:
- **Procedurally generated** — infinite unique instances, no memorization possible
- **Difficulty-calibrated** — 5 levels per domain, enabling curriculum learning
- **Algorithmically verified** — zero LLM judges, zero human review required
- **Distractor-equipped** — for multiple-choice evaluation mode

Output is compatible with any RL training framework. Includes a ready-to-run 
evaluation harness for the Anthropic API.