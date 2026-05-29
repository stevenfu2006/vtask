# VTASK — Verifiable Task Engine

A Python library for generating unlimited parameterized reasoning tasks across 6 professional/workplace domains, where every answer is verifiable by algorithm with zero human or LLM involvement. Designed as an RL training environment.

## Why VTASK

This fills a specific gap: [reasoning-gym](https://github.com/open-thought/reasoning-gym) (NeurIPS 2025) covers math/logic/games but not professional/enterprise reasoning domains. VTASK targets:

- **Job scheduling** — minimum makespan under precedence and worker constraints
- **Project dependencies** — DAG analysis, critical path, topological ordering
- **Inventory/logistics** — day-by-day simulation of stock levels, reorder logic
- **Finite state machines** — DFA/NFA acceptance, string generation, symmetric difference
- **Spatial reasoning** — grid pathfinding with obstacles, weighted cells, waypoints
- **Temporal constraints** — ordering validation, satisfiability, PERT/CPM scheduling

Every task is:
- **Procedurally generated** — infinite unique instances, no memorization possible
- **Difficulty-calibrated** — 5 levels per domain (1=trivial, 5=complex), enabling curriculum learning
- **Algorithmically verified** — `score_answer` is pure Python, no LLM judges
- **Distractor-equipped** — plausible wrong answers for multiple-choice evaluation

## Install

```bash
pip install -e .
# With eval support (Anthropic API):
pip install -e ".[eval]"
# For development:
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

```python
import VTASK

# List available domains
print(VTASK.list_domains())
# ['scheduling', 'dependencies', 'inventory', 'fsm', 'spatial', 'temporal']

# Generate a dataset
ds = VTASK.create_dataset("scheduling", size=100, seed=42)

for entry in ds:
    print(entry.question)   # Natural language question
    print(entry.answer)     # Ground truth answer (always a string)
    print(entry.difficulty) # 1-5
    print(entry.distractors) # Plausible wrong answers

    # Verify any answer
    score = ds.score_answer("14", entry)  # 1.0 or 0.0

# Export to JSONL for training
ds.to_jsonl("scheduling_100.jsonl")
```

## Domains

### 1. Scheduling (`scheduling`)
N jobs assigned to M workers, find minimum makespan. Levels 1-5 add skill constraints, precedence constraints, and time windows.

```
Q: You have 3 workers and the following jobs:
   Job A: 4 hours, any worker
   Job B: 6 hours, any worker
   Job C: 3 hours, requires Worker 2 only
   Job B must complete before Job D can start.
   What is the minimum number of hours to complete all jobs?
A: 9
```

### 2. Dependencies (`dependencies`)
DAG-based project dependency resolution. Levels ask for valid orderings, critical path length, prerequisites, and critical path tasks.

### 3. Inventory (`inventory`)
Day-by-day inventory simulation with demand, reorders, lead times, seasonal demand, and multi-echelon supply chains.

### 4. FSM (`fsm`)
DFA/NFA acceptance questions, state tracing, accepting string generation, and symmetric difference detection.

### 5. Spatial (`spatial`)
Grid pathfinding: BFS shortest path, reachability, Dijkstra with weighted cells, and waypoint routing.

### 6. Temporal (`temporal`)
Temporal constraint satisfaction: valid orderings, earliest positions, simultaneity, satisfiability, and PERT/CPM finish times.

## Difficulty Levels

| Level | Description |
|-------|-------------|
| 1 | Trivial — minimal instances, no constraints |
| 2 | Simple — one additional constraint type |
| 3 | Moderate — multi-constraint reasoning |
| 4 | Hard — larger instances, compound constraints |
| 5 | Expert — maximum complexity per domain |

## Evaluation Harness

```python
from VTASK.harness import evaluate, make_anthropic_model

model_fn = make_anthropic_model(api_key="sk-ant-...", model="claude-sonnet-4-6")

result = evaluate(model_fn, domain="scheduling", size=100, seed=42, verbose=True)
print(result)
# Domain: scheduling
# Accuracy: 72.0% (72/100)
# By difficulty:
#   Level 1: 95.0% (19/20)
#   Level 5: 45.0% (9/20)
```

Any callable `str -> str` works as a model function:

```python
def my_model(question: str) -> str:
    # your model here
    return "42"

result = evaluate(my_model, domain="fsm", size=50)
```

## CLI

```bash
# Generate dataset
python scripts/generate_dataset.py --domain scheduling --size 1000 --output data/scheduling.jsonl
python scripts/generate_dataset.py --all --size 500 --output-dir data/

# Evaluate a model
python scripts/run_eval.py --api-key sk-ant-... --domain spatial --size 50 --verbose
python scripts/run_eval.py --api-key sk-ant-... --all --size 100
```

## Tests

```bash
pytest tests/ -v  # 41 tests, ~0.1s
```

## Performance

Generating 10,000 tasks across all domains: ~0.5 seconds.

## Interface Compatibility

Output format is compatible with [reasoning-gym](https://github.com/open-thought/reasoning-gym). The `TaskEntry` dataclass and `TaskDataset` follow the same patterns for easy integration with existing RL training pipelines.
