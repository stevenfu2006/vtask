"""
Domain 6: Temporal Constraint Ordering
"""
from __future__ import annotations
import random
from collections import defaultdict, deque
from vtask.base import TaskEntry, TaskGenerator

EVENT_NAMES = [
    "kickoff", "review", "approval", "testing", "launch", "handoff", "audit",
    "planning", "design", "implementation", "deployment", "monitoring",
    "evaluation", "presentation", "signoff",
]


class TemporalGenerator(TaskGenerator):
    domain = "temporal"
    difficulty_range = (1, 5)

    def _has_cycle(self, edges: list[tuple], nodes: list) -> bool:
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for src, dst in edges:
            adj[src].append(dst)
            in_degree[dst] += 1
        for n in nodes:
            if n not in in_degree:
                in_degree[n] = 0
        queue = deque([n for n in nodes if in_degree[n] == 0])
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for nxt in adj[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        return visited != len(nodes)

    def _topo_levels(self, edges: list[tuple], nodes: list) -> dict[str, int]:
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for src, dst in edges:
            adj[src].append(dst)
            in_degree[dst] += 1
        for n in nodes:
            if n not in in_degree:
                in_degree[n] = 0
        queue = deque([n for n in nodes if in_degree[n] == 0])
        level = {n: 0 for n in nodes}
        while queue:
            node = queue.popleft()
            for nxt in adj[node]:
                level[nxt] = max(level[nxt], level[node] + 1)
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        return level

    def _has_path(self, edges: list[tuple], src: str, dst: str) -> bool:
        adj = defaultdict(list)
        for s, d in edges:
            adj[s].append(d)
        visited = set()
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                return True
            for nxt in adj[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def _build_dag_edges(self, rng: random.Random, nodes: list, density: float = 0.4) -> list[tuple]:
        """Build a random DAG by only adding forward edges."""
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if rng.random() < density:
                    edges.append((nodes[i], nodes[j]))
        return edges

    def _earliest_position(self, edges: list[tuple], nodes: list, target: str) -> int:
        """1-indexed position of target in topological order."""
        levels = self._topo_levels(edges, nodes)
        return levels[target] + 1  # 1-indexed

    def _pert_earliest_finish(self, nodes: list, edges: list[tuple], durations: dict) -> dict[str, float]:
        """PERT/CPM earliest finish times."""
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for src, dst in edges:
            adj[src].append(dst)
            in_degree[dst] += 1
        for n in nodes:
            if n not in in_degree:
                in_degree[n] = 0
        queue = deque([(n, durations[n]) for n in nodes if in_degree[n] == 0])
        earliest_finish = {n: durations[n] for n in nodes}
        while queue:
            node, ef = queue.popleft()
            earliest_finish[node] = max(earliest_finish[node], ef)
            for nxt in adj[node]:
                candidate = earliest_finish[node] + durations[nxt]
                if candidate > earliest_finish[nxt]:
                    earliest_finish[nxt] = candidate
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append((nxt, earliest_finish[nxt]))
        return earliest_finish

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        rng = random.Random(seed)
        events = rng.sample(EVENT_NAMES, len(EVENT_NAMES))

        if difficulty == 1:
            n = 4
            nodes = events[:n]
            # Build a simple DAG with clear ordering
            edges = self._build_dag_edges(rng, nodes, density=0.5)
            # Ensure it's acyclic (it should be by construction)
            levels = self._topo_levels(edges, nodes)
            valid_order = sorted(nodes, key=lambda x: levels[x])

            # Generate 4 orderings (1 valid, 3 invalid)
            orderings = [valid_order[:]]
            attempts = 0
            while len(orderings) < 4 and attempts < 50:
                attempts += 1
                candidate = nodes[:]
                rng.shuffle(candidate)
                if not self._is_valid_order(candidate, edges) and candidate not in orderings:
                    orderings.append(candidate)
            while len(orderings) < 4:
                bad = valid_order[:]
                if len(bad) >= 2:
                    i, j = rng.sample(range(len(bad)), 2)
                    bad[i], bad[j] = bad[j], bad[i]
                    if not self._is_valid_order(bad, edges):
                        orderings.append(bad)
                    else:
                        orderings.append(bad[::-1])
                else:
                    orderings.append(bad[::-1])

            rng.shuffle(orderings)
            letters = ["A", "B", "C", "D"]
            correct_idx = next(i for i, o in enumerate(orderings) if self._is_valid_order(o, edges))
            correct_letter = letters[correct_idx]

            constraint_lines = "\n".join(f"- {s} must happen before {d}" for s, d in edges)
            ordering_lines = "\n".join(
                f"  {letters[i]}: {' → '.join(o)}" for i, o in enumerate(orderings[:4])
            )
            question = (
                f"A sequence of events has the following constraints:\n{constraint_lines}\n\n"
                f"Which of the following is a valid ordering of all events?\n{ordering_lines}\n\n"
                f"Answer with just the letter (A, B, C, or D)."
            )
            answer = correct_letter
            distractors = [l for l in letters if l != correct_letter]
            metadata = {
                "nodes": nodes,
                "edges": edges,
                "orderings": orderings[:4],
                "correct_letter": correct_letter,
                "mode": "valid_ordering",
            }

        elif difficulty == 2:
            n = 5
            nodes = events[:n]
            edges = self._build_dag_edges(rng, nodes, density=0.45)
            target = rng.choice(nodes)
            levels = self._topo_levels(edges, nodes)
            pos = levels[target] + 1  # 1-indexed
            answer = str(pos)
            constraint_lines = "\n".join(f"- {s} must happen before {d}" for s, d in edges)
            question = (
                f"A sequence of events has the following ordering constraints:\n{constraint_lines}\n\n"
                f"Events: {', '.join(nodes)}\n\n"
                f"What is the earliest position (1 = first) that \"{target}\" can occupy "
                f"in a valid ordering of all events? Answer with just the number."
            )
            distractors = [str(pos + 1), str(pos + 2), str(max(1, pos - 1))]
            distractors = [d for d in distractors if d != answer]
            metadata = {
                "nodes": nodes,
                "edges": edges,
                "target": target,
                "correct_position": pos,
                "mode": "earliest_position",
            }

        elif difficulty == 3:
            n = 6
            nodes = events[:n]
            edges = self._build_dag_edges(rng, nodes, density=0.35)
            # Pick two events
            a, b = rng.sample(nodes, 2)
            # Can they be simultaneous? Yes if no directed path between them in either direction
            path_a_b = self._has_path(edges, a, b)
            path_b_a = self._has_path(edges, b, a)
            can_simultaneous = not path_a_b and not path_b_a
            answer = "yes" if can_simultaneous else "no"
            constraint_lines = "\n".join(f"- {s} must happen before {d}" for s, d in edges)
            question = (
                f"A sequence of events has the following constraints:\n{constraint_lines}\n\n"
                f"Can \"{a}\" and \"{b}\" happen simultaneously (at the same time)? "
                f"Answer with just \"yes\" or \"no\"."
            )
            distractors = ["no" if can_simultaneous else "yes"]
            metadata = {
                "nodes": nodes,
                "edges": edges,
                "event_a": a,
                "event_b": b,
                "correct_answer": answer,
                "mode": "simultaneity",
            }

        elif difficulty == 4:
            n = 7
            nodes = events[:n]
            # Sometimes introduce a cycle for unsatisfiable case
            make_cycle = rng.random() < 0.4
            if make_cycle:
                edges = self._build_dag_edges(rng, nodes, density=0.3)
                # Add a back edge to create a cycle
                if len(nodes) >= 2:
                    i, j = rng.sample(range(len(nodes)), 2)
                    if i > j:
                        i, j = j, i
                    edges.append((nodes[j], nodes[i]))
                satisfiable = not self._has_cycle(edges, nodes)
            else:
                edges = self._build_dag_edges(rng, nodes, density=0.35)
                satisfiable = not self._has_cycle(edges, nodes)

            answer = "yes" if satisfiable else "no"
            constraint_lines = "\n".join(f"- {s} must happen before {d}" for s, d in edges)
            question = (
                f"A project has the following temporal constraints:\n{constraint_lines}\n\n"
                f"Are these constraints satisfiable (can all events be scheduled without contradiction)? "
                f"Answer with just \"yes\" or \"no\"."
            )
            distractors = ["no" if satisfiable else "yes"]
            metadata = {
                "nodes": nodes,
                "edges": edges,
                "satisfiable": satisfiable,
                "correct_answer": answer,
                "mode": "satisfiability",
            }

        else:  # difficulty 5
            n = 8
            nodes = events[:n]
            edges = self._build_dag_edges(rng, nodes, density=0.4)
            durations = {e: rng.randint(1, 10) for e in nodes}
            target = rng.choice(nodes)
            ef = self._pert_earliest_finish(nodes, edges, durations)
            earliest_finish = ef[target]
            answer = str(earliest_finish)
            constraint_lines = []
            for e in nodes:
                prereqs = [s for s, d in edges if d == e]
                prereq_str = ", ".join(prereqs) if prereqs else "none"
                constraint_lines.append(f"- {e}: duration {durations[e]} days (after: {prereq_str})")
            question = (
                f"A project has the following events, durations, and ordering constraints:\n"
                + "\n".join(constraint_lines)
                + f"\n\nEach event can only start after all its prerequisites finish. "
                f"What is the earliest day that \"{target}\" can finish? "
                f"Answer with just the number."
            )
            distractors = [str(earliest_finish + 1), str(earliest_finish + 2), str(max(1, earliest_finish - 1))]
            distractors = [d for d in distractors if d != answer]
            metadata = {
                "nodes": nodes,
                "edges": edges,
                "durations": durations,
                "target": target,
                "correct_answer": earliest_finish,
                "mode": "earliest_finish",
            }

        return TaskEntry(
            question=question,
            answer=answer,
            distractors=distractors,
            difficulty=difficulty,
            domain=self.domain,
            metadata=metadata,
            task_id="",
        )

    def _is_valid_order(self, ordering: list[str], edges: list[tuple]) -> bool:
        pos = {n: i for i, n in enumerate(ordering)}
        for src, dst in edges:
            if src not in pos or dst not in pos:
                return False
            if pos[src] >= pos[dst]:
                return False
        return True

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        mode = entry.metadata.get("mode", "valid_ordering")
        answer = answer.strip()

        if mode == "valid_ordering":
            return 1.0 if answer.upper() == entry.metadata["correct_letter"] else 0.0

        elif mode == "earliest_position":
            try:
                return 1.0 if int(answer) == entry.metadata["correct_position"] else 0.0
            except ValueError:
                return 0.0

        elif mode in ("simultaneity", "satisfiability"):
            norm = answer.lower()
            return 1.0 if norm == entry.metadata["correct_answer"].lower() else 0.0

        elif mode == "earliest_finish":
            try:
                return 1.0 if int(answer) == entry.metadata["correct_answer"] else 0.0
            except ValueError:
                return 0.0

        return 0.0
