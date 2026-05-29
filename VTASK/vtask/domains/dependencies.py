"""
Domain 2: Project Dependency Resolution
"""
from __future__ import annotations
import random
from collections import defaultdict, deque
from vtask.base import TaskEntry, TaskGenerator

TASK_NAMES = [
    "Design", "Prototype", "Test", "Review", "Deploy", "Document",
    "QA", "Integrate", "Configure", "Build", "Plan", "Validate",
    "Analyze", "Implement", "Audit", "Migrate", "Refactor", "Package",
]


class DependencyGenerator(TaskGenerator):
    domain = "dependencies"
    difficulty_range = (1, 5)

    def _build_dag(self, rng: random.Random, n_tasks: int, task_names: list,
                   durations: dict) -> list[tuple]:
        """Build a random DAG (no cycles). Returns list of (src, dst) edges."""
        tasks = task_names[:n_tasks]
        edges = []
        for i in range(1, n_tasks):
            # Each task depends on at least 1 earlier task (by index)
            n_deps = rng.randint(1, min(2, i))
            parents = rng.sample(tasks[:i], n_deps)
            for p in parents:
                edges.append((p, tasks[i]))
        # Remove duplicate edges
        edges = list(set(edges))
        return edges

    def _critical_path(self, tasks: list, edges: list[tuple], durations: dict) -> tuple[int, list[str]]:
        """Returns (critical_path_length, list_of_tasks_on_critical_path)."""
        # Topological sort
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for src, dst in edges:
            adj[src].append(dst)
            in_degree[dst] += 1
        for t in tasks:
            if t not in in_degree:
                in_degree[t] = 0

        queue = deque([t for t in tasks if in_degree[t] == 0])
        earliest = {t: durations[t] for t in tasks}
        predecessor = {t: None for t in tasks}
        topo = []

        while queue:
            node = queue.popleft()
            topo.append(node)
            for nxt in adj[node]:
                candidate = earliest[node] + durations[nxt]
                if candidate > earliest[nxt]:
                    earliest[nxt] = candidate
                    predecessor[nxt] = node
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        cp_len = max(earliest.values())
        # Find the end node(s) with max earliest finish
        end_nodes = [t for t in tasks if earliest[t] == cp_len]
        # Trace back critical path
        path = []
        node = end_nodes[0]
        while node is not None:
            path.append(node)
            node = predecessor[node]
        path.reverse()
        return cp_len, path

    def _topo_levels(self, tasks: list, edges: list[tuple]) -> dict[str, int]:
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        for src, dst in edges:
            adj[src].append(dst)
            in_degree[dst] += 1
        for t in tasks:
            if t not in in_degree:
                in_degree[t] = 0

        queue = deque([t for t in tasks if in_degree[t] == 0])
        level = {t: 0 for t in tasks}
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

    def _is_valid_topo_order(self, ordering: list[str], edges: list[tuple]) -> bool:
        pos = {t: i for i, t in enumerate(ordering)}
        for src, dst in edges:
            if src not in pos or dst not in pos:
                return False
            if pos[src] >= pos[dst]:
                return False
        return True

    def _ancestors(self, tasks: list, edges: list[tuple], target: str) -> set:
        """Return all tasks that must complete before target (direct or transitive)."""
        rev_adj = defaultdict(list)
        for src, dst in edges:
            rev_adj[dst].append(src)
        visited = set()
        queue = deque([target])
        while queue:
            node = queue.popleft()
            for prev in rev_adj[node]:
                if prev not in visited:
                    visited.add(prev)
                    queue.append(prev)
        return visited

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        rng = random.Random(seed)
        names = rng.sample(TASK_NAMES, len(TASK_NAMES))

        if difficulty == 1:
            n = 4
            tasks = names[:n]
            # Simple chain
            edges = [(tasks[i], tasks[i + 1]) for i in range(n - 1)]
            # Generate one valid ordering and 3 invalid ones
            valid_order = tasks[:]
            # Shuffle to get another valid order? For a chain, only one valid order exists.
            orderings = [valid_order[:]]
            for _ in range(3):
                bad = valid_order[:]
                # Swap two adjacent elements to violate a constraint
                i = rng.randint(0, n - 2)
                bad[i], bad[i + 1] = bad[i + 1], bad[i]
                orderings.append(bad)
            rng.shuffle(orderings)
            correct_idx = next(i for i, o in enumerate(orderings) if self._is_valid_topo_order(o, edges))
            letters = ["A", "B", "C", "D"]
            correct_letter = letters[correct_idx]
            dep_lines = "\n".join(f"- {dst} requires {src}" for src, dst in edges)
            ordering_lines = "\n".join(
                f"  {letters[i]}: {' → '.join(o)}" for i, o in enumerate(orderings)
            )
            question = (
                f"A project has the following tasks and dependencies:\n{dep_lines}\n\n"
                f"Which of the following is a valid order to complete all tasks?\n{ordering_lines}\n\n"
                f"Answer with just the letter (A, B, C, or D)."
            )
            answer = correct_letter
            distractors = [l for l in letters if l != correct_letter]
            metadata = {
                "tasks": tasks,
                "edges": edges,
                "orderings": orderings,
                "correct_letter": correct_letter,
                "mode": "ordering",
            }

        elif difficulty == 2:
            n = 6
            tasks = names[:n]
            durations = {t: rng.randint(1, 8) for t in tasks}
            edges = self._build_dag(rng, n, tasks, durations)
            cp_len, cp_tasks = self._critical_path(tasks, edges, durations)
            dep_lines = []
            for t in tasks:
                prereqs = [s for s, d in edges if d == t]
                prereq_str = ", ".join(prereqs) if prereqs else "none"
                dep_lines.append(f"- {t}: {durations[t]} days (requires: {prereq_str})")
            question = (
                f"A software project has the following tasks and dependencies:\n"
                + "\n".join(dep_lines)
                + f"\n\nWhat is the minimum number of days to complete the entire project?"
            )
            answer = str(cp_len)
            distractors = [str(cp_len + 1), str(cp_len + 2), str(cp_len - 1)]
            distractors = [d for d in distractors if d != answer and int(d) > 0]
            metadata = {
                "tasks": tasks,
                "edges": edges,
                "durations": durations,
                "correct_cp": cp_len,
                "mode": "critical_path_length",
            }

        elif difficulty == 3:
            n = 8
            tasks = names[:n]
            durations = {t: rng.randint(1, 6) for t in tasks}
            edges = self._build_dag(rng, n, tasks, durations)
            target = rng.choice(tasks[2:])  # Pick a non-root task
            ancestors = self._ancestors(tasks, edges, target)
            # "Which tasks CAN'T run before task X" = ancestors of X (must complete before X)
            answer_list = sorted(ancestors)
            answer = ", ".join(answer_list)
            dep_lines = []
            for t in tasks:
                prereqs = [s for s, d in edges if d == t]
                prereq_str = ", ".join(prereqs) if prereqs else "none"
                dep_lines.append(f"- {t} (requires: {prereq_str})")
            question = (
                f"A project has the following tasks and dependencies:\n"
                + "\n".join(dep_lines)
                + f"\n\nWhich tasks must be completed before \"{target}\" can start? "
                f"List them in alphabetical order, separated by commas. "
                f"If none, write \"none\"."
            )
            if not answer_list:
                answer = "none"
            metadata = {
                "tasks": tasks,
                "edges": edges,
                "target": target,
                "ancestors": sorted(ancestors),
                "mode": "prerequisites",
            }
            distractors = []

        elif difficulty == 4:
            n = 10
            tasks = names[:n]
            durations = {t: rng.randint(2, 10) for t in tasks}
            edges = self._build_dag(rng, n, tasks, durations)
            cp_len, cp_tasks = self._critical_path(tasks, edges, durations)
            dep_lines = []
            for t in tasks:
                prereqs = [s for s, d in edges if d == t]
                prereq_str = ", ".join(prereqs) if prereqs else "none"
                dep_lines.append(f"- {t}: {durations[t]} days (requires: {prereq_str})")
            question = (
                f"A project has the following tasks and dependencies:\n"
                + "\n".join(dep_lines)
                + f"\n\nWhat is the minimum number of days to complete the entire project?"
            )
            answer = str(cp_len)
            distractors = [str(cp_len + 2), str(cp_len + 3), str(cp_len - 1)]
            distractors = [d for d in distractors if d != answer and int(d) > 0]
            metadata = {
                "tasks": tasks,
                "edges": edges,
                "durations": durations,
                "correct_cp": cp_len,
                "mode": "critical_path_length",
            }

        else:  # difficulty 5
            n = 12
            tasks = names[:n]
            durations = {t: rng.randint(1, 8) for t in tasks}
            edges = self._build_dag(rng, n, tasks, durations)
            cp_len, cp_tasks = self._critical_path(tasks, edges, durations)
            dep_lines = []
            for t in tasks:
                prereqs = [s for s, d in edges if d == t]
                prereq_str = ", ".join(prereqs) if prereqs else "none"
                dep_lines.append(f"- {t}: {durations[t]} days (requires: {prereq_str})")
            answer = ", ".join(sorted(cp_tasks))
            question = (
                f"A project has the following tasks and dependencies:\n"
                + "\n".join(dep_lines)
                + f"\n\nList all tasks on the critical path (the longest sequence of dependent tasks). "
                f"Provide them in alphabetical order, separated by commas."
            )
            metadata = {
                "tasks": tasks,
                "edges": edges,
                "durations": durations,
                "correct_cp": cp_len,
                "critical_path_tasks": sorted(cp_tasks),
                "mode": "critical_path_tasks",
            }
            distractors = []

        return TaskEntry(
            question=question,
            answer=answer,
            distractors=distractors,
            difficulty=difficulty,
            domain=self.domain,
            metadata=metadata,
            task_id="",
        )

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        mode = entry.metadata.get("mode", "ordering")
        answer = answer.strip()

        if mode == "ordering":
            return 1.0 if answer.upper() == entry.metadata["correct_letter"] else 0.0

        elif mode == "critical_path_length":
            try:
                return 1.0 if int(answer) == entry.metadata["correct_cp"] else 0.0
            except ValueError:
                return 0.0

        elif mode == "prerequisites":
            correct = entry.metadata["ancestors"]
            if not correct:
                return 1.0 if answer.lower().strip() == "none" else 0.0
            proposed = [x.strip() for x in answer.split(",") if x.strip()]
            return 1.0 if sorted(proposed) == sorted(correct) else 0.0

        elif mode == "critical_path_tasks":
            correct = entry.metadata["critical_path_tasks"]
            proposed = [x.strip() for x in answer.split(",") if x.strip()]
            return 1.0 if sorted(proposed) == sorted(correct) else 0.0

        return 0.0
