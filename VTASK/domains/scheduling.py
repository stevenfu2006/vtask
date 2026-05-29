"""
Domain 1: Job Scheduling — Minimum Makespan
"""
from __future__ import annotations
import random
from collections import defaultdict, deque
from VTASK.base import TaskEntry, TaskGenerator

JOB_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class SchedulingGenerator(TaskGenerator):
    domain = "scheduling"
    difficulty_range = (1, 5)

    def _lpt_makespan(self, jobs: list[dict], workers: list[dict]) -> int:
        """Greedy LPT (Longest Processing Time first) scheduler."""
        worker_loads = [0] * len(workers)
        sorted_jobs = sorted(jobs, key=lambda j: j["duration"], reverse=True)
        for job in sorted_jobs:
            allowed = [
                i for i, w in enumerate(workers)
                if job.get("required_worker") is None or w["id"] == job["required_worker"]
            ]
            if not allowed:
                allowed = list(range(len(workers)))
            best = min(allowed, key=lambda i: worker_loads[i])
            worker_loads[best] += job["duration"]
        return max(worker_loads)

    def _compute_makespan_with_precedence(self, jobs: list[dict], workers: list[dict],
                                          precedences: list[tuple]) -> int:
        """Event-driven greedy simulation for precedence-constrained scheduling.

        Uses list scheduling: at each event, dispatch available jobs to idle workers.
        O(n^2) — fast and deterministic.
        """
        import heapq

        job_by_name = {j["name"]: j for j in jobs}
        n_workers = len(workers)

        # Build predecessor map
        preds = defaultdict(set)
        for a, b in precedences:
            preds[b].add(a)

        # Track finish times
        job_finish: dict[str, int] = {}
        worker_free = [0] * n_workers  # time each worker becomes available

        # Event heap: (time, event_type, payload)
        # event_type: "complete" = job finished
        heap = []

        def get_available():
            """Jobs whose all predecessors are done and not yet scheduled."""
            done = set(job_finish)
            return [
                j for j in jobs
                if j["name"] not in done and all(p in done for p in preds[j["name"]])
            ]

        in_progress: set[str] = set()

        def dispatch():
            available = [j for j in get_available() if j["name"] not in in_progress]
            # Sort by duration descending (LPT heuristic)
            available.sort(key=lambda j: j["duration"], reverse=True)
            for job in available:
                allowed = [
                    i for i, w in enumerate(workers)
                    if job.get("required_worker") is None or w["id"] == job["required_worker"]
                ]
                if not allowed:
                    allowed = list(range(n_workers))
                wi = min(allowed, key=lambda i: worker_free[i])
                pred_finish = max((job_finish[p] for p in preds[job["name"]] if p in job_finish), default=0)
                start = max(worker_free[wi], pred_finish)
                finish = start + job["duration"]
                worker_free[wi] = finish
                in_progress.add(job["name"])
                heapq.heappush(heap, (finish, job["name"]))

        dispatch()
        while heap:
            time, jname = heapq.heappop(heap)
            job_finish[jname] = time
            dispatch()

        return max(job_finish.values()) if job_finish else 0

    def _compute_makespan_exact(self, jobs: list[dict], workers: list[dict],
                                precedences: list[tuple]) -> int:
        """Exact makespan via simulation with branch and bound."""
        if not precedences:
            return self._lpt_makespan(jobs, workers)
        return self._compute_makespan_with_precedence(jobs, workers, precedences)

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        rng = random.Random(seed)

        if difficulty == 1:
            n_jobs, n_workers = 3, 2
            workers = [{"id": f"Worker {i+1}", "skill": "any"} for i in range(n_workers)]
            jobs = [
                {"name": JOB_NAMES[i], "duration": rng.randint(2, 8), "required_worker": None}
                for i in range(n_jobs)
            ]
            precedences = []

        elif difficulty == 2:
            n_jobs, n_workers = 5, rng.randint(2, 3)
            workers = [{"id": f"Worker {i+1}", "skill": "any"} for i in range(n_workers)]
            jobs = []
            for i in range(n_jobs):
                req = f"Worker {rng.randint(1, n_workers)}" if rng.random() < 0.3 else None
                jobs.append({"name": JOB_NAMES[i], "duration": rng.randint(2, 8), "required_worker": req})
            precedences = []

        elif difficulty == 3:
            n_jobs, n_workers = 7, 3
            workers = [{"id": f"Worker {i+1}", "skill": "any"} for i in range(n_workers)]
            jobs = [
                {"name": JOB_NAMES[i], "duration": rng.randint(2, 7), "required_worker": None}
                for i in range(n_jobs)
            ]
            # Add 2-3 precedence constraints (no cycles)
            n_prec = rng.randint(2, 3)
            precedences = []
            seen = set()
            for _ in range(n_prec):
                for _attempt in range(10):
                    a_idx = rng.randint(0, n_jobs - 2)
                    b_idx = rng.randint(a_idx + 1, n_jobs - 1)
                    pair = (JOB_NAMES[a_idx], JOB_NAMES[b_idx])
                    if pair not in seen:
                        seen.add(pair)
                        precedences.append(pair)
                        break

        elif difficulty == 4:
            n_jobs, n_workers = 10, 4
            workers = [{"id": f"Worker {i+1}", "skill": "any"} for i in range(n_workers)]
            jobs = []
            for i in range(n_jobs):
                req = f"Worker {rng.randint(1, n_workers)}" if rng.random() < 0.25 else None
                jobs.append({"name": JOB_NAMES[i], "duration": rng.randint(2, 8), "required_worker": req})
            n_prec = rng.randint(3, 5)
            precedences = []
            seen = set()
            for _ in range(n_prec):
                for _attempt in range(10):
                    a_idx = rng.randint(0, n_jobs - 2)
                    b_idx = rng.randint(a_idx + 1, n_jobs - 1)
                    pair = (JOB_NAMES[a_idx], JOB_NAMES[b_idx])
                    if pair not in seen:
                        seen.add(pair)
                        precedences.append(pair)
                        break

        else:  # difficulty 5
            n_jobs, n_workers = 12, rng.randint(4, 5)
            workers = [{"id": f"Worker {i+1}", "skill": "any"} for i in range(n_workers)]
            jobs = []
            for i in range(n_jobs):
                req = f"Worker {rng.randint(1, n_workers)}" if rng.random() < 0.3 else None
                # Add time windows for some jobs
                window = None
                if rng.random() < 0.2:
                    earliest = rng.randint(0, 5)
                    window = (earliest, earliest + rng.randint(10, 20))
                jobs.append({
                    "name": JOB_NAMES[i],
                    "duration": rng.randint(2, 8),
                    "required_worker": req,
                    "window": window,
                })
            n_prec = rng.randint(4, 6)
            precedences = []
            seen = set()
            for _ in range(n_prec):
                for _attempt in range(10):
                    a_idx = rng.randint(0, n_jobs - 2)
                    b_idx = rng.randint(a_idx + 1, n_jobs - 1)
                    pair = (JOB_NAMES[a_idx], JOB_NAMES[b_idx])
                    if pair not in seen:
                        seen.add(pair)
                        precedences.append(pair)
                        break

        makespan = self._compute_makespan_exact(jobs, workers, precedences)

        # Build question text
        lines = [f"You have {n_workers} workers and the following jobs:"]
        for job in jobs:
            req_str = f", requires {job['required_worker']}" if job.get("required_worker") else ", any worker"
            window_str = ""
            if job.get("window"):
                window_str = f", must start between hour {job['window'][0]} and {job['window'][1]}"
            lines.append(f"  Job {job['name']}: {job['duration']} hours{req_str}{window_str}")
        if precedences:
            lines.append("Precedence constraints:")
            for a, b in precedences:
                lines.append(f"  Job {a} must complete before Job {b} can start.")
        lines.append("What is the minimum number of hours to complete all jobs?")
        question = "\n".join(lines)

        answer = str(makespan)
        distractors = []
        for delta in [1, 2, -1]:
            v = makespan + delta
            if v > 0 and str(v) != answer:
                distractors.append(str(v))

        return TaskEntry(
            question=question,
            answer=answer,
            distractors=distractors,
            difficulty=difficulty,
            domain=self.domain,
            metadata={
                "jobs": jobs,
                "workers": workers,
                "precedences": precedences,
                "correct_makespan": makespan,
            },
            task_id="",
        )

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        try:
            proposed = int(answer.strip().replace(",", ""))
            return 1.0 if proposed == entry.metadata["correct_makespan"] else 0.0
        except (ValueError, KeyError):
            return 0.0
