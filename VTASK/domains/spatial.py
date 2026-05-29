"""
Domain 5: Grid-based Spatial Reasoning
"""
from __future__ import annotations
import random
from collections import deque
import heapq
from VTASK.base import TaskEntry, TaskGenerator


class SpatialGenerator(TaskGenerator):
    domain = "spatial"
    difficulty_range = (1, 5)

    def _generate_grid(self, rng: random.Random, size: int, obstacle_density: float,
                       start=(0, 0), end=None) -> list[list[str]]:
        if end is None:
            end = (size - 1, size - 1)
        for _ in range(100):
            grid = [["." for _ in range(size)] for _ in range(size)]
            for r in range(size):
                for c in range(size):
                    if (r, c) not in (start, end) and rng.random() < obstacle_density:
                        grid[r][c] = "#"
            grid[start[0]][start[1]] = "S"
            grid[end[0]][end[1]] = "E"
            if self._bfs_shortest_path(grid, start, end, size) is not None:
                return grid
        # Fallback: clear path along top/right
        grid = [["." for _ in range(size)] for _ in range(size)]
        grid[start[0]][start[1]] = "S"
        grid[end[0]][end[1]] = "E"
        return grid

    def _bfs_shortest_path(self, grid, start, end, size=None) -> int | None:
        if size is None:
            size = len(grid)
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        queue = deque([(start[0], start[1], 0)])
        visited[start[0]][start[1]] = True
        while queue:
            r, c, dist = queue.popleft()
            if (r, c) == end:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != "#":
                    visited[nr][nc] = True
                    queue.append((nr, nc, dist + 1))
        return None

    def _dijkstra(self, grid, weights, start, end) -> int | None:
        rows, cols = len(grid), len(grid[0])
        dist = [[float("inf")] * cols for _ in range(rows)]
        dist[start[0]][start[1]] = 0
        heap = [(0, start[0], start[1])]
        while heap:
            d, r, c = heapq.heappop(heap)
            if (r, c) == end:
                return d
            if d > dist[r][c]:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
                    cost = weights.get((nr, nc), 1)
                    nd = d + cost
                    if nd < dist[nr][nc]:
                        dist[nr][nc] = nd
                        heapq.heappush(heap, (nd, nr, nc))
        return None

    def _bfs_with_waypoints(self, grid, start, end, waypoints: list) -> int | None:
        """BFS with state = (position, frozenset of visited waypoints)."""
        rows, cols = len(grid), len(grid[0])
        waypoint_set = frozenset(waypoints)
        start_visited = frozenset()
        if start in waypoint_set:
            start_visited = frozenset([start])
        queue = deque([(start[0], start[1], start_visited, 0)])
        seen = {(start[0], start[1], start_visited)}
        while queue:
            r, c, visited_wp, dist = queue.popleft()
            if (r, c) == end and visited_wp == waypoint_set:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#":
                    new_visited = visited_wp
                    if (nr, nc) in waypoint_set:
                        new_visited = visited_wp | frozenset([(nr, nc)])
                    key = (nr, nc, new_visited)
                    if key not in seen:
                        seen.add(key)
                        queue.append((nr, nc, new_visited, dist + 1))
        return None

    def _grid_to_string(self, grid) -> str:
        return "\n".join(" ".join(row) for row in grid)

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        rng = random.Random(seed)

        if difficulty == 1:
            size = 4
            start = (0, 0)
            end = (size - 1, size - 1)
            # No obstacles, only right/down moves
            steps = (end[0] - start[0]) + (end[1] - start[1])
            answer = str(steps)
            question = (
                f"You are on a {size}x{size} grid at position (row 0, col 0). "
                f"You need to reach (row {end[0]}, col {end[1]}). "
                f"You can only move right or down. "
                f"What is the minimum number of steps to reach the destination?"
            )
            distractors = [str(steps + 1), str(steps + 2), str(steps - 1)]
            distractors = [d for d in distractors if d != answer and int(d) > 0]
            metadata = {"correct_answer": steps, "mode": "steps_no_obstacles"}

        elif difficulty == 2:
            size = 6
            start = (0, 0)
            end = (size - 1, size - 1)
            grid = self._generate_grid(rng, size, obstacle_density=0.2, start=start, end=end)
            steps = self._bfs_shortest_path(grid, start, end)
            answer = str(steps)
            grid_str = self._grid_to_string(grid)
            question = (
                f"Navigate this {size}x{size} grid from S to E. '#' cells are walls.\n"
                f"You can move up, down, left, or right (not diagonally).\n\n"
                f"{grid_str}\n\n"
                f"What is the minimum number of steps to reach E?"
            )
            distractors = [str(steps + 1), str(steps + 2), str(steps + 3)]
            metadata = {
                "grid": grid,
                "start": list(start),
                "end": list(end),
                "correct_answer": steps,
                "mode": "bfs_shortest",
            }

        elif difficulty == 3:
            size = 8
            start = (0, 0)
            end = (size - 1, size - 1)
            # Higher obstacle density — may be unreachable
            for _ in range(20):
                grid = [["." for _ in range(size)] for _ in range(size)]
                for r in range(size):
                    for c in range(size):
                        if (r, c) not in (start, end) and rng.random() < 0.35:
                            grid[r][c] = "#"
                grid[start[0]][start[1]] = "S"
                grid[end[0]][end[1]] = "E"
                steps = self._bfs_shortest_path(grid, start, end)
                reachable = steps is not None
                if rng.random() < 0.4:  # Force some unreachable cases
                    break
                if reachable:
                    break
            answer = "yes" if reachable else "no"
            grid_str = self._grid_to_string(grid)
            question = (
                f"Navigate this {size}x{size} grid from S to E. '#' cells are walls.\n"
                f"You can move up, down, left, or right (not diagonally).\n\n"
                f"{grid_str}\n\n"
                f"Can you reach E from S? Answer with just \"yes\" or \"no\"."
            )
            distractors = ["no" if reachable else "yes"]
            metadata = {
                "grid": grid,
                "start": list(start),
                "end": list(end),
                "reachable": reachable,
                "correct_answer": answer,
                "mode": "reachability",
            }

        elif difficulty == 4:
            size = 8
            start = (0, 0)
            end = (size - 1, size - 1)
            grid = self._generate_grid(rng, size, obstacle_density=0.2, start=start, end=end)
            # Assign weights: some cells cost 2
            weights = {}
            heavy_cells = set()
            for r in range(size):
                for c in range(size):
                    if grid[r][c] == "." and rng.random() < 0.25:
                        weights[(r, c)] = 2
                        heavy_cells.add((r, c))
                        grid[r][c] = "2"
            # S and E always cost 1
            weights[start] = 1
            weights[end] = 1
            grid[start[0]][start[1]] = "S"
            grid[end[0]][end[1]] = "E"
            cost = self._dijkstra(grid, weights, start, end)
            answer = str(cost)
            grid_str = self._grid_to_string(grid)
            question = (
                f"Navigate this {size}x{size} grid from S to E. '#' cells are walls.\n"
                f"Moving through a '2' cell costs 2 steps. All other passable cells cost 1 step.\n"
                f"You can move up, down, left, or right (not diagonally).\n\n"
                f"{grid_str}\n\n"
                f"What is the minimum cost to reach E?"
            )
            distractors = [str(cost + 1), str(cost + 2), str(cost + 3)]
            metadata = {
                "grid": grid,
                "weights": {str(k): v for k, v in weights.items()},
                "start": list(start),
                "end": list(end),
                "correct_answer": cost,
                "mode": "weighted_path",
            }

        else:  # difficulty 5
            size = 10
            start = (0, 0)
            end = (size - 1, size - 1)
            # Place 2-3 waypoints
            n_waypoints = rng.randint(2, 3)
            grid = self._generate_grid(rng, size, obstacle_density=0.15, start=start, end=end)
            all_empty = [(r, c) for r in range(size) for c in range(size)
                         if grid[r][c] == "." and (r, c) not in (start, end)]
            waypoints = []
            wp_labels = ["W1", "W2", "W3"]
            for i in range(n_waypoints):
                if not all_empty:
                    break
                wp = rng.choice(all_empty)
                all_empty.remove(wp)
                waypoints.append(wp)
                grid[wp[0]][wp[1]] = wp_labels[i]

            steps = self._bfs_with_waypoints(grid, start, end, waypoints)
            if steps is None:
                # Fallback: clear a path
                grid = [["." for _ in range(size)] for _ in range(size)]
                grid[start[0]][start[1]] = "S"
                grid[end[0]][end[1]] = "E"
                waypoints = []
                steps = self._bfs_shortest_path(grid, start, end)

            answer = str(steps)
            grid_str = self._grid_to_string(grid)
            wp_desc = ", ".join(f"W{i+1}" for i in range(len(waypoints)))
            question = (
                f"Navigate this {size}x{size} grid from S to E. '#' cells are walls.\n"
                f"You must visit all waypoints ({wp_desc}) before reaching E.\n"
                f"You can move up, down, left, or right (not diagonally).\n\n"
                f"{grid_str}\n\n"
                f"What is the minimum number of steps to visit all waypoints and reach E?"
            )
            distractors = [str(steps + 1), str(steps + 2), str(steps + 3)]
            metadata = {
                "grid": grid,
                "start": list(start),
                "end": list(end),
                "waypoints": [list(wp) for wp in waypoints],
                "correct_answer": steps,
                "mode": "waypoints",
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

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        mode = entry.metadata.get("mode", "bfs_shortest")
        answer = answer.strip()

        if mode == "reachability":
            norm = answer.lower()
            return 1.0 if norm == entry.metadata["correct_answer"].lower() else 0.0

        try:
            proposed = int(answer.replace(",", ""))
            return 1.0 if proposed == entry.metadata["correct_answer"] else 0.0
        except ValueError:
            return 0.0
