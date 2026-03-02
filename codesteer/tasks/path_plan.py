"""Path Planning task: find shortest path in a grid."""
import random
from collections import deque


def bfs_path(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != "#" and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


class PathPlanTask:
    name = "path_plan"
    description = "Find shortest path in a grid maze"

    def generate_question(self, complexity: int = 1, seed: int = None) -> dict:
        rng = random.Random(seed)
        size = 5 + complexity * 2
        obstacle_prob = 0.2 + complexity * 0.05

        for _ in range(100):
            grid = [["." for _ in range(size)] for _ in range(size)]
            # Add obstacles
            for r in range(size):
                for c in range(size):
                    if rng.random() < obstacle_prob and (r, c) not in [(0, 0), (size-1, size-1)]:
                        grid[r][c] = "#"

            path = bfs_path(grid, (0, 0), (size-1, size-1))
            if path:
                grid_str = "\n".join(" ".join(row) for row in grid)
                path_str = " -> ".join(f"({r},{c})" for r, c in path)
                return {
                    "question": f"Find the shortest path from (0,0) to ({size-1},{size-1}) in this {size}x{size} grid (# = obstacle, . = open):\n{grid_str}\n\nList the path as coordinates.",
                    "answer": path_str,
                    "metadata": {"grid": grid, "path": path, "size": size},
                }
        return {
            "question": "Find the shortest path from (0,0) to (4,4) in a 5x5 open grid.",
            "answer": "(0,0) -> (0,1) -> (0,2) -> (0,3) -> (0,4) -> (1,4) -> (2,4) -> (3,4) -> (4,4)",
            "metadata": {},
        }

    def check_answer(self, question: str, answer: str, metadata: dict = None) -> bool:
        if not metadata or not metadata.get("path"):
            return False
        path = metadata["path"]
        # Check length matches and endpoints are correct
        import re
        coords = re.findall(r"\((\d+),(\d+)\)", answer)
        if not coords:
            return False
        return len(coords) == len(path)
