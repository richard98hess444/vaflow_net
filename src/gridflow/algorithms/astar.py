"""Classic A* pathfinding on a grid.

Inherits BreadthFirstSearch only to reuse `generate_random_grid`.
"""

import heapq
import math

from .bfs import BreadthFirstSearch


class AStar(BreadthFirstSearch):
    """Classic A* algorithm (8-neighbor, octile heuristic)."""

    # 8-neighbor moves: (row change, col change)
    MOVES = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    def __init__(self, heuristic_weight=1.0):
        super().__init__()
        self.heuristic_weight = heuristic_weight

    def heuristic(self, a, b):
        """Octile distance between two cells, scaled by heuristic_weight.

        Admissible for 8-connected grids where orthogonal steps cost 1 and
        diagonal steps cost sqrt(2).
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        octile = (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
        return self.heuristic_weight * octile

    def compute_path(self, grid, start, goal):
        """
        Find the shortest path from start to goal using A*.

        Args:
            grid: 2D list where 0=free, 1=obstacle
            start: tuple (row, col) starting cell
            goal: tuple (row, col) target cell

        Returns:
            path: list of (row, col) from start to goal, or [] if unreachable.
        """
        n = len(grid)


        open_heap = [(self.heuristic(start, goal), start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            r, c = current
            for dr, dc in self.MOVES:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if not self._is_inside(nr, nc, n):
                    continue
                if grid[nr][nc] == 1:  # obstacle
                    continue
                if neighbor in closed:
                    continue

                # Prevent cutting through blocked diagonal corners.
                if dr != 0 and dc != 0:
                    if grid[r + dr][c] == 1 and grid[r][c + dc] == 1:
                        continue

                step_cost = math.sqrt(2) if (dr != 0 and dc != 0) else 1
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, neighbor))

        return []  # no path found

    def _reconstruct_path(self, came_from, current):
        """Rebuild the path from goal back to start, then reverse it."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]