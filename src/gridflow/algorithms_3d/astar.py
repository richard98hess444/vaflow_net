"""Classic A* pathfinding on a 3D voxel grid.

The 3D counterpart of :mod:`gridflow.algorithms.astar`. Moves over the
26-neighbor stencil with an octile-style (3D) admissible heuristic. Inherits
:class:`BreadthFirstSearch3D` only to reuse ``generate_random_grid`` and
``_is_inside``.
"""

import heapq
import math

import numpy as np

from .bfs import BreadthFirstSearch3D, DIRS_3D


class AStar3D(BreadthFirstSearch3D):
    """Classic A* on a 26-connected voxel grid."""

    # 26-neighbor moves: (di, dj, dk) (reuses the shared BFS3D stencil).
    MOVES = DIRS_3D

    def __init__(self, heuristic_weight=1.0):
        super().__init__()
        self.heuristic_weight = heuristic_weight

    def heuristic(self, a, b):
        """3D octile distance between two voxels, scaled by heuristic_weight.

        Admissible for 26-connected grids: sort the per-axis deltas and charge
        the cheapest available diagonal (3-axis, then 2-axis, then straight)
        moves, with costs sqrt(3), sqrt(2), 1.
        """
        d = sorted((abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])))
        dmin, dmid, dmax = d
        # dmin triple-axis steps, (dmid-dmin) double-axis steps, rest straight.
        octile = (
            math.sqrt(3) * dmin
            + math.sqrt(2) * (dmid - dmin)
            + 1.0 * (dmax - dmid)
        )
        return self.heuristic_weight * octile

    def _can_step(self, grid, i, j, k, di, dj, dk, n):
        """Whether a (possibly diagonal) step avoids cutting through solids.

        Mirrors the 2D corner guard: a diagonal move is blocked only if every
        axis-aligned face voxel obtained by cancelling one moving axis is an
        obstacle (or off-grid), so the step never slips through solid material.
        """
        moved = (di != 0) + (dj != 0) + (dk != 0)
        if moved <= 1:
            return True
        blocked = 0
        checked = 0
        for axis, dd in enumerate((di, dj, dk)):
            if dd == 0:
                continue
            step = [di, dj, dk]
            step[axis] = 0
            si, sj, sk = i + step[0], j + step[1], k + step[2]
            checked += 1
            if not self._is_inside(si, sj, sk, n) or grid[si, sj, sk] == 1:
                blocked += 1
        return blocked != checked

    def compute_path(self, grid: np.ndarray, start, goal):
        """Find the shortest path from start to goal using A*.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            start: (i, j, k) starting voxel.
            goal: (i, j, k) target voxel.

        Returns:
            path: list of (i, j, k) from start to goal, or [] if unreachable.
        """
        n = len(grid)

        open_heap = [(self.heuristic(start, goal), start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            i, j, k = current
            for di, dj, dk in self.MOVES:
                ni, nj, nk = i + di, j + dj, k + dk
                neighbor = (ni, nj, nk)

                if not self._is_inside(ni, nj, nk, n):
                    continue
                if grid[ni, nj, nk] == 1:
                    continue
                if neighbor in closed:
                    continue
                if not self._can_step(grid, i, j, k, di, dj, dk, n):
                    continue

                moved = (di != 0) + (dj != 0) + (dk != 0)
                step_cost = math.sqrt(moved)
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
