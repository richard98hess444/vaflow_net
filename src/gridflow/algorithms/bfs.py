"""Breadth-First Search (BFS) algorithm for grid flow computation."""

import random
from collections import deque

import numpy as np

# 8-neighbor directions: row change, col change, arrow
DIRS = [
    (-1,  0, "↑"),
    ( 1,  0, "↓"),
    ( 0, -1, "←"),
    ( 0,  1, "→"),
    (-1, -1, "↖"),
    (-1,  1, "↗"),
    ( 1, -1, "↙"),
    ( 1,  1, "↘"),
]


class BreadthFirstSearch:
    """BFS class for computing grid flow with various properties."""

    def __init__(self):
        # 8-neighbor directions: row change, col change, arrow
        self.dirs = DIRS
        self.grid_size = 7
        self.obstacle_prob = 0.2

    def _is_inside(self, r, c, n):
        """Check if position (r, c) is inside an nxn grid."""
        return 0 <= r < n and 0 <= c < n

    def generate_random_grid(self, n=None, obstacle_prob=None, seed=None):
        """
        Generate a random grid with obstacles.

        Args:
            n: Grid size (must be odd), defaults to self.grid_size
            obstacle_prob: Probability of obstacle, defaults to self.obstacle_prob
            seed: Random seed for reproducibility

        Returns:
            grid: (n, n) numpy array where 0=free, 1=obstacle (center is always
                free)
        """
        if n is None:
            n = self.grid_size
        if obstacle_prob is None:
            obstacle_prob = self.obstacle_prob
        if seed is not None:
            random.seed(seed)

        assert n > 0 and n % 2 == 1, "n must be a positive odd number"

        grid = np.zeros((n, n), dtype=np.int8)
        center = (n // 2, n // 2)

        for r in range(n):
            for c in range(n):
                if (r, c) == center:
                    continue

                if random.random() < obstacle_prob:
                    grid[r, c] = 1

        for r in range(n):
            for c in range(n):
                if (r, c) == center:
                    continue
                if grid[r, c] == 0:
                    continue

                # Ensure at least one free neighbor for each obstacle
                has_obstacle_neighbor = any(
                    self._is_inside(r + dr, c + dc, n) and grid[r + dr, c + dc] == 1
                    for dr, dc, _ in self.dirs
                )
                if not has_obstacle_neighbor:
                    grid[r, c] = 0

        return grid

    def generate_sparse_grid(
        self,
        n=None,
        obstacle_prob=None,
        min_side=6,
        max_side=10,
        seed=None,
    ):
        """Generate a grid whose obstacles are large rectangular chunks.

        Unlike ``generate_random_grid`` (which decides each cell independently
        and produces scattered single-cell speckle), this stamps random
        axis-aligned rectangles. Every rectangle has at least one side of
        ``min_side`` cells, so at 5 cm/cell each obstacle chunk is at least
        ``min_side * 5`` cm on a side (30 cm by default) and covers at least
        ``min_side`` cells. Overlapping rectangles merge into larger blobs, so
        the layout still looks random.

        Args:
            n: Grid size (must be odd), defaults to self.grid_size.
            obstacle_prob: Target *fraction of cells* covered by obstacles
                (area coverage), defaults to self.obstacle_prob.
            min_side: Minimum length (in cells) of a rectangle's longer side.
            max_side: Maximum length (in cells) of a rectangle's longer side.
            seed: Random seed for reproducibility.

        Returns:
            grid: (n, n) numpy array where 0=free, 1=obstacle (center is always
                free).
        """
        if n is None:
            n = self.grid_size
        if obstacle_prob is None:
            obstacle_prob = self.obstacle_prob

        assert n > 0 and n % 2 == 1, "n must be a positive odd number"
        assert 1 <= min_side <= max_side, "require 1 <= min_side <= max_side"

        # Local RNG so we don't disturb the global random state.
        rng = random.Random(seed)

        grid = np.zeros((n, n), dtype=np.int8)
        center = (n // 2, n // 2)

        target_cells = int(obstacle_prob * n * n)
        covered = 0
        # Cap attempts so we always terminate even if the grid is small/full.
        max_attempts = max(1, 4 * target_cells)
        attempts = 0

        while covered < target_cells and attempts < max_attempts:
            attempts += 1

            # One side is the "long" side (>= min_side); the other is thinner.
            long_side = rng.randint(min_side, max_side)
            short_side = rng.randint(1, min_side)
            if rng.random() < 0.5:
                h, w = long_side, short_side
            else:
                h, w = short_side, long_side

            # Clamp to the grid and choose a random top-left corner.
            h = min(h, n)
            w = min(w, n)
            r0 = rng.randint(0, n - h)
            c0 = rng.randint(0, n - w)

            for r in range(r0, r0 + h):
                for c in range(c0, c0 + w):
                    if (r, c) == center:
                        continue
                    if grid[r, c] == 0:
                        grid[r, c] = 1
                        covered += 1

        grid[center[0], center[1]] = 0  # keep the center free
        return grid

    def compute_flow_map(self, grid, source=None):
        """
        Compute outward flow from the center using BFS.

        Args:
            grid: (n, n) numpy array where 0=free space, 1=obstacle

        Returns:
            flow: 2D list with arrow directions showing flow out from center
                "#" for obstacle, "S" for source, "X" for unreachable free cells
        """
        n = len(grid)
        center = (n // 2, n // 2)
        if source is not None:
            center = source

        flow_map = [["X" for _ in range(n)] for _ in range(n)]
        visited = [[False for _ in range(n)] for _ in range(n)]

        cr, cc = center
        flow_map[cr][cc] = "S"
        visited[cr][cc] = True

        q = deque()
        q.append(center)

        while q:
            r, c = q.popleft()

            for dr, dc, arrow in DIRS:
                nr, nc = r + dr, c + dc

                if not self._is_inside(nr, nc, n):
                    continue

                if grid[nr, nc] == 1:  # grid == 1 is an obstacle
                    continue

                if visited[nr][nc]:
                    continue

                visited[nr][nc] = True

                # The water moved from (r, c) to (nr, nc),
                # so the arrow at (nr, nc) points in that movement direction.
                flow_map[nr][nc] = arrow

                q.append((nr, nc))

        # Mark obstacles
        for r in range(n):
            for c in range(n):
                if grid[r, c] == 1:
                    flow_map[r][c] = "#"

        return flow_map
