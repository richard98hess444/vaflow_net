"""Breadth-First Search (BFS) algorithm for grid flow computation."""

import random
from collections import deque

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
            grid: 2D list where 0=free, 1=obstacle (center is always free)
        """
        if n is None:
            n = self.grid_size
        if obstacle_prob is None:
            obstacle_prob = self.obstacle_prob
        if seed is not None:
            random.seed(seed)

        assert n > 0 and n % 2 == 1, "n must be a positive odd number"

        grid = [[0 for _ in range(n)] for _ in range(n)]
        center = (n // 2, n // 2)

        for r in range(n):
            for c in range(n):
                if (r, c) == center:
                    continue

                if random.random() < obstacle_prob:
                    grid[r][c] = 1

        return grid

    def compute_flow_map(self, grid, source=None):
        """
        Compute outward flow from the center using BFS.

        Args:
            grid: 2D list where 0=free space, 1=obstacle

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

                if grid[nr][nc] == 1: # grid == 1 is an obstacle
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
                if grid[r][c] == 1:
                    flow_map[r][c] = "#"

        return flow_map
