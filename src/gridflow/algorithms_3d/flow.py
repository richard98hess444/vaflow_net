"""3D Minecraft-style water flow with viscosity and strength filtering.

The 3D counterpart of :mod:`gridflow.algorithms.flow`. All ``(row, col)``
logic becomes ``(i, j, k)``, the 8-neighbor stencil becomes 26-neighbor, and
the diagonal corner-cutting guard generalizes to voxels: a diagonal step is
blocked only if *every* axis-aligned face it would slide past is obstructed.
"""

from collections import deque

import numpy as np
from scipy.ndimage import convolve

from .bfs import BreadthFirstSearch3D


class WaterFlow3D(BreadthFirstSearch3D):
    """Volumetric water flow with viscosity and strength filtering."""

    def __init__(self):
        super().__init__()
        self.viscosity_factor = 0.5  # low 0.0 - high 1.0

    def _can_move(self, grid, i, j, k, ni, nj, nk):
        """Decide whether water can move from (i,j,k) to (ni,nj,nk).

        Blocks moves outside the grid, into obstacles, and diagonal moves that
        would squeeze between obstacles. For a diagonal step, we require that at
        least one of the axis-aligned voxels adjacent to both the source and the
        destination is free, preventing the flow from slipping through a solid
        edge or corner.
        """
        n = len(grid)
        if not self._is_inside(ni, nj, nk, n):
            return False
        if grid[ni, nj, nk] == 1:
            return False

        di, dj, dk = ni - i, nj - j, nk - k
        moved_axes = sum(1 for d in (di, dj, dk) if d != 0)
        if moved_axes <= 1:
            return True  # face-adjacent move: no corner cutting possible

        # For an edge/corner diagonal move, check the intermediate face voxels
        # obtained by cancelling one moving axis at a time. If ALL of them are
        # obstacles, the water would have to cut through solid material.
        blocked = 0
        checked = 0
        for axis, d in enumerate((di, dj, dk)):
            if d == 0:
                continue
            step = [di, dj, dk]
            step[axis] = 0  # cancel this axis
            si, sj, sk = i + step[0], j + step[1], k + step[2]
            checked += 1
            if not self._is_inside(si, sj, sk, n) or grid[si, sj, sk] == 1:
                blocked += 1
        if blocked == checked:
            return False
        return True

    def compute_viscosity_map(self, grid: np.ndarray, source=None):
        """Compute a viscosity field from the density of nearby obstacles.

        viscosity = 1 - viscosity_factor**(weighted obstacle-neighbor count),
        where face neighbors count 1.0, edge neighbors 0.5 and corner neighbors
        0.25 (weight = 0.5 ** (extra moving axes)).

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            source: optional (i, j, k) forced to zero viscosity.

        Returns:
            viscosity_map: (n, n, n) array (-1 for obstacles).
        """
        n = len(grid)
        viscosity_map = np.ones((n, n, n))

        # The weighted obstacle-neighbor count is a convolution of the binary
        # obstacle mask with a fixed 3x3x3 kernel whose weight for offset
        # (di, dj, dk) is 0.5 ** (moved_axes - 1): 1.0 for the 6 faces, 0.5 for
        # the 12 edges, 0.25 for the 8 corners, 0 at the center. mode="constant"
        # (cval=0) reproduces the original's "off-grid neighbors don't count".
        kernel = np.zeros((3, 3, 3))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    moved_axes = (di != 0) + (dj != 0) + (dk != 0)
                    kernel[di + 1, dj + 1, dk + 1] = 0.5 ** (moved_axes - 1)

        obstacle = (grid == 1).astype(float)
        obstacle_count = convolve(obstacle, kernel, mode="constant", cval=0.0)
        viscosity_map = 1.0 - (self.viscosity_factor**obstacle_count)
        viscosity_map[grid == 1] = -1

        if source is not None:
            viscosity_map[source] = 0.0
        return viscosity_map

    def compute_arrival_time_map(self, grid: np.ndarray, source=None):
        """BFS hop-count field from the source (-1 for unreachable/obstacles)."""
        n = len(grid)
        center = (n // 2, n // 2, n // 2)
        if source is not None:
            center = source

        arrival_time_map = np.full((n, n, n), -1.0)
        arrival_time_map[center] = 0.0

        q = deque([center])
        while q:
            i, j, k = q.popleft()
            for di, dj, dk in self.dirs:
                ni, nj, nk = i + di, j + dj, k + dk
                if not self._can_move(grid, i, j, k, ni, nj, nk):
                    continue
                if arrival_time_map[ni, nj, nk] != -1:
                    continue
                arrival_time_map[ni, nj, nk] = arrival_time_map[i, j, k] + 1
                q.append((ni, nj, nk))

        return arrival_time_map

    def compute_strength_map(self, arrival_time_map: np.ndarray, source=None):
        """Strength = 1 / sqrt(arrival_time) (0 for unreachable)."""
        sqrt_time = np.sqrt(
            arrival_time_map,
            where=arrival_time_map > 0,
            out=np.zeros_like(arrival_time_map),
        )
        strength_map = np.divide(
            1.0,
            sqrt_time,
            out=np.zeros_like(arrival_time_map),
            where=sqrt_time > 0,
        )
        if source is not None:
            strength_map[source] = 2.0
        return strength_map

    def compute_flow_map(self, grid: np.ndarray, source=None):
        """Outward flow direction vectors, respecting ``_can_move``.

        Overrides :meth:`BreadthFirstSearch3D.compute_flow_map` so the 3D
        corner-cutting rule is honored.
        """
        n = len(grid)
        center = (n // 2, n // 2, n // 2)
        if source is not None:
            center = source

        flow_map = np.empty((n, n, n), dtype=object)
        flow_map[...] = "X"
        visited = np.zeros((n, n, n), dtype=bool)

        ci, cj, ck = center
        flow_map[ci, cj, ck] = "S"
        visited[ci, cj, ck] = True

        q = deque([center])
        while q:
            i, j, k = q.popleft()
            for di, dj, dk in self.dirs:
                ni, nj, nk = i + di, j + dj, k + dk
                if not self._can_move(grid, i, j, k, ni, nj, nk):
                    continue
                if visited[ni, nj, nk]:
                    continue
                visited[ni, nj, nk] = True
                flow_map[ni, nj, nk] = (di, dj, dk)
                q.append((ni, nj, nk))

        flow_map[grid == 1] = "#"
        return flow_map

    def compute_filtered_map(
        self,
        strength_map: np.ndarray,
        viscosity_map: np.ndarray,
        grid: np.ndarray,
        source=None,
    ):
        """filtered = strength * (1 - viscosity); -1 on obstacles/unreachable."""
        filtered = strength_map * (1.0 - viscosity_map)
        blocked = (grid == 1) | (strength_map == -1)
        filtered[blocked] = -1.0
        return filtered

    def apply_strength_threshold(
        self, filtered_map: np.ndarray, grid: np.ndarray, threshold=0.5, source=None
    ):
        """Turn weak-flow free voxels into soft obstacles (value 2)."""
        n = len(grid)
        center = (n // 2, n // 2, n // 2)
        if source is not None:
            center = source
        thresholded_grid = grid.copy()

        # Free voxels whose flow strength falls in [0, threshold) become soft
        # obstacles (value 2); the center voxel is always preserved.
        weak = (grid == 0) & (filtered_map >= 0) & (filtered_map < threshold)
        weak[center] = False
        thresholded_grid[weak] = 2
        return thresholded_grid

    def reverse_flow_map(self, flow_map: np.ndarray):
        """Negate every direction vector (visualization only)."""
        n = len(flow_map)
        reversed_map = np.empty((n, n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    cell = flow_map[i, j, k]
                    if isinstance(cell, tuple):
                        reversed_map[i, j, k] = tuple(-d for d in cell)
                    else:
                        reversed_map[i, j, k] = cell
        return reversed_map

    def convert_flow_map_to_vectors(self, flow_map: np.ndarray, strength_map):
        """Convert a direction-tuple flow map into scaled (n,n,n,3) vectors."""
        n = len(flow_map)
        vectors = np.zeros((n, n, n, 3))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    cell = flow_map[i, j, k]
                    if isinstance(cell, tuple):
                        vectors[i, j, k] = np.array(cell) * strength_map[i, j, k]
        return vectors

    def find_path_flow(
        self, arrival_time_map, viscosity_map, source, grid=None, lateral_bias=0.0
    ):
        """Trace a non-backtracking path from center to source.

        Each step may move to a neighbor whose arrival time is <= the current
        cell's (downhill or lateral), never uphill. Among allowed neighbors the
        lowest (viscosity, arrival-time) wins; lateral moves get ``lateral_bias``
        added so they are only taken when meaningfully less viscous. A visited
        set guarantees termination.
        """
        n = len(arrival_time_map)
        center = (n // 2, n // 2, n // 2)

        if arrival_time_map[center] < 0:
            return []

        path = [center]
        visited = {center}
        current = center

        while current != source:
            i, j, k = current
            current_time = arrival_time_map[i, j, k]

            candidates = []
            for di, dj, dk in self.dirs:
                ni, nj, nk = i + di, j + dj, k + dk
                if not self._is_inside(ni, nj, nk, n):
                    continue
                if (ni, nj, nk) in visited:
                    continue
                if arrival_time_map[ni, nj, nk] < 0:
                    continue
                if arrival_time_map[ni, nj, nk] > current_time:
                    continue
                if grid is not None and not self._can_move(grid, i, j, k, ni, nj, nk):
                    continue
                candidates.append((ni, nj, nk))

            if not candidates:
                break

            def _score(x):
                is_lateral = arrival_time_map[x] == current_time
                eff_viscosity = viscosity_map[x] + (lateral_bias if is_lateral else 0.0)
                return (eff_viscosity, arrival_time_map[x])

            current = min(candidates, key=_score)
            visited.add(current)
            path.append(current)

        return path
