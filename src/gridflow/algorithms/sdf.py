"""Signed Distance Function (SDF) on a 2D grid and the repulsive force it induces.

The signed distance function ``phi`` assigns to every cell its distance to the
nearest obstacle boundary::

    phi(x) = +dist(x, obstacles)   in free space   (x outside every obstacle)
    phi(x) = -dist(x, free)        inside obstacles (x within an obstacle)

so ``phi`` is (approximately) zero on the boundary, positive in free space and
negative inside obstacles. Its magnitude is the local clearance and its gradient
``grad phi`` is a unit vector that, in free space, points straight away from the
nearest obstacle.

From ``phi`` we build an *artificial potential field* (APF) repulsive force in
the style of Khatib: every cell closer than an influence radius ``d0`` is pushed
away from the obstacles, with a force that blows up as the clearance vanishes::

    U_rep(d) = 0.5 * gain * (1/d - 1/d0)^2      for 0 < d <= d0
    U_rep(d) = 0                                 for d > d0

    F_rep    = -grad U_rep
             = gain * (1/d - 1/d0) * (1/d^2) * grad phi      (d <= d0)
             = 0                                              (d  > d0)

where ``d = phi`` is the clearance in free space. The result is a field of
(drow, dcol) vectors that visibly fan out from every obstacle, strongest where
the grid is tightest.

Inherits :class:`BreadthFirstSearch` only to reuse ``generate_random_grid`` and
``_is_inside`` (mirroring how :class:`Eikonal` reuses the grid generator).
"""

import numpy as np

from .bfs import BreadthFirstSearch


class SignedDistanceField(BreadthFirstSearch):
    """Compute a grid signed distance field and its APF repulsive force."""

    def _nearest_distance(self, targets: np.ndarray, n: int) -> np.ndarray:
        """Euclidean distance from every cell to the nearest target cell.

        Args:
            targets: (M, 2) array of (row, col) target coordinates.
            n: Grid size.

        Returns:
            (n, n) array of distances (``inf`` everywhere if there are no
            targets).
        """
        dist = np.full((n, n), np.inf)
        if len(targets) == 0:
            return dist

        rows = np.arange(n).reshape(n, 1)
        cols = np.arange(n).reshape(1, n)
        # Accumulate the running minimum one target at a time so the memory stays
        # O(n^2) instead of O(n^2 * M).
        for tr, tc in targets:
            d = np.sqrt((rows - tr) ** 2 + (cols - tc) ** 2)
            np.minimum(dist, d, out=dist)
        return dist

    def compute_sdf_map(self, grid: list[list[int]]) -> np.ndarray:
        """Compute the signed distance field of a binary grid.

        Args:
            grid: 2D list where 0=free, 1=obstacle.

        Returns:
            sdf: (n, n) array. Positive = clearance to the nearest obstacle in
                free space; negative = depth inside an obstacle. Free cells hold
                ``inf`` only when the grid contains no obstacles at all.
        """
        n = len(grid)
        grid_arr = np.asarray(grid)
        obstacle_coords = np.argwhere(grid_arr == 1)
        free_coords = np.argwhere(grid_arr == 0)

        dist_to_obstacle = self._nearest_distance(obstacle_coords, n)
        dist_to_free = self._nearest_distance(free_coords, n)

        # Free cells carry their (positive) distance to the nearest obstacle;
        # obstacle cells carry the negative distance to the nearest free cell.
        sdf = np.where(grid_arr == 0, dist_to_obstacle, -dist_to_free)
        return sdf

    def compute_repulsive_force_map(
        self,
        grid: list[list[int]],
        sdf: np.ndarray = None,
        influence: float = 3.0,
        gain: float = 1.0,
    ) -> np.ndarray:
        """Compute the APF repulsive force induced by the obstacles.

        Args:
            grid: 2D list where 0=free, 1=obstacle.
            sdf: Optional precomputed signed distance field (see
                :meth:`compute_sdf_map`); recomputed if omitted.
            influence: Influence radius ``d0`` (in cells). Cells whose clearance
                exceeds this feel no force.
            gain: Repulsion strength ``eta``.

        Returns:
            force: (n, n, 2) array of (drow, dcol) force vectors. Zero on
                obstacles and outside the influence radius; elsewhere it points
                away from the nearest obstacle and grows as clearance shrinks.
        """
        n = len(grid)
        grid_arr = np.asarray(grid)
        if sdf is None:
            sdf = self.compute_sdf_map(grid)

        # grad phi points toward increasing clearance, i.e. away from obstacles.
        # np.gradient returns (d/drow, d/dcol), matching our (drow, dcol) order.
        finite = np.where(np.isfinite(sdf), sdf, 0.0)
        grad_row, grad_col = np.gradient(finite)
        grad_norm = np.hypot(grad_row, grad_col)

        force = np.zeros((n, n, 2))
        for r in range(n):
            for c in range(n):
                if grid_arr[r][c] == 1:
                    continue  # force is undefined inside obstacles

                d = sdf[r][c]
                if not np.isfinite(d) or d <= 0.0 or d > influence:
                    continue  # unreachable, on a boundary, or out of range

                g = grad_norm[r][c]
                if g < 1e-9:
                    continue  # flat field: repulsion direction is undefined

                # APF magnitude; grows without bound as the clearance d -> 0.
                magnitude = gain * (1.0 / d - 1.0 / influence) / (d * d)
                force[r][c][0] = magnitude * grad_row[r][c] / g
                force[r][c][1] = magnitude * grad_col[r][c] / g
        return force

    def to_display_map(self, sdf: np.ndarray, grid: list[list[int]]) -> np.ndarray:
        """Convert an SDF to a plottable clearance map.

        Obstacle cells (negative interior distances) and any non-finite entries
        are set to ``-1`` so the shared ``plot_time_map`` helper renders them as
        blocked, leaving the positive free-space clearance to form the heatmap.
        """
        n = len(grid)
        out = np.array(sdf, dtype=float)
        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1 or not np.isfinite(out[r][c]):
                    out[r][c] = -1.0
        return out
