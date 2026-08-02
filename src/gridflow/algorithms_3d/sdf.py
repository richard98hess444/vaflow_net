"""3D Signed Distance Function (SDF) and the repulsive force it induces.

The 3D counterpart of :mod:`gridflow.algorithms.sdf`. The signed distance
function ``phi`` assigns to every voxel its distance to the nearest obstacle
boundary::

    phi(x) = +dist(x, obstacles)   in free space   (x outside every obstacle)
    phi(x) = -dist(x, free)        inside obstacles (x within an obstacle)

so ``phi`` is (approximately) zero on the boundary, positive in free space and
negative inside obstacles. Its magnitude is the local clearance and its gradient
``grad phi`` points, in free space, straight away from the nearest obstacle.

From ``phi`` we build a Khatib-style artificial potential field (APF) repulsive
force: every voxel closer than an influence radius ``d0`` is pushed away from
the obstacles, with a force that blows up as the clearance vanishes::

    F_rep = gain * (1/d - 1/d0) * (1/d**p) * grad_hat(phi)   (0 < d <= d0)
    F_rep = 0                                                (d > d0)

where ``d = phi`` is the clearance in free space. The result is a field of
``(di, dj, dk)`` vectors fanning out from every obstacle.

Inherits :class:`BreadthFirstSearch3D` only to reuse ``generate_random_grid``
and ``_is_inside`` (mirroring how :class:`Eikonal3D` reuses the grid generator).
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from .bfs import BreadthFirstSearch3D


class SignedDistanceField3D(BreadthFirstSearch3D):
    """Compute a voxel signed distance field and its APF repulsive force."""

    def _nearest_distance(self, targets: np.ndarray, n: int) -> np.ndarray:
        """Euclidean distance from every voxel to the nearest target voxel.

        Args:
            targets: (M, 3) array of (i, j, k) target coordinates.
            n: Grid size.

        Returns:
            (n, n, n) array of distances (``inf`` everywhere if there are no
            targets).
        """
        dist = np.full((n, n, n), np.inf)
        if len(targets) == 0:
            return dist

        ii = np.arange(n).reshape(n, 1, 1)
        jj = np.arange(n).reshape(1, n, 1)
        kk = np.arange(n).reshape(1, 1, n)
        # Accumulate the running minimum one target at a time so memory stays
        # O(n^3) instead of O(n^3 * M).
        for ti, tj, tk in targets:
            d = np.sqrt((ii - ti) ** 2 + (jj - tj) ** 2 + (kk - tk) ** 2)
            np.minimum(dist, d, out=dist)
        return dist

    def compute_sdf_map(self, grid: np.ndarray) -> np.ndarray:
        """Compute the signed distance field of a binary voxel grid.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.

        Returns:
            sdf: (n, n, n) array. Positive = clearance to the nearest obstacle
                in free space; negative = depth inside an obstacle. Free voxels
                hold ``inf`` only when the grid contains no obstacles at all.
        """
        is_free = grid == 0
        has_obstacle = not bool(is_free.all())
        has_free = bool(is_free.any())

        if has_obstacle:
            dist_to_obstacle = distance_transform_edt(is_free)
        else:
            dist_to_obstacle = np.full(grid.shape, np.inf)

        if has_free:
            dist_to_free = distance_transform_edt(~is_free)
        else:
            dist_to_free = np.full(grid.shape, np.inf)

        return np.where(is_free, dist_to_obstacle, -dist_to_free)

    def compute_repulsive_force_map(
        self,
        grid: np.ndarray,
        sdf: np.ndarray = None,
        influence: float = 3.0,
        gain: float = 1.0,
        cell_size: float = 1.0,
        falloff_power: float = 2.0,
    ) -> np.ndarray:
        """Compute the APF repulsive force induced by the obstacles.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            sdf: Optional precomputed signed distance field (see
                :meth:`compute_sdf_map`); recomputed if omitted. It is always in
                *voxel* units; ``cell_size`` converts it to physical distance.
            influence: Influence radius ``d0``. Voxels whose clearance exceeds
                this feel no force.
            gain: Repulsion strength ``eta``.
            cell_size: Physical distance represented by one voxel. The default
                1.0 keeps the "1 voxel = 1 distance" behavior.
            falloff_power: Exponent ``p`` on the clearance in the magnitude law
                ``gain * (1/d - 1/d0) / d**p``. ``2.0`` is the standard Khatib
                potential (magnitude ~1/d**3, steepest); lower values decay more
                gently.

        Returns:
            force: (n, n, n, 3) array of (di, dj, dk) force vectors. Zero on
                obstacles and outside the influence radius; elsewhere it points
                away from the nearest obstacle and grows as clearance shrinks.
        """
        n = len(grid)
        if sdf is None:
            sdf = self.compute_sdf_map(grid)

        # grad phi points toward increasing clearance, i.e. away from obstacles.
        # np.gradient returns (d/di, d/dj, d/dk), matching our (di, dj, dk) order.
        finite = np.where(np.isfinite(sdf), sdf, 0.0)
        grad_i, grad_j, grad_k = np.gradient(finite)
        grad_norm = np.sqrt(grad_i**2 + grad_j**2 + grad_k**2)

        d = sdf * cell_size
        valid = (
            (grid != 1)
            & np.isfinite(d)
            & (d > 0.0)
            & (d <= influence)
            & (grad_norm > 1e-9)
        )

        d_safe = np.where(valid, d, 1.0)
        g_safe = np.where(valid, grad_norm, 1.0)
        magnitude = gain * (1.0 / d_safe - 1.0 / influence) / (d_safe**falloff_power)
        scale = np.where(valid, magnitude / g_safe, 0.0)

        force = np.zeros((n, n, n, 3))
        force[..., 0] = scale * grad_i
        force[..., 1] = scale * grad_j
        force[..., 2] = scale * grad_k
        return force

    def to_display_map(self, sdf: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Convert an SDF to a plottable clearance map.

        Obstacle voxels (negative interior distances) and any non-finite entries
        are set to ``-1`` so the shared ``plot_time_map_3d`` helper renders them
        as blocked, leaving the positive free-space clearance to form the map.
        """
        out = np.array(sdf, dtype=float)
        out[(grid == 1) | ~np.isfinite(out)] = -1.0
        return out
