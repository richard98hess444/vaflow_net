"""3D Eikonal equation solver via the Fast Marching Method (FMM).

The 3D counterpart of :mod:`gridflow.algorithms.eikonal`. The Eikonal equation

    |grad T(x)| = 1 / f(x)          with   T(source) = 0

is solved on a voxel lattice with a Dijkstra-like single pass. The local update
generalizes the 2D Godunov upwind scheme to three axes: given the smallest
upwind arrival time along each of the (i, j, k) axes, we solve

    sum_axes (T - T_axis)^2 = (h / f)^2

including axes one at a time (cheapest first) until the solution is consistent
with the excluded neighbors. This is the standard first-order 3D FMM solver.

Inherits :class:`BreadthFirstSearch3D` only to reuse grid generation and
``_is_inside``.
"""

import heapq
import math

import numpy as np

from .bfs import BreadthFirstSearch3D


class Eikonal3D(BreadthFirstSearch3D):
    """Solve the Eikonal equation on a 3D grid with the Fast Marching Method."""

    FAR = 0
    TRIAL = 1
    FROZEN = 2

    # 6-neighbor (face) stencil for the upwind Godunov update.
    NEIGHBORS = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    def __init__(self, h=1.0):
        super().__init__()
        self.h = h  # voxel spacing

    def build_speed_map(self, grid: np.ndarray, free_speed=1.0) -> np.ndarray:
        """Speed field ``f``: ``free_speed`` on free voxels, 0 on obstacles."""
        speed = np.full(grid.shape, float(free_speed))
        speed[grid == 1] = 0.0
        return speed

    def _solve_quadratic(
        self, T: np.ndarray, speed: np.ndarray, i: int, j: int, k: int, n: int
    ) -> float:
        """Godunov upwind update for a single voxel (1, 2 or 3 axes)."""
        f = speed[i, j, k]
        if f <= 0.0:
            return math.inf
        tau = self.h / f

        # Smallest upwind arrival time along each axis.
        t_i = min(
            T[i - 1, j, k] if i > 0 else math.inf,
            T[i + 1, j, k] if i < n - 1 else math.inf,
        )
        t_j = min(
            T[i, j - 1, k] if j > 0 else math.inf,
            T[i, j + 1, k] if j < n - 1 else math.inf,
        )
        t_k = min(
            T[i, j, k - 1] if k > 0 else math.inf,
            T[i, j, k + 1] if k < n - 1 else math.inf,
        )

        axes = sorted(a for a in (t_i, t_j, t_k) if math.isfinite(a))
        if not axes:
            return math.inf

        # Incrementally include axes (cheapest first). Using the first m sorted
        # upwind values a_1..a_m, solve  m*T^2 - 2*S*T + (Q - tau^2) = 0  where
        # S = sum(a), Q = sum(a^2). Accept the root once it stays <= the next
        # (excluded) neighbor value.
        s = 0.0
        q = 0.0
        for m, a in enumerate(axes, start=1):
            s += a
            q += a * a
            disc = s * s - m * (q - tau * tau)
            if disc < 0:
                continue  # not consistent at this order, add another axis
            t = (s + math.sqrt(disc)) / m
            next_a = axes[m] if m < len(axes) else math.inf
            if t <= next_a:
                return t
        # Fallback (numerical corner cases): cheapest 1D update.
        return axes[0] + tau

    def compute_arrival_time_map(
        self,
        grid: np.ndarray,
        source: tuple[int, int, int] = None,
        speed_map: np.ndarray = None,
    ) -> np.ndarray:
        """Solve the Eikonal equation and return the arrival-time field.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            source: (i, j, k) wavefront origin. Defaults to the grid center.
            speed_map: optional (n, n, n) speed field.

        Returns:
            T: (n, n, n) arrival times. Obstacles / unreachable voxels hold inf.
        """
        n = len(grid)
        if source is None:
            source = (n // 2, n // 2, n // 2)
        if speed_map is None:
            speed_map = self.build_speed_map(grid)

        T = np.full((n, n, n), math.inf)
        status = np.zeros((n, n, n), dtype=int)

        si, sj, sk = source
        T[si, sj, sk] = 0.0
        status[si, sj, sk] = self.TRIAL
        heap = [(0.0, si, sj, sk)]

        while heap:
            t, i, j, k = heapq.heappop(heap)
            if status[i, j, k] == self.FROZEN:
                continue
            status[i, j, k] = self.FROZEN

            for di, dj, dk in self.NEIGHBORS:
                ni, nj, nk = i + di, j + dj, k + dk
                if not self._is_inside(ni, nj, nk, n):
                    continue
                if speed_map[ni, nj, nk] <= 0.0:
                    continue
                if status[ni, nj, nk] == self.FROZEN:
                    continue

                new_t = self._solve_quadratic(T, speed_map, ni, nj, nk, n)
                if new_t < T[ni, nj, nk]:
                    T[ni, nj, nk] = new_t
                    status[ni, nj, nk] = self.TRIAL
                    heapq.heappush(heap, (new_t, ni, nj, nk))

        return T

    def to_display_map(
        self, travel_time_map: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        """Convert an ``inf``-padded field to a plottable map (-1 = blocked)."""
        out = np.array(travel_time_map, dtype=float)
        out[(grid == 1) | ~np.isfinite(out)] = -1.0
        return out

    def compute_gradient_map(
        self, travel_time_map: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        """Descent (flow) direction ``-grad T / |grad T|`` per voxel.

        Central differences where possible, one-sided next to obstacles.
        Fully vectorized: per axis it builds the shifted upwind/downwind
        neighbor values and validity masks, then selects central / one-sided /
        zero differences exactly like the reference per-voxel scheme.

        Returns:
            vectors: (n, n, n, 3) unit (di, dj, dk) flow vectors. Zero on
                obstacles / unreachable voxels.
        """
        n = len(grid)
        T = np.array(travel_time_map, dtype=float)
        valid = (grid != 1) & np.isfinite(T)
        # Center values gated by validity (invalid -> 0, never used via masks).
        Tc = np.where(valid, T, 0.0)

        grad = np.zeros((n, n, n, 3))
        for axis in range(3):
            hi_val = np.zeros((n, n, n))
            hi_ok = np.zeros((n, n, n), dtype=bool)
            lo_val = np.zeros((n, n, n))
            lo_ok = np.zeros((n, n, n), dtype=bool)

            # Neighbor at +1 along axis: out[..:-1..] = value[..1:..].
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            dst[axis] = slice(0, n - 1)
            src[axis] = slice(1, n)
            hi_val[tuple(dst)] = Tc[tuple(src)]
            hi_ok[tuple(dst)] = valid[tuple(src)]

            # Neighbor at -1 along axis: out[..1:..] = value[..:-1..].
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            dst[axis] = slice(1, n)
            src[axis] = slice(0, n - 1)
            lo_val[tuple(dst)] = Tc[tuple(src)]
            lo_ok[tuple(dst)] = valid[tuple(src)]

            both = hi_ok & lo_ok
            grad[..., axis] = np.where(
                both,
                (hi_val - lo_val) / 2.0,
                np.where(
                    hi_ok,
                    hi_val - Tc,
                    np.where(lo_ok, Tc - lo_val, 0.0),
                ),
            )

        grad[~valid] = 0.0
        norm = np.linalg.norm(grad, axis=-1)
        vectors = np.zeros((n, n, n, 3))
        mask = norm > 1e-9
        vectors[mask] = -grad[mask] / norm[mask, None]
        return vectors

    def find_path(
        self,
        travel_time_map: np.ndarray,
        source: tuple[int, int, int] = None,
        start: tuple[int, int, int] = None,
    ) -> list[tuple[int, int, int]]:
        """Trace a route by steepest descent on the travel-time field."""
        n = len(travel_time_map)
        center = (n // 2, n // 2, n // 2)
        if source is None:
            source = center
        if start is None:
            start = center

        if math.isinf(travel_time_map[start]):
            return []

        path = [start]
        visited = {start}
        current = start

        while current != source:
            i, j, k = current
            best = None
            best_t = travel_time_map[i, j, k]
            for di, dj, dk in self.dirs:  # 26-neighbor descent
                ni, nj, nk = i + di, j + dj, k + dk
                if not self._is_inside(ni, nj, nk, n):
                    continue
                if (ni, nj, nk) in visited:
                    continue
                t = travel_time_map[ni, nj, nk]
                if math.isinf(t):
                    continue
                if t < best_t:
                    best_t = t
                    best = (ni, nj, nk)

            if best is None:
                break  # local minimum (should only happen at the source)
            visited.add(best)
            path.append(best)
            current = best

        return path
