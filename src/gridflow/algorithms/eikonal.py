"""Eikonal equation solver via the Fast Marching Method (FMM).

The Eikonal equation is a first-order nonlinear PDE describing the first-arrival
time ``T`` of a wavefront expanding from a source::

    |grad T(x)| = 1 / f(x)          with   T(source) = 0

where ``f(x) > 0`` is the local propagation speed. Wherever the speed is
constant (``f = 1``) the solution ``T`` is simply the Euclidean distance to the
source; a varying speed field bends the wavefront around slow regions and
obstacles.

This is the continuous counterpart of the integer BFS ``arrival_time_map`` used
elsewhere in :mod:`gridflow`: BFS counts hops on an 8-neighbor lattice, while
the Eikonal solution measures true travel time and yields a smooth field whose
gradient gives a well-defined flow direction everywhere.

The equation is solved with the Fast Marching Method (FMM), a Dijkstra-like
single-pass scheme that visits nodes in order of increasing ``T`` and updates
each node with the Godunov upwind discretization of ``|grad T| = 1/f``.

Inherits :class:`BreadthFirstSearch` only to reuse ``generate_random_grid`` and
``_is_inside`` (mirroring how :class:`AStar` reuses the grid generator).
"""

import heapq
import math

import numpy as np

from .bfs import BreadthFirstSearch


class Eikonal(BreadthFirstSearch):
    """Solve the Eikonal equation on a 2D grid with the Fast Marching Method."""

    # Node states used by the marching front.
    FAR = 0  # not yet reached
    TRIAL = 1  # on the wavefront (tentative T, still in the heap)
    FROZEN = 2  # accepted (final T)

    # 4-neighbor stencil for the upwind Godunov update.
    NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, h=1.0):
        super().__init__()
        self.h = h  # grid spacing (distance between adjacent cell centers)

    def build_speed_map(self, grid: list[list[int]], free_speed=1.0) -> np.ndarray:
        """Build a speed field ``f`` from a binary grid.

        Free cells propagate at ``free_speed``; obstacles get speed 0, which
        corresponds to an infinite cost and blocks the wavefront.

        Args:
            grid: 2D list where 0=free, 1=obstacle.
            free_speed: Propagation speed assigned to free cells.

        Returns:
            speed: (n, n) numpy array of local speeds (0 on obstacles).
        """
        n = len(grid)
        speed = np.full((n, n), float(free_speed))
        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    speed[r][c] = 0.0
        return speed

    def _solve_quadratic(self, T, speed, r, c, n):
        """Godunov upwind update for a single cell.

        Solves ``(T - Tx)^2 + (T - Ty)^2 = (h / f)^2`` using only the smaller
        (upwind) neighbor in each axis, which is the standard first-order FMM
        local solver.

        Args:
            T: Current travel-time array (inf where not yet known).
            speed: Speed field ``f``.
            r, c: Cell to update.
            n: Grid size.

        Returns:
            The candidate travel time for cell (r, c) (``inf`` if unreachable).
        """
        f = speed[r][c]
        if f <= 0.0:
            return math.inf
        tau = self.h / f  # local cost of crossing one cell

        # Smallest arrival time among the two vertical neighbors...
        t_up = T[r - 1][c] if r > 0 else math.inf
        t_dn = T[r + 1][c] if r < n - 1 else math.inf
        t_row = min(t_up, t_dn)

        # ...and the two horizontal neighbors.
        t_lf = T[r][c - 1] if c > 0 else math.inf
        t_rt = T[r][c + 1] if c < n - 1 else math.inf
        t_col = min(t_lf, t_rt)

        # Order the upwind values so a <= b.
        a, b = (t_row, t_col) if t_row <= t_col else (t_col, t_row)
        if math.isinf(a):
            return math.inf  # no valid upwind neighbor -> cannot update

        # Try the 1D update first; it is valid when the wavefront only comes
        # from a single axis (i.e. the result stays below the other neighbor).
        candidate = a + tau
        if candidate <= b:
            return candidate

        # Otherwise both axes contribute: solve the quadratic. The discriminant
        # is non-negative here because candidate > b implies (b - a) < tau.
        disc = 2.0 * tau * tau - (a - b) ** 2
        return (a + b + math.sqrt(disc)) / 2.0

    def compute_travel_time_map(
        self,
        grid: list[list[int]],
        source: tuple[int, int] = None,
        speed_map: np.ndarray = None,
    ) -> np.ndarray:
        """Solve the Eikonal equation and return the travel-time field.

        Args:
            grid: 2D list where 0=free, 1=obstacle.
            source: (row, col) wavefront origin. Defaults to the grid center.
            speed_map: Optional (n, n) speed field. Defaults to speed 1 on free
                cells and 0 on obstacles (see :meth:`build_speed_map`).

        Returns:
            T: (n, n) numpy array of arrival times. Obstacles and unreachable free cells hold ``inf``.
        """
        n = len(grid)
        if source is None:
            source = (n // 2, n // 2)
        if speed_map is None:
            speed_map = self.build_speed_map(grid)

        T = np.full((n, n), math.inf)
        status = np.zeros((n, n), dtype=int)  # all FAR

        sr, sc = source
        T[sr][sc] = 0.0
        status[sr][sc] = self.TRIAL
        heap = [(0.0, sr, sc)]

        while heap:
            t, r, c = heapq.heappop(heap)

            # Skip stale heap entries (a cell may be pushed several times).
            if status[r][c] == self.FROZEN:
                continue
            status[r][c] = self.FROZEN

            for dr, dc in self.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not self._is_inside(nr, nc, n):
                    continue
                if speed_map[nr][nc] <= 0.0:  # obstacle: wavefront cannot enter
                    continue
                if status[nr][nc] == self.FROZEN:
                    continue

                new_t = self._solve_quadratic(T, speed_map, nr, nc, n)
                if new_t < T[nr][nc]:
                    T[nr][nc] = new_t
                    status[nr][nc] = self.TRIAL
                    heapq.heappush(heap, (new_t, nr, nc))

        return T

    def to_display_map(self, travel_time_map, grid):
        """Convert an ``inf``-padded travel-time field to a plottable map.

        Obstacles and unreachable cells become ``-1`` so they render as empty,
        matching the convention used by the ``plot_*`` helpers.
        """
        n = len(grid)
        out = np.array(travel_time_map, dtype=float)
        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1 or math.isinf(out[r][c]):
                    out[r][c] = -1.0
        return out

    def compute_gradient_field(self, travel_time_map, grid):
        """Compute the descent (flow) direction ``-grad T / |grad T|``.

        The wavefront normal is ``grad T``; moving *down* the field (``-grad T``)
        travels back toward the source along the shortest-time route. Central
        differences are used where possible, falling back to one-sided
        differences next to obstacles so no artificial gradients leak in.

        Args:
            travel_time_map: Travel-time field from
                :meth:`compute_travel_time_map`.
            grid: 2D list where 0=free, 1=obstacle.

        Returns:
            vectors: (n, n, 2) array of unit (drow, dcol) flow vectors. Zero on
                obstacles / unreachable cells.
        """
        n = len(grid)
        T = np.array(travel_time_map, dtype=float)

        def valid(r, c):
            return (
                self._is_inside(r, c, n) and grid[r][c] != 1 and not math.isinf(T[r][c])
            )

        vectors = np.zeros((n, n, 2))
        for r in range(n):
            for c in range(n):
                if not valid(r, c):
                    continue

                can_up, can_dn = valid(r - 1, c), valid(r + 1, c)
                if can_up and can_dn:
                    drow = (T[r + 1][c] - T[r - 1][c]) / 2.0
                elif can_dn:
                    drow = T[r + 1][c] - T[r][c]
                elif can_up:
                    drow = T[r][c] - T[r - 1][c]
                else:
                    drow = 0.0

                can_lf, can_rt = valid(r, c - 1), valid(r, c + 1)
                if can_lf and can_rt:
                    dcol = (T[r][c + 1] - T[r][c - 1]) / 2.0
                elif can_rt:
                    dcol = T[r][c + 1] - T[r][c]
                elif can_lf:
                    dcol = T[r][c] - T[r][c - 1]
                else:
                    dcol = 0.0

                norm = math.hypot(drow, dcol)
                if norm > 1e-9:
                    vectors[r][c][0] = -drow / norm  # descend -> toward source
                    vectors[r][c][1] = -dcol / norm
        return vectors

    def find_path(self, travel_time_map, source=None, start=None):
        """Trace a route by steepest descent on the travel-time field.

        Starting from ``start`` (the robot/center by default), repeatedly step
        to the 8-neighbor with the smallest travel time until the source is
        reached. Because ``T`` decreases monotonically to 0 at the source, this
        greedy descent follows the wavefront back to its origin.

        Args:
            travel_time_map: Travel-time field from
                :meth:`compute_travel_time_map`.
            source: (row, col) wavefront origin. Defaults to the grid center.
            start: (row, col) to start from. Defaults to the grid center.

        Returns:
            path: list of (row, col) from ``start`` to ``source`` (empty if the
                start cell is unreachable).
        """
        n = len(travel_time_map)
        center = (n // 2, n // 2)
        if source is None:
            source = center
        if start is None:
            start = center

        sr, sc = start
        if math.isinf(travel_time_map[sr][sc]):
            return []  # start not reachable from the source

        path = [start]
        visited = {start}
        current = start

        while current != source:
            r, c = current
            best = None
            best_t = travel_time_map[r][c]
            for dr, dc, _ in self.dirs:  # 8-neighbor descent
                nr, nc = r + dr, c + dc
                if not self._is_inside(nr, nc, n):
                    continue
                if (nr, nc) in visited:
                    continue
                t = travel_time_map[nr][nc]
                if math.isinf(t):
                    continue
                if t < best_t:
                    best_t = t
                    best = (nr, nc)

            if best is None:
                break  # local minimum (should only happen at the source)
            visited.add(best)
            path.append(best)
            current = best

        return path
