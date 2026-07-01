from collections import deque
from .bfs import BreadthFirstSearch

import numpy as np


class WaterFlow(BreadthFirstSearch):
    """Minecraft-style water flow with viscosity and strength filtering."""

    def __init__(self):
        super().__init__()
        self.viscosity_factor = 0.5  # low 0.0 - high 1.0

    def _can_move(self, grid, r, c, nr, nc):
        """
        Decide whether water can move from (r, c) to (nr, nc).

        Rules:
        - Cannot move outside the map
        - Cannot move into obstacle
        - If moving diagonally, cannot cut through blocked corners
        """
        n = len(grid)
        if not self._is_inside(nr, nc, n):
            return False
        if grid[nr][nc] == 1:
            return False

        dr = nr - r
        dc = nc - c

        # If this is diagonal movement, check corner cutting
        if dr != 0 and dc != 0:
            sr1, sc1 = r + dr, c
            sr2, sc2 = r, c + dc
            if grid[sr1][sc1] == 1 and grid[sr2][sc2] == 1:
                return False

        return True

    def compute_viscosity_map(self, grid: list, source=None):
        """
        Compute viscosity map based on nearby obstacles.

        For each cell, count the number of obstacle neighbors.
        viscosity = 0.5^(num_obstacle_neighbors)

        Args:
            grid: 2D list where 0=free, 1=obstacle
            source: tuple (row, col) indicating the source position
        Returns:
            viscosity_map: 2D numpy array with viscosity values (-1 for obstacles)
        """
        n = len(grid)
        viscosity_map = np.array([[1.0 for _ in range(n)] for _ in range(n)])

        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    viscosity_map[r][c] = -1
                else:
                    # Count obstacle neighbors, weighting orthogonal (up/down/
                    # left/right) fully and diagonal neighbors at half.
                    obstacle_count = 0.0
                    for dr, dc, _ in self.dirs:
                        nr, nc_check = r + dr, c + dc
                        if self._is_inside(nr, nc_check, n) and grid[nr][nc_check] == 1:
                            if dr != 0 and dc != 0:
                                obstacle_count += 0.5  # diagonal
                            else:
                                obstacle_count += 1.0  # orthogonal

                    # Apply viscosity factor per (weighted) obstacle neighbor
                    viscosity_map[r][c] = 1 - (self.viscosity_factor**obstacle_count)

        if source is not None:
            viscosity_map[source[0]][source[1]] = 0.0  # source has no viscosity
        return viscosity_map

    def compute_arrival_time_map(self, grid: list, source=None):
        n = len(grid)
        center = (n // 2, n // 2)
        if source is not None:
            center = source

        arrival_time_map = np.array([[-1.0 for _ in range(n)] for _ in range(n)])

        cr, cc = center
        arrival_time_map[cr][cc] = 0.0

        q = deque()
        q.append((cr, cc))

        while q:
            r, c = q.popleft()
            for dr, dc, _ in self.dirs:
                nr = r + dr
                nc = c + dc
                if not self._can_move(grid, r, c, nr, nc):
                    continue
                if arrival_time_map[nr][nc] != -1:
                    continue

                arrival_time_map[nr][nc] = arrival_time_map[r][c] + 1
                q.append((nr, nc))

        return arrival_time_map

    def compute_strength_map(self, arrival_time_map: np.ndarray, source=None):
        """
        Compute strength map as the inverse of arrival time.

        strength = 1 / arrival_time

        Args:
            arrival_time_map: 2D numpy array with arrival times (-1 for unreachable)

        Returns:
            strength_map: 2D numpy array with strength values (-1 for unreachable)
        """
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
            strength_map[source[0]][source[1]] = 2.0
        return strength_map

    def compute_flow_map(self, grid: list, source=None):
        """
        Compute outward flow arrows from source using movement constraints.

        This overrides BFS.compute_flow_map so WaterFlow respects `_can_move`,
        including the diagonal corner-cutting rule.
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
        q.append((cr, cc))

        while q:
            r, c = q.popleft()

            for dr, dc, arrow in self.dirs:
                nr, nc = r + dr, c + dc

                if not self._can_move(grid, r, c, nr, nc):
                    continue

                if visited[nr][nc]:
                    continue

                visited[nr][nc] = True
                flow_map[nr][nc] = arrow
                q.append((nr, nc))

        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    flow_map[r][c] = "#"

        return flow_map

    def compute_filtered_map(
        self,
        strength_map: np.ndarray,
        viscosity_map: np.ndarray,
        grid: list,
        source=None,
    ):
        """
        Compute filtered strength by multiplying strength and viscosity.

        filtered_strength = strength_map * viscosity_map

        Args:
            strength_map: Strength at each cell
            viscosity_map: Viscosity at each cell
            grid: Grid for reference (to identify obstacles)

        Returns:
            filtered_map: 2D numpy array with filtered values
        """
        n = len(grid)
        cr, cc = n // 2, n // 2
        if source is not None:
            cr, cc = source
        filtered = np.array([[-1.0 for _ in range(n)] for _ in range(n)])

        for r in range(n):
            for c in range(n):
                if grid[r][c] == 1:
                    filtered[r][c] = -1
                # elif (r, c) == (cr, cc):
                #     filtered[r][c] = 1.0
                elif strength_map[r][c] == -1:
                    filtered[r][c] = -1
                else:
                    filtered[r][c] = strength_map[r][c] * (1 - viscosity_map[r][c])

        return filtered

    def apply_strength_threshold(
        self, filtered_map: np.ndarray, grid: list, threshold=0.5, source=None
    ):
        """
        Apply threshold to filtered strength map to create soft obstacles.

        Cells with filtered_strength < threshold become soft obstacles.

        Args:
            filtered_map: Filtered strength values
            grid: Original grid
            threshold: Strength threshold for creating soft obstacles

        Returns:
            thresholded_grid: 0=free, 1=hard obstacle, 2=soft obstacle
        """
        n = len(grid)
        cr, cc = n // 2, n // 2
        if source is not None:
            cr, cc = source
        thresholded_grid = np.array([[grid[r][c] for c in range(n)] for r in range(n)])

        for r in range(n):
            for c in range(n):
                if (r, c) == (cr, cc):
                    continue
                if grid[r][c] == 0:
                    fs = filtered_map[r][c]
                    if fs >= 0 and fs < threshold:
                        thresholded_grid[r][c] = 2

        return thresholded_grid

    def reverse_flow_map(self, flow_map):
        """
        Reverse the direction of every arrow in a flow map (visualization only).

        Each directional arrow is flipped to point the opposite way. Non-arrow
        markers ("S", "#", "X") are left unchanged.

        Args:
            flow_map: 2D list of arrow/marker strings

        Returns:
            reversed_map: 2D list with the same shape and marker convention
        """
        opposite = {
            "↑": "↓",
            "↓": "↑",
            "←": "→",
            "→": "←",
            "↖": "↘",
            "↘": "↖",
            "↗": "↙",
            "↙": "↗",
        }

        return [[opposite.get(cell, cell) for cell in row] for row in flow_map]

    def convert_flow_map_to_vectors(self, flow_map, strength_map):
        n = len(flow_map)
        vectors = np.zeros((n, n, 2))
        direction_map = {
            "↑": np.array([-1, 0]),
            "↓": np.array([1, 0]),
            "←": np.array([0, -1]),
            "→": np.array([0, 1]),
            "↖": np.array([-1, -1]),
            "↘": np.array([1, 1]),
            "↗": np.array([-1, 1]),
            "↙": np.array([1, -1]),
        }
        for r in range(n):
            for c in range(n):
                if flow_map[r][c] in direction_map:
                    vectors[r][c] = direction_map[flow_map[r][c]] * strength_map[r][c]
        return vectors

    def find_path(self, strength_map, viscosity_map, source):
        n = len(strength_map)
        center = (n // 2, n // 2)
        path = []
        visited = set()
        current = center

        while current != source:
            path.append(current)
            visited.add(current)
            r, c = current

            candidates = []
            for dr, dc, _ in self.dirs:
                nr, nc = r + dr, c + dc
                if not self._is_inside(nr, nc, n):
                    continue
                if (nr, nc) in visited:
                    continue
                if strength_map[nr][nc] < 0:  # obstacle / unreachable
                    continue
                candidates.append((nr, nc))

            if not candidates:
                break  # dead end, can't improve

            # Prefer largest strength; break ties by smallest viscosity.
            best_neighbor = max(
                candidates,
                key=lambda x: (strength_map[x[0]][x[1]], -viscosity_map[x[0]][x[1]]),
            )

            current = best_neighbor

        if current == source:
            path.append(source)

        return path

    def find_path_guaranteed(self, arrival_time_map, viscosity_map, source, grid=None):
        """
        Find a path from center to source that is guaranteed to arrive.

        Walks strictly downhill on `arrival_time_map`: each step moves to a
        neighbor whose arrival time is exactly one less than the current cell.
        Since arrival time decreases monotonically to 0 at the source, this
        always terminates at the source (when one is reachable).

        Among valid downhill neighbors, ties are broken by smallest viscosity.

        Args:
            arrival_time_map: BFS hop counts from the source (-1 for unreachable)
            viscosity_map: Viscosity at each cell (used for tie-breaking)
            source: tuple (row, col) of the source / target
            grid: optional grid for movement constraints (corner-cutting). If
                provided, steps must satisfy `_can_move`.

        Returns:
            path: list of (row, col) from center to source. Empty if the center
                is unreachable from the source.
        """
        n = len(arrival_time_map)
        center = (n // 2, n // 2)

        cr, cc = center
        if arrival_time_map[cr][cc] < 0:
            return []  # center not reachable from source

        path = [center]
        current = center

        while current != source:
            r, c = current
            target_time = arrival_time_map[r][c] - 1

            candidates = []
            for dr, dc, _ in self.dirs:
                nr, nc = r + dr, c + dc
                if not self._is_inside(nr, nc, n):
                    continue
                if grid is not None and not self._can_move(grid, r, c, nr, nc):
                    continue
                if arrival_time_map[nr][nc] == target_time:
                    candidates.append((nr, nc))

            if not candidates:
                break  # should not happen if arrival map is consistent

            # Break ties by smallest viscosity.
            best_neighbor = min(
                candidates,
                key=lambda x: viscosity_map[x[0]][x[1]],
            )

            current = best_neighbor
            path.append(current)

        return path

    def find_path_flow(
        self, arrival_time_map, viscosity_map, source, grid=None, lateral_bias=0.0
    ):
        """
        Find a path from center to source that follows the flow.

        The path never flows backward: each step may only move to a neighbor
        whose arrival time is <= the current cell's arrival time (downhill or
        lateral), never uphill. Among those allowed neighbors, the one with the
        lowest viscosity is chosen, so the path may take a sideways step (same
        arrival time) to pass through a lower-viscosity area. Ties in viscosity
        are broken by preferring progress (smaller arrival time).

        A `visited` set guarantees termination, since arrival time is never
        allowed to increase.

        Args:
            arrival_time_map: BFS hop counts from the source (-1 for unreachable)
            viscosity_map: Viscosity at each cell (lower is easier to traverse)
            source: tuple (row, col) of the source / target
            grid: optional grid for movement constraints (corner-cutting). If
                provided, steps must satisfy `_can_move`.
            lateral_bias: penalty added to a lateral (same arrival time) move's
                viscosity when comparing against downhill moves. A lateral step
                is only taken if its viscosity is lower than the best downhill
                option by more than `lateral_bias`. 0.0 reproduces the original
                behavior (always take the lowest viscosity); larger values make
                the path prefer progress and wander less.

        Returns:
            path: list of (row, col) from center to source. Empty if the center
                is unreachable. May stop early if it gets boxed in by visited
                cells.
        """
        n = len(arrival_time_map)
        center = (n // 2, n // 2)

        cr, cc = center
        if arrival_time_map[cr][cc] < 0:
            return []  # center not reachable from source

        path = [center]
        visited = {center}
        current = center

        while current != source:
            r, c = current
            current_time = arrival_time_map[r][c]

            candidates = []
            for dr, dc, _ in self.dirs:
                nr, nc = r + dr, c + dc
                if not self._is_inside(nr, nc, n):
                    continue
                if (nr, nc) in visited:
                    continue
                if arrival_time_map[nr][nc] < 0:  # obstacle / unreachable
                    continue
                if arrival_time_map[nr][nc] > current_time:  # would flow backward
                    continue
                if grid is not None and not self._can_move(grid, r, c, nr, nc):
                    continue
                candidates.append((nr, nc))

            if not candidates:
                break  # boxed in by visited cells

            # Effective viscosity: lateral (same-time) moves get `lateral_bias`
            # added so they are only chosen when meaningfully less viscous than
            # the best downhill option. Ties broken by progressing (smaller time).
            def _score(x):
                is_lateral = arrival_time_map[x[0]][x[1]] == current_time
                eff_viscosity = viscosity_map[x[0]][x[1]] + (
                    lateral_bias if is_lateral else 0.0
                )
                return (eff_viscosity, arrival_time_map[x[0]][x[1]])

            best_neighbor = min(candidates, key=_score)

            current = best_neighbor
            visited.add(current)
            path.append(current)

        return path
