import numpy as np

def solidify_obstacles(grid, center=None, obstacle_val=1):
    """Turn thin obstacle *surfaces* into *solid* obstacles via shadow casting.

    A LiDAR only sees the surface of an obstacle that faces the robot, so the
    region *behind* that surface stays "free" in the occupancy grid even though
    the robot can never reach it. This function sits the robot at ``center`` and,
    for every free cell, checks whether it has a clear line of sight back to the
    robot. If the straight ray from the robot to the cell is blocked by an
    obstacle, the cell is occluded and gets filled in as an obstacle.

    Parameters
    ----------
    grid : np.ndarray
        2D occupancy grid. ``0`` = free, ``obstacle_val`` = obstacle.
    center : tuple[int, int] | None
        (row, col) of the robot. Defaults to the array centre.
    obstacle_val : int
        The value used for obstacle cells. Free cells are assumed to be 0.

    Returns
    -------
    np.ndarray
        A new grid with the occluded cells filled in as solid obstacles.
    """
    grid = np.asarray(grid)
    n_rows, n_cols = grid.shape
    if center is None:
        center = (n_rows // 2, n_cols // 2)
    cr, cc = center

    filled = grid.copy()
    obstacle = grid == obstacle_val

    for r in range(n_rows):
        for c in range(n_cols):
            if obstacle[r, c]:
                continue  # already a surface obstacle
            if not _has_line_of_sight(obstacle, cr, cc, r, c):
                filled[r, c] = obstacle_val
    return filled


def _has_line_of_sight(obstacle, r0, c0, r1, c1):
    """True if no obstacle lies strictly between (r0, c0) and (r1, c1).

    Walks the ray from the robot to the target cell. Both endpoints are
    excluded: the robot cell is the origin, and the target itself is what we
    are testing (it may be the visible surface).
    """
    dr = r1 - r0
    dc = c1 - c0
    # Sample densely (2 samples per cell) so diagonal rays cannot slip
    # through the gap between two diagonally-adjacent obstacle cells.
    steps = 2 * max(abs(dr), abs(dc))
    if steps == 0:
        return True
    for i in range(1, steps):
        r = int(round(r0 + dr * i / steps))
        c = int(round(c0 + dc * i / steps))
        if obstacle[r, c]:
            return False
    return True