import numpy as np


def solidify_pixels(grid, depth=4, samples_per_cell=2, center=None, obstacle_val=1):
    """Fill the occluded band directly behind thin LiDAR surfaces.

    For every cell, march toward the robot (``center``) up to ``depth`` cells; if
    an obstacle is hit the cell is occluded and marked occupied. This fills a
    gap-free band of thickness ~``depth`` behind each seen surface.

    Parameters
    ----------
    grid : np.ndarray
        2D occupancy grid. ``0`` = free, ``obstacle_val`` = obstacle.
    depth : int
        Band thickness in cells. ``0`` returns the grid unchanged.
    samples_per_cell : int
        Sub-steps per cell while marching, so diagonal rays can't slip between
        diagonally-adjacent obstacles (``2`` is plenty).
    center : tuple[int, int] | None
        (row, col) of the robot. Defaults to the array centre.
    obstacle_val : int
        Value used for obstacle cells. Free cells are assumed to be 0.

    Returns
    -------
    np.ndarray
        A new grid with the occluded band filled in as solid obstacles.
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