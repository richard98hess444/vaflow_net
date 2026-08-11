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
    if depth <= 0:
        return filled

    obstacle = grid == obstacle_val

    # Per-cell unit direction pointing toward the robot (center - cell).
    ii, jj = np.meshgrid(
        np.arange(n_rows, dtype=np.float64),
        np.arange(n_cols, dtype=np.float64),
        indexing="ij",
    )
    dr = cr - ii  # (n_rows, n_cols)
    dc = cc - jj
    dist = np.maximum(np.sqrt(dr * dr + dc * dc), 1e-6)
    ur, uc = dr / dist, dc / dist

    # March toward the robot; a cell is occluded if any sample lands on obstacle.
    hit = np.zeros((n_rows, n_cols), dtype=bool)
    steps = (
        np.arange(1, depth * samples_per_cell + 1, dtype=np.float64) / samples_per_cell
    )
    for s in steps:
        sr = np.rint(ii + s * ur).astype(np.int64)
        sc = np.rint(jj + s * uc).astype(np.int64)
        inb = (sr >= 0) & (sr < n_rows) & (sc >= 0) & (sc < n_cols)
        # Stop at the robot: never march past the center, otherwise cells near
        # the robot get occluded by obstacles on the *opposite* side.
        inb &= s <= dist
        sr = np.clip(sr, 0, n_rows - 1)
        sc = np.clip(sc, 0, n_cols - 1)
        hit |= obstacle[sr, sc] & inb

    filled[hit] = obstacle_val
    return filled


def solidify_voxels(grid, depth=6, samples_per_cell=2, center=None, obstacle_val=1):
    """Fill the occluded shell directly behind thin LiDAR surfaces (3D).

    The 3D counterpart of :func:`solidify_pixels`. For every voxel, march toward
    the robot (``center``) up to ``depth`` voxels along the true 3D line of
    sight; if an obstacle is hit the voxel is occluded and marked occupied. The
    march never passes the robot, so a surface on one side can't solidify voxels
    on the opposite side. Fills a gap-free shell of thickness ~``depth``.

    Parameters
    ----------
    grid : np.ndarray
        3D voxel grid ``(nx, ny, nz)``. ``0`` = free, ``obstacle_val`` = obstacle.
    depth : int
        Shell thickness in voxels. ``0`` returns the grid unchanged.
    samples_per_cell : int
        Sub-steps per voxel while marching, so diagonal rays can't slip between
        diagonally-adjacent obstacles (``2`` is plenty).
    center : tuple[int, int, int] | None
        (ix, iy, iz) of the robot voxel. Defaults to the array centre.
    obstacle_val : int
        Value used for obstacle voxels. Free voxels are assumed to be 0.

    Returns
    -------
    np.ndarray
        A new grid with the occluded shell filled in as solid obstacles.
    """
    grid = np.asarray(grid)
    nx, ny, nz = grid.shape
    if center is None:
        center = (nx // 2, ny // 2, nz // 2)
    cx, cy, cz = center

    filled = grid.copy()
    if depth <= 0:
        return filled

    obstacle = grid == obstacle_val

    # Per-voxel unit direction pointing toward the robot (center - voxel).
    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nz, dtype=np.float64),
        indexing="ij",
    )
    dx = cx - ii  # (nx, ny, nz)
    dy = cy - jj
    dz = cz - kk
    dist = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)
    ux, uy, uz = dx / dist, dy / dist, dz / dist

    # March toward the robot; a voxel is occluded if any sample lands on obstacle.
    hit = np.zeros((nx, ny, nz), dtype=bool)
    steps = (
        np.arange(1, depth * samples_per_cell + 1, dtype=np.float64) / samples_per_cell
    )
    for s in steps:
        sx = np.rint(ii + s * ux).astype(np.int64)
        sy = np.rint(jj + s * uy).astype(np.int64)
        sz = np.rint(kk + s * uz).astype(np.int64)
        inb = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny) & (sz >= 0) & (sz < nz)
        # Stop at the robot: never march past the center, otherwise voxels near
        # the robot get occluded by obstacles on the *opposite* side.
        inb &= s <= dist
        sx = np.clip(sx, 0, nx - 1)
        sy = np.clip(sy, 0, ny - 1)
        sz = np.clip(sz, 0, nz - 1)
        hit |= obstacle[sx, sy, sz] & inb

    filled[hit] = obstacle_val
    return filled
