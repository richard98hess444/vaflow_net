import functools

import numpy as np


@functools.lru_cache(maxsize=16)
def _march_indices(shape, center, depth, samples_per_cell):
    """Precompute per-step flattened sample indices for the occlusion march.

    The ray geometry depends only on ``shape``/``center``/``depth`` (fixed across
    frames), so it is computed once and cached. Samples that fall out of bounds
    or past the robot map to a sentinel slot (``prod(shape)``) so a padded
    obstacle array reads them as free. Returns an ``(num_steps, prod(shape))``
    int32 index array shared by :func:`solidify_pixels` / :func:`solidify_voxels`.
    """
    ndim = len(shape)
    coords = np.meshgrid(
        *[np.arange(s, dtype=np.float64) for s in shape], indexing="ij"
    )
    deltas = [center[a] - coords[a] for a in range(ndim)]
    dist = np.maximum(np.sqrt(sum(d * d for d in deltas)), 1e-6)
    units = [d / dist for d in deltas]

    n = int(np.prod(shape))
    steps = (
        np.arange(1, depth * samples_per_cell + 1, dtype=np.float64) / samples_per_cell
    )
    idx = np.empty((len(steps), n), dtype=np.int32)
    for t, s in enumerate(steps):
        samp = [np.rint(coords[a] + s * units[a]).astype(np.int64) for a in range(ndim)]
        inb = s <= dist
        for a in range(ndim):
            inb = inb & (samp[a] >= 0) & (samp[a] < shape[a])
        clipped = [np.clip(samp[a], 0, shape[a] - 1) for a in range(ndim)]
        flat = np.ravel_multi_index(clipped, shape)
        idx[t] = np.where(inb, flat, n).ravel()
    return idx


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
    if center is None:
        center = tuple(s // 2 for s in grid.shape)

    filled = grid.copy()
    if depth <= 0:
        return filled

    # Geometry (shape/center/depth) is fixed across frames, so the ray-march
    # sample indices are cached; each call only gathers the obstacle mask.
    idx = _march_indices(grid.shape, tuple(center), depth, samples_per_cell)
    obstacle = np.append((grid == obstacle_val).ravel(), False)  # sentinel = free
    hit = obstacle[idx].any(axis=0).reshape(grid.shape)

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
    if center is None:
        center = tuple(s // 2 for s in grid.shape)

    filled = grid.copy()
    if depth <= 0:
        return filled

    # Geometry (shape/center/depth) is fixed across frames, so the ray-march
    # sample indices are cached; each call only gathers the obstacle mask.
    idx = _march_indices(grid.shape, tuple(center), depth, samples_per_cell)
    obstacle = np.append((grid == obstacle_val).ravel(), False)  # sentinel = free
    hit = obstacle[idx].any(axis=0).reshape(grid.shape)

    filled[hit] = obstacle_val
    return filled
