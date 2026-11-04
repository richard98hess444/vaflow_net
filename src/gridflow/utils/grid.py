import functools

import numpy as np
from scipy.ndimage import convolve


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


def fill_stagnant_flow(field, grid, iters=4, normalize=False):
    """Fill zero-flow free cells with the mean of their moving neighbors' vectors.

    Incompressible flow has genuine stagnation points -- most visibly the domain
    corners (on two walls in 2D, three in 3D), where no-penetration zeroes every
    velocity component -- leaving holes that are discontinuous with the field
    around them. For a navigation *target* those holes are undesirable. This
    inpaints them: each pass sets every zero free cell that has at least one
    moving neighbor to the average of its moving neighbors' vectors, repeated
    ``iters`` times to reach into 2-3 cell pockets.

    Obstacles and fully isolated pockets (no moving neighbor at all, e.g. an
    unreachable region) are left at zero. Works for 2D ``(n, n, 2)`` and 3D
    ``(n, n, n, 3)`` fields.

    Args:
        field: (..., D) flow/direction field (last axis is the vector).
        grid: (...) array where 0=free, non-zero=obstacle (matches ``field``'s
            spatial shape).
        iters: Number of propagation passes.
        normalize: If True, renormalize the filled cells to unit length (use for
            a unit-direction field); if False, keep the averaged magnitude (use
            for a magnitude-carrying field).

    Returns:
        A new field with the reachable stagnation holes filled.
    """
    field = np.array(field, dtype=float)
    ndim = field.shape[-1]
    free = grid == 0
    kernel = np.ones((3,) * ndim, dtype=float)
    kernel[(1,) * ndim] = 0.0  # 3^D neighborhood, center excluded

    for _ in range(iters):
        mag = np.linalg.norm(field, axis=-1)
        moving = free & (mag > 1e-9)
        stagnant = free & ~moving
        den = convolve(moving.astype(float), kernel, mode="constant", cval=0.0)
        fillable = stagnant & (den > 0)
        if not fillable.any():
            break

        moving_f = moving.astype(float)
        out = field.copy()
        for c in range(ndim):
            contrib = field[..., c] * moving_f  # raw neighbor vectors, 0 elsewhere
            avg_c = convolve(contrib, kernel, mode="constant", cval=0.0) / np.where(
                den > 0, den, 1.0
            )
            out[..., c] = np.where(fillable, avg_c, field[..., c])

        if normalize:
            nm = np.linalg.norm(out, axis=-1)
            renorm = fillable & (nm > 1e-9)
            for c in range(ndim):
                out[..., c] = np.where(
                    renorm, out[..., c] / np.where(nm > 1e-9, nm, 1.0), out[..., c]
                )
        field = out

    return field

