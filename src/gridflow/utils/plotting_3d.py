"""3D visualization helpers for volumetric grids and flow fields.

The 3D counterpart of :mod:`gridflow.utils.plotting`. Because a solid voxel
volume hides its interior, these helpers lean on transparency, scatter clouds
and optional axis-slicing so the source, obstacles, flow field and path stay
legible. Coordinates follow the algorithm convention ``(i, j, k)`` mapped to
matplotlib ``(x=i, y=j, z=k)``.
"""

import numpy as np
import matplotlib.pyplot as plt


def _new_3d_axes(n: int):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes((0.05, 0.05, 0.8, 0.9), projection="3d")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_xlabel("i (x)")
    ax.set_ylabel("j (y)")
    ax.set_zlabel("k (z)")
    ax.set_box_aspect((1, 1, 1))
    return fig, ax


def _draw_obstacles(ax, grid: np.ndarray, obstacle_val=1, color="black", alpha=0.25):
    filled = grid == obstacle_val
    if filled.any():
        ax.voxels(filled, facecolors=color, edgecolor=(0, 0, 0, 0.15), alpha=alpha)


def _draw_obstacles_fast(
    ax, grid: np.ndarray, obstacle_val=1, color="black", alpha=0.25
):
    filled = grid == obstacle_val
    if not filled.any():
        return

    padded = np.pad(filled, 1, mode="constant", constant_values=False)
    interior = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    surface = filled & ~interior

    idx = np.argwhere(surface)
    if len(idx):
        ax.scatter(
            idx[:, 0] + 0.5,
            idx[:, 1] + 0.5,
            idx[:, 2] + 0.5,
            c=color,
            marker="s",
            s=40,
            alpha=alpha,
            edgecolors="none",
        )


def _draw_soft_obstacles(ax, grid: np.ndarray):
    hard = grid == 1
    soft = grid == 2
    if hard.any():
        ax.voxels(hard, facecolors="black", edgecolor=(0, 0, 0, 0.15), alpha=0.3)
    if soft.any():
        ax.voxels(soft, facecolors="gray", edgecolor=(0, 0, 0, 0.1), alpha=0.15)


def _draw_center_and_source(ax, n: int, source=None):
    c = n // 2
    ax.scatter([c + 0.5], [c + 0.5], [c + 0.5], color="red", s=120, label="center")
    if source is not None:
        si, sj, sk = source
        ax.scatter(
            [si + 0.5],
            [sj + 0.5],
            [sk + 0.5],
            color="green",
            s=120,
            label="source",
        )


def plot_grid_3d(grid: np.ndarray, source=None, title="3D grid"):
    n = len(grid)
    fig, ax = _new_3d_axes(n)
    _draw_obstacles(ax, grid, alpha=0.5)
    _draw_center_and_source(ax, n, source)
    ax.set_title(title)
    ax.legend(loc="upper left")
    plt.show()


def plot_time_map_3d(
    time_map: np.ndarray,
    grid: np.ndarray,
    source=None,
    cmap="viridis",
    title="Arrival time",
):
    n = len(grid)
    fig, ax = _new_3d_axes(n)
    _draw_obstacles(ax, grid, alpha=0.5)

    finite = np.isfinite(time_map) & (np.asarray(time_map) >= 0) & (grid == 0)
    idx = np.argwhere(finite)
    if len(idx):
        vals = np.asarray(time_map)[finite]
        p = ax.scatter(
            idx[:, 0] + 0.5,
            idx[:, 1] + 0.5,
            idx[:, 2] + 0.5,
            c=vals,
            cmap=cmap,
            s=250,
            alpha=0.9,
        )
        # Dedicated colorbar axes so the main 3D axes keeps its fixed size.
        cax = fig.add_axes((0.88, 0.2, 0.03, 0.6))
        fig.colorbar(p, cax=cax, label="value")

    _draw_center_and_source(ax, n, source)
    ax.set_title(title)
    plt.show()


def plot_vector_field_3d(
    vectors: np.ndarray,
    grid: np.ndarray,
    source=None,
    color="blue",
    length=1.0,
    stride=1,
    linewidth=1.0,
    arrow_length_ratio=0.5,
    normalize=True,
    min_magnitude=0.0,
    fast_obstacles=False,
    title="Flow field",
    alpha=1.0,
):
    n = len(grid)
    fig, ax = _new_3d_axes(n)
    if fast_obstacles:
        _draw_obstacles_fast(ax, grid, alpha=0.5)
    else:
        _draw_obstacles(ax, grid, alpha=0.5)

    vecs = np.asarray(vectors, dtype=float)

    keep = np.zeros((n, n, n), dtype=bool)
    keep[::stride, ::stride, ::stride] = True
    keep &= grid != 1
    mag = np.linalg.norm(vecs, axis=-1)
    keep &= mag > max(min_magnitude, 0.0)

    ii, jj, kk = np.nonzero(keep)
    if ii.size:
        us, vs, ws = vecs[ii, jj, kk].T
        ax.quiver(
            ii + 0.5,
            jj + 0.5,
            kk + 0.5,
            us,
            vs,
            ws,
            length=length,
            normalize=normalize,
            color=color,
            linewidth=linewidth,
            arrow_length_ratio=arrow_length_ratio,
            alpha=alpha,
        )

    _draw_center_and_source(ax, n, source)
    ax.set_title(title)
    plt.show()


def plot_path_3d(path, grid: np.ndarray, source=None, title="Path"):
    n = len(grid)
    fig, ax = _new_3d_axes(n)
    if np.any(grid == 2):
        _draw_soft_obstacles(ax, grid)
    else:
        _draw_obstacles(ax, grid, alpha=0.12)

    if path:
        arr = np.asarray(path, dtype=float) + 0.5
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color="orange", linewidth=2)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color="orange", s=40)

    _draw_center_and_source(ax, n, source)
    ax.set_title(title)
    plt.show()
