"""3D visualization helpers for volumetric grids and flow fields.

The 3D counterpart of :mod:`gridflow.utils.plotting`. Because a solid voxel
volume hides its interior, these helpers lean on transparency, scatter clouds
and optional axis-slicing so the source, obstacles, flow field and path stay
legible. Coordinates follow the algorithm convention ``(i, j, k)`` mapped to
matplotlib ``(x=i, y=j, z=k)``.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def _configure_3d_axes(ax, n: int):
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_xlabel("i (x)")
    ax.set_ylabel("j (y)")
    ax.set_zlabel("k (z)")
    ax.set_box_aspect((1, 1, 1))
    return ax


def _new_3d_axes(n: int):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes((0.05, 0.05, 0.8, 0.9), projection="3d")
    _configure_3d_axes(ax, n)
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


def plot_grid_3d(grid: np.ndarray, source=None, title="3D grid", fast_obstacles=False):
    n = len(grid)
    fig, ax = _new_3d_axes(n)
    if fast_obstacles:
        _draw_obstacles_fast(ax, grid, alpha=0.9)
    else:
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
    fast_obstacles=False,
):
    n = len(grid)
    fig, ax = _new_3d_axes(n)
    if fast_obstacles:
        _draw_obstacles_fast(ax, grid, alpha=0.9)
    else:
        _draw_obstacles(ax, grid, alpha=0.9)

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
    ax=None,
):
    n = len(grid)
    standalone = ax is None
    if standalone:
        _, ax = _new_3d_axes(n)

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
    if title:
        ax.set_title(title)
    if standalone:
        plt.show()
    return ax


def plot_layer_vector_field_3d(
    vectors: np.ndarray,
    grid: np.ndarray,
    z=None,
    source=None,
    stride=1,
    color="blue",
    bar_color="gold",
    bar_cmap=None,
    bar_alpha=0.5,
    bar_width=0.8,
    length=1.0,
    normalize=True,
    bar_scale=1.0,
    min_magnitude=0.0,
    title="Flow layer",
    ax=None,
):
    n = len(grid)
    if z is None:
        z = n // 2
    standalone = ax is None
    if standalone:
        _, ax = _new_3d_axes(n)

    zc = z + 0.5
    vecs = np.asarray(vectors, dtype=float)
    layer = vecs[:, :, z]  # (n, n, 3)
    layer_grid = grid[:, :, z]

    filled = np.zeros((n, n, n), dtype=bool)
    filled[:, :, z] = layer_grid == 1
    if filled.any():
        ax.voxels(filled, facecolors="black", edgecolor=(0, 0, 0, 0.15), alpha=1.0)

    keep = np.zeros((n, n), dtype=bool)
    keep[::stride, ::stride] = True
    keep &= layer_grid != 1
    mag = np.linalg.norm(layer, axis=-1)
    keep &= mag > max(min_magnitude, 0.0)

    ii, jj = np.nonzero(keep)
    if ii.size:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        vx = layer[ii, jj, 0]
        vy = layer[ii, jj, 1]
        vz = layer[ii, jj, 2]
        base_z = np.full(ii.shape, zc)

        planar = np.linalg.norm(np.column_stack([vx, vy]), axis=1)
        safe = planar > 0
        ux = np.zeros_like(vx)
        uy = np.zeros_like(vy)
        ux[safe] = vx[safe] / planar[safe]
        uy[safe] = vy[safe] / planar[safe]

        if normalize:
            shaft_len = np.full(ii.shape, length)
        else:
            shaft_len = planar * length

        x0 = ii + 0.5
        y0 = jj + 0.5
        x1 = x0 + ux * shaft_len
        y1 = y0 + uy * shaft_len

        # Arrowhead wings: rotate the reversed direction by +/- head angle.
        head_len = shaft_len * 0.4
        head_angle = np.radians(28.0)
        ca, sa = np.cos(head_angle), np.sin(head_angle)
        bx, by = -ux, -uy  # pointing back from the tip
        # +angle wing
        lx = (bx * ca - by * sa) * head_len
        ly = (bx * sa + by * ca) * head_len
        # -angle wing
        rx = (bx * ca + by * sa) * head_len
        ry = (-bx * sa + by * ca) * head_len

        segments = []
        for k in range(ii.size):
            if not safe[k]:
                continue
            zk = zc
            tip = (x1[k], y1[k], zk)
            segments.append([(x0[k], y0[k], zk), tip])
            segments.append([tip, (x1[k] + lx[k], y1[k] + ly[k], zk)])
            segments.append([tip, (x1[k] + rx[k], y1[k] + ry[k], zk)])

        if segments:
            ax.add_collection3d(
                Line3DCollection(segments, colors=color, linewidths=2.0)
            )

        dz = vz * bar_scale
        half = bar_width / 2.0

        if bar_cmap is not None:
            cmap = mpl.colormaps[bar_cmap]
            norm = mpl.colors.Normalize(vmin=vz.min(), vmax=vz.max())
            bar_colors = cmap(norm(vz))
            face_colors = np.repeat(bar_colors, 6, axis=0)
            ax.bar3d(
                ii + 0.5 - half,
                jj + 0.5 - half,
                base_z,
                bar_width,
                bar_width,
                dz,
                color=face_colors,
                alpha=bar_alpha,
                edgecolor="none",
                shade=True,
            )
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(vz)
            ax.figure.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="vz")
        else:
            ax.bar3d(
                ii + 0.5 - half,
                jj + 0.5 - half,
                base_z,
                bar_width,
                bar_width,
                dz,
                color=bar_color,
                alpha=bar_alpha,
                edgecolor="none",
                shade=True,
            )

    _draw_center_and_source(ax, n, source)
    if title:
        ax.set_title(title)
    if standalone:
        plt.show()
    return ax


def plot_layer_vector_field_2d(
    vectors: np.ndarray,
    grid: np.ndarray,
    z=None,
    source=None,
    stride=1,
    color="blue",
    cmap="GnBu",
    obstacle_color="black",
    length=1.0,
    normalize=True,
    min_magnitude=0.0,
    title="Flow layer",
    ax=None,
):
    n = len(grid)
    if z is None:
        z = n // 2
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    vecs = np.asarray(vectors, dtype=float)
    layer = vecs[:, :, z]  # (n, n, 3)
    layer_grid = grid[:, :, z]

    vz = layer[:, :, 2]
    free = layer_grid != 1

    colormap = mpl.colormaps[cmap]
    if free.any():
        vmin = float(vz[free].min())
        vmax = float(vz[free].max())
    else:
        vmin, vmax = 0.0, 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    rgba = colormap(norm(vz))  # (n, n, 4)
    obstacle_rgba = mpl.colors.to_rgba(obstacle_color)
    rgba[~free] = obstacle_rgba

    ax.imshow(
        np.transpose(rgba, (1, 0, 2)),
        origin="lower",
        extent=(0, n, 0, n),
        interpolation="nearest",
    )

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array(vz[free] if free.any() else vz)
    fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02, label="vz")

    keep = np.zeros((n, n), dtype=bool)
    keep[::stride, ::stride] = True
    keep &= free
    mag = np.linalg.norm(layer[:, :, :2], axis=-1)
    keep &= mag > max(min_magnitude, 0.0)

    ii, jj = np.nonzero(keep)
    if ii.size:
        vx = layer[ii, jj, 0]
        vy = layer[ii, jj, 1]
        ax.quiver(
            ii + 0.5,
            jj + 0.5,
            vx,
            vy,
            angles="xy",
            scale_units="xy",
            scale=1.0 / length,
            color=color,
            width=0.004,
            pivot="tail",
        )

    c = n // 2
    ax.scatter([c + 0.5], [c + 0.5], color="red", s=120, label="center")
    if source is not None:
        si, sj = source[0], source[1]
        ax.scatter([si + 0.5], [sj + 0.5], color="green", s=120, label="source")

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xlabel("i (x)")
    ax.set_ylabel("j (y)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if standalone:
        plt.show()
    return ax


def plot_prediction_vs_target_3d(
    pred_field: np.ndarray,
    target_field: np.ndarray,
    grid: np.ndarray,
    source=None,
    stride=1,
    fast_obstacles=False,
    suptitle=None,
):
    n = len(grid)
    fig = plt.figure(figsize=(14, 7))

    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    _configure_3d_axes(ax_gt, n)
    plot_vector_field_3d(
        target_field,
        grid,
        source=source,
        stride=stride,
        fast_obstacles=fast_obstacles,
        title="Ground truth",
        ax=ax_gt,
    )

    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
    _configure_3d_axes(ax_pred, n)
    plot_vector_field_3d(
        pred_field,
        grid,
        source=source,
        stride=stride,
        fast_obstacles=fast_obstacles,
        title="Prediction",
        ax=ax_pred,
    )

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()

    return fig


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
