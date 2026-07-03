"""Plotting utilities for visualizing grids and flow maps."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def print_grid(grid):
    """Print grid in text format."""
    for row in grid:
        print(" ".join(str(x) for x in row))


def print_flow(flow):
    """Print flow map in text format."""
    for row in flow:
        print(" ".join(row))


def setup_axes(ax: plt.Axes, n: int):
    """Apply aligned grid lines and centered tick labels."""
    half = n // 2
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks(range(n))
    ax.set_xticklabels(range(-half, half + 1))
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(-half, half + 1))


def overlay_obstacles_and_source(ax: plt.Axes, grid: list[list[int]], source=None):
    """Draw obstacles as black, the center cell as red, and the source as green."""
    n = len(grid)
    cr, cc = n // 2, n // 2
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, color="black", zorder=2
                    )
                )
    ax.add_patch(mpatches.Rectangle((cc - 0.5, cr - 0.5), 1, 1, color="red", zorder=3))
    if source is not None:
        sr, sc = source
        ax.add_patch(
            mpatches.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="green", zorder=3)
        )


def draw_vector_field(ax, vectors, positions=None, color="blue"):
    """Draw a flow field as a quiver plot on `ax` (no obstacle overlay).

    This is the shared quiver-drawing primitive used by the training
    visualizer and the simulation renderer, so the arrow-drawing logic lives
    in one place.

    Args:
        ax: matplotlib Axes to draw on.
        vectors: (R, R, 2) array of (drow, dcol) vectors.
        positions: optional (R,) array of coordinates (in cell units) along
            each axis. Defaults to integer cells range(R).
        color: arrow color.
    """
    r = vectors.shape[0]
    if positions is None:
        positions = np.arange(r)
    rows, cols = np.meshgrid(positions, positions, indexing="ij")
    # vectors[..., 0] = drow (y, downward), vectors[..., 1] = dcol (x, rightward)
    u = vectors[..., 1]  # x-component (rightward)
    v = vectors[..., 0]  # y-component (row increases downward)
    ax.quiver(
        cols,
        rows,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        zorder=4,
    )
    return ax


def plot_grid(grid: list[list[int]], source=None):
    """Plot a grid with obstacles and source."""
    n = len(grid)
    data = np.zeros((n, n))  # all white background

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)

    setup_axes(ax, n)
    overlay_obstacles_and_source(ax, grid, source)

    plt.show()


def plot_flow(flow: list[list[str]], source=None):
    """Plot flow arrows on a grid."""
    n = len(flow)
    data = np.zeros((n, n))
    arrows = np.empty((n, n), dtype=object)

    # Build a grid mask to reuse overlay_obstacles_and_source
    grid = [[1 if flow[r][c] == "#" else 0 for c in range(n)] for r in range(n)]

    for r in range(n):
        for c in range(n):
            cell = flow[r][c]
            if cell not in ("S", "#"):
                data[r][c] = 0.0  # light gray for cells with flow arrows
                arrows[r][c] = cell

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)

    setup_axes(ax, n)
    overlay_obstacles_and_source(ax, grid, source)

    font_size = max(6, int(72 / n))
    for r in range(n):
        for c in range(n):
            if arrows[r][c]:
                ax.text(
                    c,
                    r,
                    arrows[r][c],
                    ha="center",
                    va="center",
                    color="blue",
                    fontsize=font_size,
                    zorder=4,
                )

    plt.show()


def plot_scalar_map(
    data_map: list[list[float]],
    grid: list[list[int]],
    cmap: str,
    vmin: float,
    vmax: float,
    label_fn,
    source=None,
):
    """Shared helper for scalar heatmap plots with text annotations."""
    n = len(grid)
    data = np.zeros((n, n))
    cr, cc = n // 2, n // 2

    for r in range(n):
        for c in range(n):
            val = data_map[r][c]
            if grid[r][c] == 1 or (r == cr and c == cc) or val < 0:
                data[r][c] = 0.0
            else:
                data[r][c] = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    setup_axes(ax, n)
    overlay_obstacles_and_source(ax, grid, source)

    font_size = max(6, int(72 / n))
    for r in range(n):
        for c in range(n):
            val = data_map[r][c]
            if grid[r][c] == 1 or val < 0:
                continue
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            text_color = "white" if norm > 0.5 else "black"
            ax.text(
                c,
                r,
                label_fn(val),
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
                zorder=4,
            )
    plt.show()


def plot_strength_map(strength_map, grid, source=None):
    """Plot strength map with values as text."""
    plot_scalar_map(strength_map, grid, "GnBu", 0, 0.8, lambda v: f"{v:.2f}", source)


def plot_viscosity_map(viscosity_map, grid, source=None):
    """Plot viscosity map with values as text."""
    plot_scalar_map(viscosity_map, grid, "cool", 0, 1, lambda v: f"{v:.2f}", source)


def plot_filtered_map(filtered_strength_map, grid, source=None):
    """Plot filtered strength map (strength x viscosity)."""
    plot_scalar_map(
        filtered_strength_map, grid, "GnBu", 0, 0.5, lambda v: f"{v:.2f}", source
    )


def plot_arrival_time_map(arrival_time_map, grid, source=None):
    """
    Visualize the BFS arrival time (hop count from source) for each cell.
    Lighter = closer to source, darker = farther away.
    """
    n = len(grid)
    data = np.zeros((n, n))
    cr, cc = n // 2, n // 2

    reachable = [
        arrival_time_map[r][c]
        for r in range(n)
        for c in range(n)
        if arrival_time_map[r][c] >= 0 and not (r == cr and c == cc)
    ]
    max_time = max(reachable) if reachable else 1

    for r in range(n):
        for c in range(n):
            t = arrival_time_map[r][c]
            if grid[r][c] == 1 or (r == cr and c == cc):
                data[r][c] = 0.0
            elif t == -1:
                data[r][c] = 0.0
            else:
                data[r][c] = t / max_time

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap="GnBu", interpolation="nearest")
    setup_axes(ax, n)
    overlay_obstacles_and_source(ax, grid, source)

    font_size = max(6, int(72 / n))
    for r in range(n):
        for c in range(n):
            t = arrival_time_map[r][c]
            if grid[r][c] == 1:
                continue
            if r == cr and c == cc:
                ax.text(
                    c,
                    r,
                    "0",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=font_size,
                    fontweight="bold",
                    zorder=4,
                )
            elif t >= 0:
                brightness = t / max_time
                text_color = "white" if brightness > 0.5 else "black"
                ax.text(
                    c,
                    r,
                    str(t),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=font_size,
                    zorder=4,
                )
    plt.show()


def plot_thresholded_grid(thresholded_grid, source=None):
    """
    Visualize the thresholded grid.

    - White  = free
    - Black  = hard obstacle
    - Gray   = soft obstacle (below strength threshold)
    - Red    = robot (center)
    - Green  = source
    """
    n = len(thresholded_grid)
    cr, cc = n // 2, n // 2
    data = np.zeros((n, n))

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    setup_axes(ax, n)

    for r in range(n):
        for c in range(n):
            if thresholded_grid[r][c] == 1:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, color="black", zorder=2
                    )
                )
            elif thresholded_grid[r][c] == 2:
                ax.add_patch(
                    mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, color="gray", zorder=2)
                )

    ax.add_patch(mpatches.Rectangle((cc - 0.5, cr - 0.5), 1, 1, color="red", zorder=3))
    if source is not None:
        sr, sc = source
        ax.add_patch(
            mpatches.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="green", zorder=3)
        )
    plt.show()


def plot_flow_on_thresholded_grid(
    flow_map, thresholded_grid, filtered_strength_map=None, source=None
):
    """
    Plot flow arrows on a thresholded grid.

    - Black = hard obstacle
    - Gray  = soft obstacle
    - Red   = robot (center)
    - Green = source
    - Blue arrows = flow directions, sized by remapping filtered_strength_map from 0..1 to 0.15..1
    """
    n = len(flow_map)
    cr, cc = n // 2, n // 2
    data = np.zeros((n, n))
    arrows = np.empty((n, n), dtype=object)

    for r in range(n):
        for c in range(n):
            cell = flow_map[r][c]
            if cell not in ("S", "#"):
                arrows[r][c] = cell

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    setup_axes(ax, n)

    for r in range(n):
        for c in range(n):
            if thresholded_grid[r][c] == 1:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, color="black", zorder=2
                    )
                )
            elif thresholded_grid[r][c] == 2:
                ax.add_patch(
                    mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, color="gray", zorder=2)
                )

    ax.add_patch(mpatches.Rectangle((cc - 0.5, cr - 0.5), 1, 1, color="red", zorder=3))
    if source is not None:
        sr, sc = source
        ax.add_patch(
            mpatches.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="green", zorder=3)
        )

    base_font_size = max(6, int(72 / n))
    min_scale = 0.15

    for r in range(n):
        for c in range(n):
            if arrows[r][c]:
                if filtered_strength_map is not None:
                    fs = filtered_strength_map[r][c]
                    clamped_strength = min(1.0, max(0.0, fs)) if fs >= 0 else 0.0
                    scale = min_scale + (1.0 - min_scale) * clamped_strength
                else:
                    scale = 1.0
                ax.text(
                    c,
                    r,
                    arrows[r][c],
                    ha="center",
                    va="center",
                    color="blue",
                    fontsize=base_font_size * scale * 4,
                    zorder=4,
                )
    plt.show()


def plot_path_on_thresholded_grid(path, thresholded_grid, source=None):
    """
    Plot a path as dots on a thresholded grid.

    - Black = hard obstacle
    - Gray  = soft obstacle
    - Red   = robot (center)
    - Green = source
    - Orange dots/line = path from center to source
    """
    n = len(thresholded_grid)
    cr, cc = n // 2, n // 2
    data = np.zeros((n, n))

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    setup_axes(ax, n)

    for r in range(n):
        for c in range(n):
            if thresholded_grid[r][c] == 1:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, color="black", zorder=2
                    )
                )
            elif thresholded_grid[r][c] == 2:
                ax.add_patch(
                    mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, color="gray", zorder=2)
                )

    ax.add_patch(mpatches.Rectangle((cc - 0.5, cr - 0.5), 1, 1, color="red", zorder=3))
    if source is not None:
        sr, sc = source
        ax.add_patch(
            mpatches.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="green", zorder=3)
        )

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="orange", linewidth=2, zorder=4)
        ax.scatter(xs, ys, color="orange", s=60, zorder=5)

    plt.show()


def plot_travel_time_map(travel_time_map, grid, source=None):
    """Plot a continuous Eikonal travel-time (wavefront arrival) field.

    Unlike ``plot_arrival_time_map`` (integer BFS hop counts), this expects a
    real-valued field where obstacles / unreachable cells are marked ``-1``.
    Lighter = closer to the source, darker = farther away.
    """
    n = len(grid)
    reachable = [
        travel_time_map[r][c]
        for r in range(n)
        for c in range(n)
        if travel_time_map[r][c] >= 0
    ]
    vmax = max(reachable) if reachable else 1.0
    plot_scalar_map(
        travel_time_map, grid, "GnBu", 0.0, vmax, lambda v: f"{v:.1f}", source
    )


def plot_vector_field_on_grid(vectors, grid, source=None, color="blue"):
    """Plot a flow/vector field over a binary grid with obstacles and source.

    Reuses the shared ``draw_vector_field`` quiver primitive so the arrow logic
    stays in one place. Arrows are drawn at cell centers.

    Args:
        vectors: (n, n, 2) array of (drow, dcol) vectors.
        grid: 2D list where 0=free, 1=obstacle.
        source: optional (row, col) drawn in green.
        color: arrow color.
    """
    n = len(grid)
    data = np.zeros((n, n))

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    setup_axes(ax, n)
    overlay_obstacles_and_source(ax, grid, source)
    draw_vector_field(ax, np.asarray(vectors), color=color)
    plt.show()

