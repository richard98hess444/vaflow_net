"""Plotting utilities for visualizing grids and flow maps."""

from matplotlib.axes import Axes
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


def setup_axes(ax: Axes, n: int):
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


def _create_grid_axes(
    n: int, data=None, *, cmap: str = "gray_r", vmin: float = 0.0, vmax: float = 1.0
) -> Axes:
    """Create a square grid plot with aligned axes and an optional background."""
    if data is None:
        data = np.zeros((n, n))

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    setup_axes(ax, n)
    return ax


def _overlay_center_and_source(ax: Axes, n: int, source=None):
    """Draw the center cell in red and the optional source cell in green."""
    cr, cc = n // 2, n // 2

    ax.add_patch(mpatches.Rectangle((cc - 0.5, cr - 0.5), 1, 1, color="red", zorder=3))
    if source is not None:
        sr, sc = source
        ax.add_patch(
            mpatches.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="green", zorder=3)
        )


def _overlay_binary_obstacles(ax: Axes, grid: np.ndarray):
    """Draw black obstacle cells from a binary occupancy grid."""
    n = len(grid)
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1, color="black", zorder=2
                    )
                )


def overlay_obstacles_and_source(ax: Axes, grid: np.ndarray, source=None):
    """Draw obstacles as black, the center cell as red, and the source as green."""
    _overlay_binary_obstacles(ax, grid)
    _overlay_center_and_source(ax, len(grid), source)


def overlay_thresholded_obstacles(ax: Axes, thresholded_grid):
    """Draw hard obstacles in black and soft obstacles in gray."""
    n = len(thresholded_grid)
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


def plot_grid(grid: np.ndarray, source=None):
    """
    Visualize the thresholded grid.

    - White  = free
    - Black  = hard obstacle
    - Gray   = soft obstacle (below strength threshold)
    - Red    = robot (center)
    - Green  = source
    """
    n = len(grid)
    ax = _create_grid_axes(n)
    overlay_obstacles_and_source(ax, grid, source)
    overlay_thresholded_obstacles(ax, grid)
    plt.show()


def plot_directions(flow: list[list[str]], source=None):
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


def _plot_scalar_map(
    data_map: np.ndarray,
    grid: np.ndarray,
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

    ax = _create_grid_axes(n, data, cmap=cmap, vmin=0, vmax=1)
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
    _plot_scalar_map(strength_map, grid, "GnBu", 0, 0.8, lambda v: f"{v:.2f}", source)


def plot_viscosity_map(viscosity_map, grid, source=None):
    """Plot viscosity map with values as text."""
    _plot_scalar_map(viscosity_map, grid, "cool", 0, 1, lambda v: f"{v:.2f}", source)


def plot_filtered_map(filtered_strength_map, grid, source=None):
    """Plot filtered strength map (strength x viscosity)."""
    _plot_scalar_map(
        filtered_strength_map, grid, "GnBu", 0, 0.5, lambda v: f"{v:.2f}", source
    )


def plot_time_map(
    time_map, grid, source=None, label_fn=lambda v: f"{v:.1f}", center_label=None
):
    """Plot a scalar time field where negative entries mark blocked cells."""
    n = len(grid)
    cr, cc = n // 2, n // 2
    reachable = [
        time_map[r][c]
        for r in range(n)
        for c in range(n)
        if time_map[r][c] >= 0
        and not (center_label is not None and r == cr and c == cc)
    ]
    vmax = max(reachable) if reachable else 1.0

    data = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            value = time_map[r][c]
            if (
                grid[r][c] == 1
                or value < 0
                or (center_label is not None and r == cr and c == cc)
            ):
                data[r][c] = 0.0
            else:
                data[r][c] = value / vmax if vmax > 0 else 0.0

    ax = _create_grid_axes(n, data, cmap="GnBu")
    overlay_obstacles_and_source(ax, grid, source)

    font_size = max(6, int(72 / n))
    for r in range(n):
        for c in range(n):
            value = time_map[r][c]
            if grid[r][c] == 1 or value < 0:
                continue
            if center_label is not None and r == cr and c == cc:
                ax.text(
                    c,
                    r,
                    center_label,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=font_size,
                    fontweight="bold",
                    zorder=4,
                )
                continue

            brightness = value / vmax if vmax > 0 else 0.0
            text_color = "white" if brightness > 0.5 else "black"
            ax.text(
                c,
                r,
                label_fn(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
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
    ax = _create_grid_axes(n)
    overlay_thresholded_obstacles(ax, thresholded_grid)
    _overlay_center_and_source(ax, n, source)

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="orange", linewidth=2, zorder=4)
        ax.scatter(xs, ys, color="orange", s=60, zorder=5)

    plt.show()


def plot_vector_field(ax: Axes, vectors, positions=None, color="blue"):
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
        scale=1.5,
        color=color,
        zorder=4,
    )
    return ax


def plot_vector_field_on_grid(vectors, grid, source=None, color="blue"):
    """Plot a flow/vector field over a binary grid with obstacles and source.

    Reuses the shared ``plot_vector_field`` quiver primitive so the arrow logic
    stays in one place. Arrows are drawn at cell centers.

    Args:
        vectors: (n, n, 2) array of (drow, dcol) vectors.
        grid: 2D list where 0=free, 1=obstacle.
        source: optional (row, col) drawn in green.
        color: arrow color.
    """
    n = len(grid)
    ax = _create_grid_axes(n)
    overlay_obstacles_and_source(ax, grid, source)
    overlay_thresholded_obstacles(ax, grid)
    plot_vector_field(ax, np.asarray(vectors), color=color)
    plt.show()
