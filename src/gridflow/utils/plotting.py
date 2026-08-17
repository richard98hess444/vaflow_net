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


def setup_axes(ax: Axes, n: int, tick_step: int | None = None):
    """Apply aligned grid lines and centered tick labels.

    Labels are centered on the middle cell (label 0) and run from ``-n//2`` to
    ``+n//2``. To avoid clutter on large grids, only every ``tick_step``-th
    label is shown. When ``tick_step`` is None it defaults to 1 for small grids
    and 10 for large ones (e.g. a size-61 grid shows -30, -20, ..., 20, 30).

    Args:
        ax: matplotlib Axes to configure.
        n: Grid size (number of cells per side).
        tick_step: Spacing between shown labels (1 = every cell). Auto-selected
            when None.
    """
    half = n // 2
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    if tick_step is None:
        tick_step = 1 if n <= 19 else 10

    positions = [i for i in range(n) if (i - half) % tick_step == 0]
    labels = [i - half for i in positions]

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)


def _create_grid_axes(
    n: int,
    data=None,
    *,
    cmap: str = "gray_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    title: str | None = None,
) -> Axes:
    """Create a square grid plot with aligned axes and an optional background."""
    if data is None:
        data = np.zeros((n, n))

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    setup_axes(ax, n)
    if title:
        ax.set_title(title)
    return ax


def _draw_robot_footprint(ax: Axes, cr: int, cc: int):
    robot_length = 5
    robot_width = 3

    for length in range(-robot_length // 2 + 1, robot_length // 2 + 1):
        for width in range(-robot_width // 2 + 1, robot_width // 2 + 1):
            r = cr + length
            c = cc + width
            ax.add_patch(
                mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, color="orange", zorder=3)
            )
    ax.add_patch(mpatches.Rectangle((cc - 0.5, cr - 2.5), 1, 1, color="red", zorder=3))


def _overlay_center_and_source(ax: Axes, n: int, source=None, draw_robot: bool = True):
    """Draw the center cell in red and the optional source cell in green."""
    cr, cc = n // 2, n // 2

    if draw_robot:
        _draw_robot_footprint(ax, cr, cc)
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


def overlay_obstacles_and_source(
    ax: Axes, grid: np.ndarray, source=None, draw_robot=True
):
    """Draw obstacles as black, the center cell as red, and the source as green."""
    _overlay_binary_obstacles(ax, grid)
    _overlay_center_and_source(ax, len(grid), source, draw_robot=draw_robot)


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


def plot_grid(grid: np.ndarray, source=None, draw_robot=True, title=None):
    """
    Visualize the thresholded grid.

    - White  = free
    - Black  = hard obstacle
    - Gray   = soft obstacle (below strength threshold)
    - Red    = robot (center)
    - Green  = source
    """
    n = len(grid)
    ax = _create_grid_axes(n, title=title)
    overlay_obstacles_and_source(ax, grid, source, draw_robot=draw_robot)
    overlay_thresholded_obstacles(ax, grid)
    plt.show()


def plot_directions(flow: list[list[str]], source=None, draw_robot=True, title=None):
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
    if title:
        ax.set_title(title)
    overlay_obstacles_and_source(ax, grid, source, draw_robot=draw_robot)

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
    add_text: bool = True,
    title=None,
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

    ax = _create_grid_axes(n, data, cmap=cmap, vmin=0, vmax=1, title=title)
    overlay_obstacles_and_source(ax, grid, source, draw_robot=False)

    font_size = max(6, int(72 / n))
    for r in range(n):
        for c in range(n):
            val = data_map[r][c]
            if grid[r][c] == 1 or val < 0:
                continue
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            text_color = "white" if norm > 0.5 else "black"
            (
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
                if add_text
                else None
            )
    plt.show()


def plot_strength_map(
    strength_map, grid, source=None, add_text: bool = True, title=None
):
    """Plot strength map with values as text."""
    _plot_scalar_map(
        strength_map,
        grid,
        "GnBu",
        0,
        0.8,
        lambda v: f"{v:.2f}",
        source,
        add_text=add_text,
        title=title,
    )


def plot_viscosity_map(
    viscosity_map, grid, source=None, add_text: bool = True, title=None
):
    """Plot viscosity map with values as text."""
    _plot_scalar_map(
        viscosity_map,
        grid,
        "cool",
        0,
        1,
        lambda v: f"{v:.2f}",
        source,
        add_text=add_text,
        title=title,
    )


def plot_filtered_map(
    filtered_strength_map, grid, source=None, add_text: bool = True, title=None
):
    """Plot filtered strength map (strength x viscosity)."""
    _plot_scalar_map(
        filtered_strength_map,
        grid,
        "GnBu",
        0,
        0.5,
        lambda v: f"{v:.2f}",
        source,
        add_text=add_text,
        title=title,
    )


def plot_time_map(time_map, grid, source=None, add_text: bool = True, title=None):
    """Plot a scalar time/arrival field over the grid.

    Negative entries mark blocked/unreachable cells (rendered as background).
    The color scale runs from 0 to the map's maximum value.
    """
    _plot_scalar_map(
        time_map,
        grid,
        "GnBu",
        0,
        np.max(time_map),
        lambda v: f"{v:.1f}",
        source,
        add_text=add_text,
        title=title,
    )


def plot_path_on_thresholded_grid(path, thresholded_grid, source=None, title=None):
    """
    Plot a path as dots on a thresholded grid.

    - Black = hard obstacle
    - Gray  = soft obstacle
    - Red   = robot (center)
    - Green = source
    - Orange dots/line = path from center to source
    """
    n = len(thresholded_grid)
    ax = _create_grid_axes(n, title=title)
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


def plot_vector_field_on_grid(
    vectors, grid, source=None, color="blue", title=None, draw_robot=True
):
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
    ax = _create_grid_axes(n, title=title)
    overlay_obstacles_and_source(ax, grid, source, draw_robot=draw_robot)
    overlay_thresholded_obstacles(ax, grid)
    plot_vector_field(ax, np.asarray(vectors), color=color)
    plt.show()


def plot_path_on_vector_field(
    path,
    vectors,
    grid,
    source=None,
    positions=None,
    color="blue",
    draw_robot=True,
    title=None,
):
    n = len(grid)
    ax = _create_grid_axes(n, title=title)
    overlay_obstacles_and_source(ax, grid, source, draw_robot=draw_robot)
    overlay_thresholded_obstacles(ax, grid)
    plot_vector_field(ax, np.asarray(vectors), positions=positions, color=color)

    if path is not None and len(path):
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="purple", linewidth=2, zorder=3)
        ax.scatter(xs, ys, color="purple", s=6, zorder=3)

    plt.show()


def plot_prediction_vs_target(
    pred_field, target_field, grid, goal=None, suptitle=None, pred_positions=None
):
    """Show predicted and ground-truth vector fields side by side.

    Args:
        pred_field: (R, R, 2) predicted vectors.
        target_field: (n, n, 2) ground-truth vectors (integer cells).
        grid: 2D obstacle grid.
        goal: optional (row, col) goal/source cell.
        suptitle: optional overall title.
        pred_positions: optional (R,) query coordinates for the prediction grid
            (used when the prediction is at a finer resolution than the cells).
    """

    def plot_multi_axes_vector_field(
        ax: Axes,
        vectors,
        grid,
        source=None,
        positions=None,
        color="blue",
        title=None,
        draw_robot=True,
    ):
        n = len(grid)
        setup_axes(ax, n)
        overlay_obstacles_and_source(ax, grid, source=source, draw_robot=draw_robot)
        overlay_thresholded_obstacles(ax, grid)
        plot_vector_field(ax, np.asarray(vectors), positions=positions, color=color)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)  # invert y so row 0 is on top
        ax.set_aspect("equal")
        if title:
            ax.set_title(title)
        return ax

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_multi_axes_vector_field(
        axes[0], target_field, grid, source=goal, title="Ground truth"
    )
    plot_multi_axes_vector_field(
        axes[1],
        pred_field,
        grid,
        source=goal,
        positions=pred_positions,
        title="Prediction",
    )

    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()

    return fig
