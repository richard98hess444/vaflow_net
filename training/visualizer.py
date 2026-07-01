"""Visualization utilities for the neural vector-field model.

Reuses the grid / obstacle / quiver drawing helpers from
``gridflow.utils.plotting`` and adds a side-by-side comparison of predicted vs
ground-truth flow vector fields.
"""

import matplotlib.pyplot as plt
import numpy as np

from gridflow.utils.plotting import (
    setup_axes,
    overlay_obstacles_and_source,
    draw_vector_field,
)


def plot_vector_field(vectors, grid, goal=None, ax=None, title=None, positions=None):
    """Draw a single vector field as a quiver plot over the grid.

    Args:
        vectors: (R, R, 2) array of (drow, dcol) vectors
        grid: 2D list/array where 1 = obstacle
        goal: optional (row, col) goal cell to highlight
        ax: optional matplotlib Axes; created if None
        title: optional title string
        positions: optional (R,) array of query coordinates (in cell units)
            along each axis. Defaults to integer cells range(len(grid)).
    """
    n = len(grid)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if positions is None:
        positions = np.arange(n)

    # Quiver arrows (shared primitive) + pixel-style grid and obstacle overlay.
    draw_vector_field(ax, vectors, positions=positions)
    setup_axes(ax, n)
    overlay_obstacles_and_source(ax, grid, source=goal)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # invert y so row 0 is on top
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    return ax


def plot_prediction_vs_target(
    pred_field, target_field, grid, goal=None, suptitle=None, pred_positions=None
):
    """Show predicted and ground-truth vector fields side by side.

    Args:
        pred_field: (R, R, 2) predicted vectors
        target_field: (n, n, 2) ground-truth vectors (integer cells)
        grid: 2D obstacle grid
        goal: optional (row, col)
        suptitle: optional overall title
        pred_positions: optional (R,) query coordinates for the prediction grid
            (used when the prediction is at a finer resolution than the cells).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_vector_field(target_field, grid, goal=goal, ax=axes[0], title="Ground truth")
    plot_vector_field(
        pred_field,
        grid,
        goal=goal,
        ax=axes[1],
        title="Prediction",
        positions=pred_positions,
    )

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    plt.show()
    return fig
