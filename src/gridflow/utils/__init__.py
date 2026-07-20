"""Utilities module for visualization and grid operations."""

from .plotting import (
    print_flow,
    print_grid,
    setup_axes,
    overlay_obstacles_and_source,
    plot_grid,
    plot_directions,
    plot_strength_map,
    plot_viscosity_map,
    plot_filtered_map,
    plot_time_map,
    plot_path_on_thresholded_grid,
    plot_vector_field,
    plot_vector_field_on_grid,
    plot_prediction_vs_target,
)

from .grid import (
    solidify_obstacles,
)

__all__ = [
    "print_grid",
    "print_flow",
    "setup_axes",
    "overlay_obstacles_and_source",
    "plot_grid",
    "plot_directions",
    "plot_strength_map",
    "plot_viscosity_map",
    "plot_filtered_map",
    "plot_time_map",
    "plot_path_on_thresholded_grid",
    "plot_vector_field",
    "plot_vector_field_on_grid",
    "plot_prediction_vs_target",
    "solidify_obstacles",
]
