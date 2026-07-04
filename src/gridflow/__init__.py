"""SimFlow - Modular water flow simulation and visualization."""

from .algorithms import BreadthFirstSearch, WaterFlow, AStar, Eikonal
from .utils import (
    print_grid,
    print_flow,
    plot_grid,
    plot_directions,
    plot_strength_map,
    plot_viscosity_map,
    plot_filtered_map,
    plot_time_map,
    plot_path_on_thresholded_grid,
    plot_vector_field_on_grid,
    overlay_obstacles_and_source,
    plot_vector_field,
    setup_axes,
)

__all__ = [
    "BreadthFirstSearch",
    "WaterFlow",
    "AStar",
    "Eikonal",
    "print_grid",
    "print_flow",
    "plot_grid",
    "plot_directions",
    "plot_strength_map",
    "plot_viscosity_map",
    "plot_filtered_map",
    "plot_time_map",
    "plot_path_on_thresholded_grid",
    "plot_vector_field_on_grid",
    "overlay_obstacles_and_source",
    "plot_vector_field",
    "setup_axes",
]
