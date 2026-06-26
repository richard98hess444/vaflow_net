"""SimFlow - Modular water flow simulation and visualization."""

from .algorithms import BreadthFirstSearch, WaterFlow, AStar
from .utils import (
    print_grid,
    print_flow,
    plot_grid,
    plot_flow,
    plot_strength_map,
    plot_viscosity_map,
    plot_filtered_map,
    plot_arrival_time_map,
    plot_thresholded_grid,
    plot_flow_on_thresholded_grid,
    plot_path_on_thresholded_grid,
    overlay_obstacles_and_source,
    setup_axes,
)

__all__ = [
    "BreadthFirstSearch",
    "WaterFlow",
    "AStar",
    "print_grid",
    "print_flow",
    "plot_grid",
    "plot_flow",
    "plot_strength_map",
    "plot_viscosity_map",
    "plot_filtered_map",
    "plot_arrival_time_map",
    "plot_thresholded_grid",
    "plot_flow_on_thresholded_grid",
    "plot_path_on_thresholded_grid",
    "overlay_obstacles_and_source",
    "setup_axes",
]
