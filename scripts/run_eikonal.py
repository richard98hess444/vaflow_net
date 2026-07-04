"""Demo: solve the Eikonal equation on a randomly generated 2D grid.

The Eikonal equation ``|grad T| = 1 / f`` is solved with the Fast Marching
Method to produce a wavefront travel-time field ``T`` emanating from the source.
From that single scalar field we derive:

  * the travel-time heatmap (a smooth, Euclidean-like arrival time),
  * the flow / gradient field ``-grad T`` (unit vectors pointing back to the
    source along the shortest-time route), and
  * a steepest-descent path from the robot (center) to the source.

Run with:  python scripts/run_eikonal.py
"""

import time

from gridflow.algorithms import Eikonal
from gridflow.utils import (
    plot_grid,
    plot_time_map,
    plot_vector_field_on_grid,
    plot_path_on_thresholded_grid,
)


def main():
    n = 21

    # The center (n//2, n//2) is the robot; `source` is the wavefront origin.
    source = (0, 17)

    eik = Eikonal()

    grid = eik.generate_random_grid(
        n=n,
        obstacle_prob=0.2,
        seed=1,
    )

    start_time = time.time()

    travel_time = eik.compute_travel_time_map(grid, source=source)
    display_map = eik.to_display_map(travel_time, grid)
    vectors = eik.compute_gradient_field(travel_time, grid)
    path = eik.find_path(travel_time, source=source)

    elapsed = time.time() - start_time
    print(f"Eikonal (FMM) computation time: {elapsed:.4f} seconds")
    print(f"Path length (cells): {len(path)}")

    plot_grid(grid, source=source)
    plot_time_map(display_map, grid, source=source)
    plot_vector_field_on_grid(vectors, grid, source=source)
    plot_path_on_thresholded_grid(path, grid, source=source)


if __name__ == "__main__":
    main()
