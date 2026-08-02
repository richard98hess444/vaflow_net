"""Demo: solve the 3D Eikonal equation (Fast Marching Method) on a voxel grid.

Run with:  python scripts/run_eikonal_3d.py
"""

import time

from gridflow.algorithms_3d import Eikonal3D
from gridflow.utils import (
    plot_grid_3d,
    plot_time_map_3d,
    plot_vector_field_3d,
    plot_path_3d,
)


def main():
    n = 11
    source = (0, n - 1, n - 1)

    eik = Eikonal3D()
    grid = eik.generate_sparse_grid(
        n=n,
        obstacle_prob=0.2,
        min_side=5,
        max_side=10,
        seed=67,
    )

    start_time = time.time()

    arrival_time_map = eik.compute_arrival_time_map(grid, source=source)
    display_map = eik.to_display_map(arrival_time_map, grid)
    gradient_map = eik.compute_gradient_map(arrival_time_map, grid)
    path = eik.find_path(
        arrival_time_map, source=source, start=(n // 2, n // 2, n // 2)
    )

    elapsed = time.time() - start_time
    print(f"Eikonal3D computation time: {elapsed:.4f} seconds")

    plot_grid_3d(grid, source=source)
    plot_time_map_3d(display_map, grid, source=source)
    plot_vector_field_3d(gradient_map, grid, source=source, title="-grad T")
    plot_path_3d(path, grid, source=source, title="path")


if __name__ == "__main__":
    main()
