"""Demo: 3D BFS / WaterFlow on a randomly generated voxel grid.

Run with:  python scripts/run_bfs_3d.py
"""

import time

from gridflow.algorithms_3d import WaterFlow3D
from gridflow.utils import plot_grid_3d, plot_vector_field_3d


def main():
    n = 11
    source = (0, n - 1, n // 2)

    water = WaterFlow3D()
    grid = water.generate_sparse_grid(
        n=n,
        obstacle_prob=0.2,
        min_side=5,
        max_side=10,
        seed=67,
    )

    start_time = time.time()

    arrival = water.compute_arrival_time_map(grid, source=source)
    strength = water.compute_strength_map(arrival, source=source)
    flow_map = water.compute_flow_map(grid, source=source)
    reversed_flow = water.reverse_flow_map(flow_map)
    vectors = water.convert_flow_map_to_vectors(reversed_flow, strength)

    elapsed = time.time() - start_time
    print(f"WaterFlow3D computation time: {elapsed:.4f} seconds")

    plot_grid_3d(grid, source=source)
    plot_vector_field_3d(vectors, grid, source=source, title="WaterFlow3D field")


if __name__ == "__main__":
    main()
