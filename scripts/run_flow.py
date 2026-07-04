import time
from gridflow.algorithms.flow import WaterFlow
from gridflow.utils import (
    plot_grid,
    plot_time_map,
    plot_strength_map,
    plot_viscosity_map,
    plot_filtered_map,
    plot_directions,
    plot_vector_field_on_grid,
)


def main():
    n = 21
    threshold = 0.05

    # source = None
    source = (0, 17)

    wf = WaterFlow()

    grid = wf.generate_random_grid(
        n=n,
        obstacle_prob=0.2,
        seed=1,
    )

    start_time = time.time()

    arrival_time_map = wf.compute_arrival_time_map(grid, source=source)
    strength_map = wf.compute_strength_map(arrival_time_map, source=source)
    viscosity_map = wf.compute_viscosity_map(grid, source=source)
    flow_map = wf.compute_flow_map(grid, source=source)
    filtered_map = wf.compute_filtered_map(
        strength_map, viscosity_map, grid, source=source
    )

    elapsed = time.time() - start_time
    print(f"Flow computation time: {elapsed:.4f} seconds")

    # Apply threshold: cells below threshold become soft obstacles (value=2)
    thresholded_grid = wf.apply_strength_threshold(
        filtered_map, grid, threshold=threshold, source=source
    )

    # Build binary grid for BFS (treat soft obstacles as hard obstacles)
    binary_thresholded_grid = [
        [1 if thresholded_grid[r][c] != 0 else 0 for c in range(n)] for r in range(n)
    ]
    if source is not None:
        binary_thresholded_grid[source[0]][source[1]] = 0  # ensure source is free
    else:
        binary_thresholded_grid[n // 2][n // 2] = 0  # ensure source is free
    flow_map_t = wf.compute_flow_map(binary_thresholded_grid, source=source)
    vectors = wf.convert_flow_map_to_vectors(flow_map_t, strength_map)

    plot_grid(grid, source=source)
    plot_time_map(arrival_time_map, grid, source=source)
    plot_strength_map(strength_map, grid, source=source)
    plot_viscosity_map(viscosity_map, grid, source=source)
    plot_filtered_map(filtered_map, grid, source=source)
    plot_directions(flow_map, source=source)

    plot_grid(thresholded_grid, source=source)
    plot_vector_field_on_grid(vectors, thresholded_grid, source=source)

    flow_map_r = wf.reverse_flow_map(flow_map_t)
    vectors_r = wf.convert_flow_map_to_vectors(flow_map_r, strength_map)
    plot_vector_field_on_grid(vectors_r, thresholded_grid, source=source)


if __name__ == "__main__":
    main()
