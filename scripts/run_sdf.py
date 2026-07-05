"""Demo: build a Signed Distance Function (SDF) on a random grid and show the
artificial-potential-field repulsive force it induces around the obstacles.

From a binary occupancy grid we compute:

  * the signed distance field ``phi`` -- the clearance to the nearest obstacle
    in free space (negative inside obstacles), and
  * the APF repulsive force ``F_rep = -grad U_rep`` -- arrows that fan out from
    every obstacle and strengthen as the clearance shrinks.

Three figures are shown: the raw grid, the clearance heatmap (the SDF reacting
to the obstacles), and the repulsive force field.

Run with:  python scripts/run_sdf.py
"""

import time

import numpy as np

from gridflow.algorithms import SignedDistanceField
from gridflow.utils import (
    plot_grid,
    plot_time_map,
    plot_vector_field_on_grid,
)


def main():
    n = 21

    sdf_solver = SignedDistanceField()
    grid = sdf_solver.generate_random_grid(n=n, obstacle_prob=0.2, seed=1)

    start_time = time.time()

    sdf = sdf_solver.compute_sdf_map(grid)
    force = sdf_solver.compute_repulsive_force_map(grid, sdf=sdf, influence=3.0, gain=1.0)

    elapsed = time.time() - start_time
    print(f"SDF + repulsive force computation time: {elapsed:.4f} seconds")

    # Scale arrows so the strongest repulsion is ~0.9 cells long while keeping
    # the relative magnitudes (near-obstacle arrows stay longer than far ones).
    magnitude = np.linalg.norm(force, axis=2)
    peak = magnitude.max()
    force_display = force / peak * 0.9 if peak > 0 else force

    clearance_map = sdf_solver.to_display_map(sdf, grid)

    plot_grid(grid)
    plot_time_map(clearance_map, grid)
    plot_vector_field_on_grid(force_display, grid, color="red")


if __name__ == "__main__":
    main()
