"""Demo: build a 3D Signed Distance Function (SDF) on a random voxel grid and
show the artificial-potential-field repulsive force it induces around obstacles.

From a binary occupancy grid we compute:

  * the signed distance field ``phi`` -- the clearance to the nearest obstacle
    in free space (negative inside obstacles), and
  * the APF repulsive force ``F_rep = -grad U_rep`` -- arrows that fan out from
    every obstacle and strengthen as the clearance shrinks.

Run with:  python scripts/run_sdf_3d.py
"""

import time
import numpy as np

from gridflow.algorithms_3d import SignedDistanceField3D
from gridflow.utils import (
    plot_grid_3d,
    plot_time_map_3d,
    plot_vector_field_3d,
)


def main():
    n = 11

    sdf_solver = SignedDistanceField3D()
    grid = sdf_solver.generate_sparse_grid(
        n=n,
        obstacle_prob=0.2,
        min_side=5,
        max_side=10,
        seed=67,
    )

    start_time = time.time()

    sdf = sdf_solver.compute_sdf_map(grid)
    force = sdf_solver.compute_repulsive_force_map(
        grid, sdf=sdf, influence=3.0, gain=1.0
    )

    elapsed = time.time() - start_time
    print(f"SDF3D + repulsive force computation time: {elapsed:.4f} seconds")

    clearance_map = sdf_solver.to_display_map(sdf, grid)
    magnitude = np.linalg.norm(force, axis=3)
    peak = magnitude.max()
    force_display = force / peak * 0.9 if peak > 0 else force

    plot_grid_3d(grid, title="3D grid")
    plot_time_map_3d(clearance_map, grid, title="Clearance (SDF)")
    plot_vector_field_3d(
        force_display, grid, color="red", normalize=False, title="Repulsive force"
    )


if __name__ == "__main__":
    main()
