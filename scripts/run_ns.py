"""Demo: solve the incompressible Navier-Stokes equations on a 2D grid.

Unlike the Eikonal wavefront (which only knows the shortest-time direction),
this treats propagation as an actual fluid: the domain is driven like water
draining to a plughole -- every free cell is a weak uniform source and the
``goal`` cell is a sink -- and the incompressible Navier-Stokes equations are
marched to steady state with Jos Stam's Stable Fluids scheme. The result is a
velocity field that flows from the robot (center) toward the goal while
respecting viscosity, inertia and incompressibility (so it hugs walls and can
recirculate behind obstacles).

From the steady velocity field we derive:

  * the speed heatmap ``|u|``,
  * the unit flow-direction field (comparable to the Eikonal gradient), and
  * a streamline path from the robot (center) to the goal.

Run with:  python scripts/run_navier_stokes.py
"""

import time
import numpy as np

from gridflow.algorithms import NavierStokes
from gridflow.utils import (
    plot_grid,
    plot_time_map,
    plot_vector_field_on_grid,
    plot_path_on_vector_field,
)


def main():
    n = 61

    # The center (n//2, n//2) is the robot; `goal` is where the fluid drains.
    goal = (0, 17)

    ns = NavierStokes(
        viscosity=0.1,
        dt=0.2,
        source_rate=0.05,
        num_steps=300,
    )

    seed = np.random.randint(0, 2**32 - 1)
    print(seed)

    grid = ns.generate_sparse_grid(
        n=n,
        obstacle_prob=0.08,
        seed=2450068298,
    )

    start_time = time.time()

    velocity_field = ns.compute_velocity_field(grid, goal=goal)
    speed_display_map = ns.to_display_map(velocity_field, grid)
    flow_field = ns.compute_flow_field(velocity_field, grid)

    elapsed = time.time() - start_time
    print(f"Navier-Stokes (Stable Fluids) computation time: {elapsed:.4f} seconds")

    # plot_grid(grid, source=goal, draw_robot=False)
    plot_vector_field_on_grid(velocity_field, grid, source=goal, draw_robot=False)
    plot_time_map(speed_display_map, grid, source=goal)
    plot_vector_field_on_grid(flow_field, grid, source=goal, draw_robot=False)


if __name__ == "__main__":
    main()
