"""Ground-truth guidance vector field shared across the whole workspace.

This is the single source of truth for how a target (drow, dcol) guidance field
is built from a grid and a goal. It is reused by:

  * the training dataset (``gridflow_nn/training/dataloader/flow_field_dataset.py``),
  * the live simulator (``gridflow_sim/simulator.py``), and
  * the exploratory notebooks (``notebook/*.ipynb``),

so every part of the project trains against and evaluates the exact same target.

:class:`VectorFieldGuidance` bundles the solver instances and the
attraction/repulsion configuration together and exposes two interchangeable
strategies for producing that target:

  * :meth:`VectorFieldGuidance.eik_method` -- the Eikonal shortest-path gradient
    (attraction toward the goal) plus the SDF artificial-potential-field
    repulsion (push away from obstacles), and
  * :meth:`VectorFieldGuidance.bfs_method` -- the original WaterFlow pipeline.

Construct one instance and reuse it; the held solvers and the ``influence`` /
``gain`` repulsion parameters are then shared across every call.
"""

import numpy as np

from .flow import WaterFlow
from .eikonal import Eikonal
from .navier_stokes import NavierStokes
from .sdf import SignedDistanceField

# Default artificial-potential-field repulsion parameters. Kept here so every
# caller shares the same balance between attraction and repulsion.
DEFAULT_INFLUENCE = 3.0
DEFAULT_GAIN = 1.0
DEFAULT_RESOLUTION = 0.1
DEFAULT_FALLOFF_POWER = 0.67


class VectorFieldGuidance:
    """Build ground-truth guidance fields, holding the solvers and config.

    The instance owns one :class:`WaterFlow`, :class:`Eikonal` and
    :class:`SignedDistanceField` solver plus the repulsion parameters, so the
    same objects and settings are reused across every guidance call (e.g. every
    sample in a dataset loop) instead of being threaded through as arguments.

    Args:
        influence: SDF repulsion influence radius ``d0`` in cells.
        gain: SDF repulsion strength.
    """

    def __init__(
        self,
        influence: float = DEFAULT_INFLUENCE,
        gain: float = DEFAULT_GAIN,
        resolution: float = DEFAULT_RESOLUTION,
        falloff_power: float = DEFAULT_FALLOFF_POWER,
    ):
        self.wf = WaterFlow()
        self.eik = Eikonal()
        self.ns = NavierStokes(viscosity=0.9, dt=0.1, source_rate=0.05, num_steps=100)
        self.sdf = SignedDistanceField()
        self.influence = influence
        self.gain = gain
        self.resolution = resolution
        self.falloff_power = falloff_power

    def eik_method(self, grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
        """Build the guidance field from the Eikonal gradient plus SDF repulsion.

        Combines the Eikonal shortest-path gradient (attraction toward ``goal``)
        with the SDF repulsion (push away from obstacles) and zeroes the goal
        cell. The ``influence`` / ``gain`` repulsion parameters come from this
        instance.

        Args:
            grid: (n, n) numpy array where 0=free, 1=obstacle.
            goal: (row, col) target cell the field should guide toward.

        Returns:
            guidance: (n, n, 2) float32 array of (drow, dcol) guidance vectors.
        """
        arrival_time_map = self.eik.compute_arrival_time_map(grid, source=goal)
        gradient_map = self.eik.compute_gradient_map(arrival_time_map, grid)

        sdf_map = self.sdf.compute_sdf_map(grid)
        repulsive_force_map = self.sdf.compute_repulsive_force_map(
            grid,
            sdf=sdf_map,
            influence=self.influence,
            gain=self.gain,
            cell_size=self.resolution,
            falloff_power=self.falloff_power,
        )

        guidance = gradient_map + repulsive_force_map
        guidance[goal[0], goal[1]] = np.array([0.0, 0.0])
        return guidance.astype(np.float32)

    def ns_method(self, grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
        """Build the guidance field from the incompressible Navier-Stokes flow.

        Returns the unit Navier-Stokes flow direction plus the SDF repulsion
        (as in :meth:`eik_method`). Pure incompressible flow tends to graze
        walls that lie between it and the goal, so the repulsion (strength
        ``gain``) restores obstacle clearance.

        Args:
            grid: (n, n) numpy array where 0=free, 1=obstacle.
            goal: (row, col) target cell the field should guide toward.

        Returns:
            guidance: (n, n, 2) float32 array of (drow, dcol) guidance vectors.
        """
        velocity = self.ns.compute_velocity_field(grid, goal=goal)
        guidance = self.ns.compute_flow_field(velocity, grid)
        guidance[goal[0], goal[1]] = np.array([0.0, 0.0])
        return guidance.astype(np.float32)

    def bfs_method(
        self, grid: np.ndarray, goal: tuple[int, int], threshold: float = 0.05
    ) -> np.ndarray:
        """Build the guidance field from the original WaterFlow pipeline.

        Args:
            grid: (n, n) numpy array where 0=free, 1=obstacle.
            goal: (row, col) target cell (the WaterFlow source).
            threshold: Strength threshold for turning weak cells into soft
                obstacles before the flow is traced.

        Returns:
            target_vectors: (n, n, 2) float32 array of (drow, dcol) vectors.
        """
        wf = self.wf

        arrival_time_map = wf.compute_arrival_time_map(grid, source=goal)
        strength_map = wf.compute_strength_map(arrival_time_map, source=goal)
        viscosity_map = wf.compute_viscosity_map(grid, source=goal)
        filtered_map = wf.compute_filtered_map(
            strength_map, viscosity_map, grid, source=goal
        )

        thresholded_grid = wf.apply_strength_threshold(
            filtered_map, grid, threshold=threshold, source=goal
        )

        binary_thresholded_grid = (thresholded_grid != 0).astype(np.int8)
        binary_thresholded_grid[goal[0], goal[1]] = 0  # ensure goal is free

        flow_map_t = wf.compute_flow_map(binary_thresholded_grid, source=goal)
        flow_map_r = wf.reverse_flow_map(flow_map_t)
        target_vectors = wf.convert_flow_map_to_vectors(flow_map_r, strength_map)

        return target_vectors.astype(np.float32)
