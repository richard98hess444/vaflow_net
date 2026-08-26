"""3D ground-truth guidance vector field.

The 3D counterpart of :mod:`gridflow.algorithms.guidance`. It is the single
source of truth for how a target ``(di, dj, dk)`` guidance field is built from a
voxel grid and a goal, bundling the 3D solvers and the attraction/repulsion
configuration together.

:class:`VectorFieldGuidance3D` exposes three interchangeable strategies:

  * :meth:`eik_method` -- the Eikonal shortest-path gradient (attraction toward
    the goal) plus the SDF artificial-potential-field repulsion (push away from
    obstacles),
  * :meth:`ns_method` -- the incompressible Navier-Stokes flow direction, and
  * :meth:`bfs_method` -- the original WaterFlow pipeline.

Construct one instance and reuse it; the held solvers and the ``influence`` /
``gain`` repulsion parameters are then shared across every guidance call.
"""

import numpy as np
import time

from .flow import WaterFlow3D
from .eikonal import Eikonal3D
from .navier_stokes import NavierStokes3D
from .sdf import SignedDistanceField3D

DEFAULT_APF_GAIN = 0.03
DEFAULT_APF_INFLUENCE = 5.0
DEFAULT_APF_RESOLUTION = 0.1
DEFAULT_APF_FALLOFF_POWER = 0.67


class VectorFieldGuidance3D:
    """Build 3D ground-truth guidance fields, holding the solvers and config.

    The instance owns one :class:`WaterFlow3D`, :class:`Eikonal3D` and
    :class:`SignedDistanceField3D` solver plus the repulsion parameters, so the
    same objects and settings are reused across every guidance call instead of
    being threaded through as arguments.

    Args:
        apf_influence: SDF repulsion influence radius ``d0`` in voxels.
        apf_gain: SDF repulsion strength.
        apf_resolution: Physical distance represented by one voxel (``cell_size``).
        apf_falloff_power: Exponent ``p`` on the clearance in the repulsion law.
    """

    def __init__(
        self,
        apf_gain: float = DEFAULT_APF_GAIN,
        apf_influence: float = DEFAULT_APF_INFLUENCE,
        apf_resolution: float = DEFAULT_APF_RESOLUTION,
        apf_falloff_power: float = DEFAULT_APF_FALLOFF_POWER,
    ):
        self.wf = WaterFlow3D()
        self.eik = Eikonal3D()
        self.ns = NavierStokes3D(viscosity=0.9, dt=0.1, source_rate=0.05, num_steps=100)
        self.sdf = SignedDistanceField3D()
        self.apf_gain = apf_gain
        self.apf_influence = apf_influence
        self.apf_resolution = apf_resolution
        self.apf_falloff_power = apf_falloff_power

    def eik_method(self, grid: np.ndarray, goal: tuple[int, int, int]) -> np.ndarray:
        """Build the guidance field from the Eikonal gradient plus SDF repulsion.

        Combines the Eikonal shortest-path gradient (attraction toward ``goal``)
        with the SDF repulsion (push away from obstacles) and zeroes the goal
        voxel. The ``influence`` / ``gain`` repulsion parameters come from this
        instance.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            goal: (i, j, k) target voxel the field should guide toward.

        Returns:
            guidance: (n, n, n, 3) float32 array of (di, dj, dk) guidance
                vectors.
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
        guidance[goal] = np.array([0.0, 0.0, 0.0])
        return guidance.astype(np.float32)

    def ns_method(self, grid: np.ndarray, goal: tuple[int, int, int]) -> np.ndarray:
        """Build the guidance field from the incompressible Navier-Stokes flow.

        Returns the unit Navier-Stokes flow direction plus the SDF repulsion
        (as in :meth:`eik_method`). Pure incompressible flow tends to graze
        walls that lie between it and the goal, so the repulsion (strength
        ``gain``) restores obstacle clearance.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            goal: (i, j, k) target voxel the field should guide toward.

        Returns:
            guidance: (n, n, n, 3) float32 array of (di, dj, dk) guidance
                vectors.
        """
        velocity = self.ns.compute_velocity_field(grid, goal=goal)
        guidance = self.ns.compute_flow_field(velocity, grid)
        guidance[goal] = np.array([0.0, 0.0, 0.0])
        return guidance.astype(np.float32)

    def bfs_method(
        self, grid: np.ndarray, goal: tuple[int, int, int], threshold: float = 0.05
    ) -> np.ndarray:
        """Build the guidance field from the original WaterFlow pipeline.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            goal: (i, j, k) target voxel (the WaterFlow source).
            threshold: Strength threshold for turning weak voxels into soft
                obstacles before the flow is traced.

        Returns:
            target_vectors: (n, n, n, 3) float32 array of (di, dj, dk) vectors.
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
        binary_thresholded_grid[goal] = 0  # ensure goal is free

        flow_map_t = wf.compute_flow_map(binary_thresholded_grid, source=goal)
        flow_map_r = wf.reverse_flow_map(flow_map_t)
        target_vectors = wf.convert_flow_map_to_vectors(flow_map_r, strength_map)

        return target_vectors.astype(np.float32)
