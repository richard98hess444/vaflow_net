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

from .flow import WaterFlow3D
from .eikonal import Eikonal3D
from .navier_stokes import NavierStokes3D
from .sdf import SignedDistanceField3D

DEFAULT_APF_GAIN = 0.03
DEFAULT_APF_INFLUENCE = 5.0
DEFAULT_APF_RESOLUTION = 0.1
DEFAULT_APF_FALLOFF_POWER = 0.67
DEFAULT_DRAG_GAIN = 5.0
DEFAULT_DRAG_INFLUENCE = 3.0
DEFAULT_NS_FORCE_BOUNDARY = 3.0


class VectorFieldGuidance3D:
    """Build 3D ground-truth guidance fields, holding the solvers and config.
    Args:
        apf_influence: SDF repulsion influence radius ``d0`` in voxels.
        apf_gain: SDF repulsion strength.
        apf_resolution: Physical distance represented by one voxel (``cell_size``).
        apf_falloff_power: Exponent ``p`` on the clearance in the repulsion law.
        drag_gain: Drag force gain.
        drag_influence: Drag force influence radius.
        ns_force_boundary: Navier-Stokes force boundary.
    """

    def __init__(
        self,
        apf_gain: float = DEFAULT_APF_GAIN,
        apf_influence: float = DEFAULT_APF_INFLUENCE,
        apf_resolution: float = DEFAULT_APF_RESOLUTION,
        apf_falloff_power: float = DEFAULT_APF_FALLOFF_POWER,
        drag_gain: float = DEFAULT_DRAG_GAIN,
        drag_influence: float = DEFAULT_DRAG_INFLUENCE,
        ns_force_boundary: float = DEFAULT_NS_FORCE_BOUNDARY,
    ):
        self.wf = WaterFlow3D()
        self.eik = Eikonal3D()
        self.ns = NavierStokes3D(viscosity=0.9, dt=0.1, source_rate=0.05, num_steps=100)
        self.sdf = SignedDistanceField3D()
        self.apf_gain = apf_gain
        self.apf_influence = apf_influence
        self.apf_resolution = apf_resolution
        self.apf_falloff_power = apf_falloff_power
        self.drag_gain = drag_gain
        self.drag_influence = drag_influence
        self.ns_force_boundary = ns_force_boundary

    def _repulsion(self, grid: np.ndarray) -> np.ndarray:
        """SDF artificial-potential-field repulsion using this instance's config."""
        sdf_map = self.sdf.compute_sdf_map(grid)
        return self.sdf.compute_repulsive_force_map(
            grid,
            sdf=sdf_map,
            influence=self.apf_influence,
            gain=self.apf_gain,
            cell_size=self.apf_resolution,
            falloff_power=self.apf_falloff_power,
        )

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
            guidance: (n, n, n, 3) float32 array of (di, dj, dk) guidance vectors.
        """
        arrival_time_map = self.eik.compute_arrival_time_map(grid, source=goal)
        gradient_map = self.eik.compute_gradient_map(arrival_time_map, grid)
        repulsive_force_map = self._repulsion(grid)

        guidance = gradient_map + repulsive_force_map
        guidance[goal] = np.array([0.0, 0.0, 0.0])
        return guidance.astype(np.float32)

    def _ns_flow(
        self,
        grid: np.ndarray,
        goal: tuple[int, int, int],
        force_field: np.ndarray = None,
        drag: bool = False,
    ) -> np.ndarray:
        """Solve NS and return the unit flow direction (optional force / drag)."""
        if drag:
            self.ns.drag_gain = self.drag_gain
            self.ns.drag_influence = self.drag_influence
        try:
            return self.ns.compute_flow_direction(
                grid, goal=goal, force_field=force_field
            )
        finally:
            self.ns.drag_gain = 0.0

    def ns_method(
        self, grid: np.ndarray, goal: tuple[int, int, int], clearance: str = "apf"
    ) -> np.ndarray:
        """Build the guidance field from the incompressible Navier-Stokes flow.

        3D counterpart of :meth:`gridflow.algorithms.guidance.VectorFieldGuidance.ns_method`.
        ``clearance`` selects how obstacle clearance is added:

          * ``"apf"``  -- SDF repulsion added *after* the solve (original).
          * ``"force"`` -- the repulsion injected as a body force *before* the
            projection (smooth, divergence-free); ``ns_force_boundary`` blends
            the reversed boundary shell back to the plain flow.
          * ``"drag"`` -- Brinkman porous drag near walls, diverting the flow to
            keep clearance from inside the solve (single solve).
          * ``"none"`` -- raw flow; clearance only from wall no-penetration.

        Args:
            grid: (n, n, n) numpy array where 0=free, 1=obstacle.
            goal: (i, j, k) target voxel the field should guide toward.
            clearance: Clearance strategy, one of the values above.

        Returns:
            guidance: (n, n, n, 3) float32 array of (di, dj, dk) guidance vectors.
        """
        if clearance == "none":
            guidance = self._ns_flow(grid, goal)
        elif clearance == "drag":
            guidance = self._ns_flow(grid, goal, drag=True)
        elif clearance == "apf":
            guidance = self._ns_flow(grid, goal) + self._repulsion(grid)
        elif clearance == "force":
            guidance = self._ns_flow(grid, goal, force_field=self._repulsion(grid))
            if self.ns_force_boundary > 0.0:
                plain = self._ns_flow(grid, goal)
                w = self.ns.boundary_taper(grid.shape, self.ns_force_boundary)
                guidance = w[..., None] * guidance + (1.0 - w[..., None]) * plain
        else:
            raise ValueError(
                f"Unknown clearance {clearance!r}; expected "
                "'none', 'apf', 'force' or 'drag'."
            )

        guidance = np.array(guidance)
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
