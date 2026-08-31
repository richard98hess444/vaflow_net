"""Incompressible Navier-Stokes flow solver (2D) for grid navigation.

Models propagation as an actual fluid rather than an Eikonal wavefront: the
velocity field carries inertia (advection), spreads by viscosity (the
Laplacian) and stays incompressible (pressure projection), so it forms boundary
layers and recirculating wakes around obstacles that a first-arrival PDE cannot.

Integrated with Jos Stam's "Stable Fluids" scheme (advect -> diffuse ->
project) and driven like water to a plughole: every free cell is a weak uniform
source and the goal is the only outlet, so the steady field points toward the
goal from everywhere. Grid spacing ``h = 1``, unit density, no-slip solids and
no-penetration domain walls.

The diffusion and pressure operators are constant for a given grid, so their
sparse Laplacian is assembled and factorized once and reused every time step.

Inherits :class:`BreadthFirstSearch` only to reuse its grid generators.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from .bfs import BreadthFirstSearch


class NavierStokes(BreadthFirstSearch):
    """Solve incompressible Navier-Stokes on a 2D grid (Stable Fluids)."""

    def __init__(
        self,
        viscosity: float = 0.1,
        dt: float = 0.2,
        source_rate: float = 0.05,
        body_force: float = 0.0,
        num_steps: int = 300,
        drag_gain: float = 0.0,
        drag_influence: float = 3.0,
    ):
        """Configure the solver.

        Args:
            viscosity: Kinematic viscosity ``nu``. Larger = thicker, more
                sluggish fluid with fatter boundary layers; 0 = inviscid.
            dt: Time step. Advection is unconditionally stable, so this mainly
                trades convergence speed against temporal accuracy.
            source_rate: Uniform inflow injected at every free cell each step.
                The goal sink absorbs the total, so this sets the overall flow
                magnitude (direction is what matters for navigation).
            body_force: Optional extra acceleration pulling the fluid straight
                toward the goal. 0 relies purely on the source/sink pressure
                field; a small positive value adds directional "pull".
            num_steps: Number of time steps to march toward steady state.
            drag_gain: Strength of the Brinkman porous drag near obstacles. 0
                disables it. When positive, a resistance ``sigma(x)`` that grows
                toward walls damps the velocity, so the incompressible flow
                diverts around the near-wall band and keeps clearance -- the
                fluid analogue of slowing the Eikonal speed near obstacles.
            drag_influence: Distance (in cells) over which the drag ramps from
                full strength at the wall to zero.
        """
        super().__init__()
        self.viscosity = viscosity
        self.dt = dt
        self.source_rate = source_rate
        self.body_force = body_force
        self.num_steps = num_steps
        self.drag_gain = drag_gain
        self.drag_influence = drag_influence

    # ------------------------------------------------------------------
    # Grid / neighbor helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _neighbor(field: np.ndarray, dr: int, dc: int, fluid: np.ndarray, mode: str):
        """Return the ``(r+dr, c+dc)`` neighbor of ``field`` for every cell.

        Off-grid neighbors and neighbors that fall on a solid cell are
        "invalid". ``mode='zero'`` fills invalid neighbors with 0 (a no-slip
        wall for velocities); ``mode='self'`` fills them with the cell's own
        value (a Neumann / no-flux wall for pressure).

        Returns ``(neighbor_values, valid_mask)``.
        """
        nb = np.roll(field, shift=(-dr, -dc), axis=(0, 1))
        valid = np.roll(fluid, shift=(-dr, -dc), axis=(0, 1)).copy()
        # Wrapped-around border cells have no real neighbor in this direction.
        if dr == 1:
            valid[-1, :] = False
        elif dr == -1:
            valid[0, :] = False
        if dc == 1:
            valid[:, -1] = False
        elif dc == -1:
            valid[:, 0] = False
        fill = 0.0 if mode == "zero" else field
        nb = np.where(valid, nb, fill)
        return nb, valid

    @staticmethod
    def _bilinear(field: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
        """Bilinearly sample ``field`` at floating-point positions ``(rr, cc)``."""
        n = field.shape[0]
        rr = np.clip(np.nan_to_num(rr), 0.0, n - 1.0)
        cc = np.clip(np.nan_to_num(cc), 0.0, n - 1.0)
        r0 = np.floor(rr).astype(int)
        c0 = np.floor(cc).astype(int)
        r1 = np.minimum(r0 + 1, n - 1)
        c1 = np.minimum(c0 + 1, n - 1)
        fr = rr - r0
        fc = cc - c0
        top = field[r0, c0] * (1 - fc) + field[r0, c1] * fc
        bot = field[r1, c0] * (1 - fc) + field[r1, c1] * fc
        return top * (1 - fr) + bot * fr

    # ------------------------------------------------------------------
    # Navier-Stokes sub-steps
    # ------------------------------------------------------------------
    def _advect(self, vr, vc, fluid):
        """Semi-Lagrangian advection: carry velocity along the flow (inertia)."""
        n = vr.shape[0]
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        r_back = rows - self.dt * vr
        c_back = cols - self.dt * vc
        vr_new = self._bilinear(vr, r_back, c_back)
        vc_new = self._bilinear(vc, r_back, c_back)
        return np.where(fluid, vr_new, 0.0), np.where(fluid, vc_new, 0.0)

    def _build_laplacian(self, fluid, idx, cells):
        """Assemble the sparse 5-point Neumann Laplacian over the fluid cells.

        ``L_ii = -deg_i`` and ``L_ij = +1`` per fluid neighbor; missing neighbors
        (walls/solids) are dropped, giving the no-flux boundary. Drives both the
        viscous solve ``I - nu*dt*L`` and the pressure Poisson ``L p = ...``.
        """
        n = fluid.shape[0]
        N = len(cells)
        r, c = cells[:, 0], cells[:, 1]
        deg = np.zeros(N)
        rows, colv, vals = [], [], []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rn, cn = r + dr, c + dc
            inb = (rn >= 0) & (rn < n) & (cn >= 0) & (cn < n)
            nb_fluid = np.zeros(N, dtype=bool)
            nb_fluid[inb] = fluid[rn[inb], cn[inb]]
            si = np.nonzero(nb_fluid)[0]
            rows.append(si)
            colv.append(idx[rn[si], cn[si]])
            vals.append(np.ones(si.size))
            deg += nb_fluid
        rows.append(np.arange(N))
        colv.append(np.arange(N))
        vals.append(-deg)
        return csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(colv))),
            shape=(N, N),
        )

    def _project(self, vr, vc, fluid, q, project_solve, pins):
        """Pressure projection enforcing ``div u = q`` via the factorized solve.

        Solves ``laplacian(p) = div(u) - q`` (goal pinned to ``p = 0`` as the
        outlet) and subtracts the forward-difference gradient. Forward gradient
        plus the backward divergence keeps the composed operator equal to the
        compact 5-point Laplacian that was factorized.
        """
        rhs = (self._divergence(vr, vc, fluid) - q)[fluid].copy()
        rhs[pins] = 0.0
        p = np.zeros_like(vr)
        p[fluid] = project_solve(rhs)

        p_r, _ = self._neighbor(p, 0, 1, fluid, "self")
        p_d, _ = self._neighbor(p, 1, 0, fluid, "self")
        vc = np.where(fluid, vc - (p_r - p), 0.0)
        vr = np.where(fluid, vr - (p_d - p), 0.0)
        return vr, vc, p

    def _divergence(self, vr, vc, fluid):
        """Backward-difference velocity divergence (walls contribute zero flux)."""
        vc_l, _ = self._neighbor(vc, 0, -1, fluid, "zero")
        vr_u, _ = self._neighbor(vr, -1, 0, fluid, "zero")
        return (vc - vc_l) + (vr - vr_u)

    @staticmethod
    def _seal_domain(vr, vc, fluid):
        """Zero the wall-normal velocity at every wall face (no-penetration)."""
        solid = ~fluid
        vr[0, :] = 0.0
        vr[-1, :] = 0.0
        vc[:, 0] = 0.0
        vc[:, -1] = 0.0

        east = np.zeros_like(fluid)
        east[:, :-1] = solid[:, 1:]
        west = np.zeros_like(fluid)
        west[:, 1:] = solid[:, :-1]
        down = np.zeros_like(fluid)
        down[:-1, :] = solid[1:, :]
        up = np.zeros_like(fluid)
        up[1:, :] = solid[:-1, :]

        vc = np.where(east & (vc > 0.0), 0.0, vc)
        vc = np.where(west & (vc < 0.0), 0.0, vc)
        vr = np.where(down & (vr > 0.0), 0.0, vr)
        vr = np.where(up & (vr < 0.0), 0.0, vr)
        return vr, vc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_velocity_field(
        self,
        grid: np.ndarray,
        goal: tuple[int, int] = None,
        num_steps: int = None,
        force_field: np.ndarray = None,
    ) -> np.ndarray:
        """March the Navier-Stokes equations to a (near) steady velocity field.

        Args:
            grid: (n, n) numpy array where 0=free, 1=obstacle.
            goal: (row, col) sink the fluid drains toward. Defaults to the grid
                center (same convention as the other solvers).
            num_steps: Override the configured number of time steps.
            force_field: Optional (n, n, 2) acceleration ``(f_row, f_col)`` added
                to every fluid cell each step *before* the pressure projection.
                Because it is injected inside the solve, the projection makes the
                result divergence-free, so e.g. an obstacle-repulsion force is
                absorbed smoothly instead of tacked on as a discontinuous spike.

        Returns:
            velocity: (n, n, 2) array of ``(v_row, v_col)`` velocities. Zero on
                obstacles and in unreachable pockets.
        """
        n = len(grid)
        if goal is None:
            goal = (n // 2, n // 2)
        if num_steps is None:
            num_steps = self.num_steps

        fluid = grid == 0
        gr, gc = goal
        if not fluid[gr, gc]:
            raise ValueError("goal must be a free cell")

        # Uniform inflow; the goal (pinned to p=0 below) is the only outlet.
        q = np.where(fluid, self.source_rate, 0.0)

        # Unit "toward the goal" direction for the optional body force.
        rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        fr = gr - rows
        fc = gc - cols
        fnorm = np.hypot(fr, fc)
        safe = fnorm > 1e-9
        dir_r = np.where(safe, fr / np.where(safe, fnorm, 1.0), 0.0)
        dir_c = np.where(safe, fc / np.where(safe, fnorm, 1.0), 0.0)

        vr = np.zeros((n, n))
        vc = np.zeros((n, n))

        # Assemble and factorize the constant operators once, reuse every step.
        idx = -np.ones((n, n), dtype=np.int64)
        cells = np.argwhere(fluid)
        idx[fluid] = np.arange(len(cells))
        laplacian = self._build_laplacian(fluid, idx, cells)

        a = self.viscosity * self.dt
        # Brinkman porous drag sigma(x): strong at walls, ramping to 0 over
        # drag_influence cells, folded into the implicit velocity solve so the
        # flow diverts smoothly around obstacles and keeps clearance.
        sigma_cells = None
        if self.drag_gain > 0.0:
            dist = distance_transform_edt(fluid)
            sigma = (
                self.drag_gain
                * np.clip(1.0 - dist / self.drag_influence, 0.0, 1.0) ** 2
            )
            sigma_cells = sigma[fluid]
        implicit_solve = None
        if a > 0.0 or sigma_cells is not None:
            implicit_op = identity(len(cells), format="csc")
            if a > 0.0:
                implicit_op = implicit_op - a * laplacian
            if sigma_cells is not None:
                implicit_op = implicit_op + self.dt * diags(sigma_cells)
            implicit_solve = splu(implicit_op.tocsc()).solve

        # The Neumann Laplacian has one null mode per connected fluid region, so
        # pin one cell per component to p=0 (the goal for its own component, an
        # arbitrary cell elsewhere) and zero the source in pockets that cannot
        # reach the goal so unreachable regions stay at rest.
        ref = int(idx[gr, gc])
        _, labels = connected_components(laplacian, directed=False)
        goal_comp = labels[ref]
        q[fluid] = np.where(labels == goal_comp, self.source_rate, 0.0)
        pins = []
        for comp in range(labels.max() + 1):
            members = np.nonzero(labels == comp)[0]
            pins.append(ref if comp == goal_comp else int(members[0]))
        pins = np.array(pins, dtype=np.int64)

        pressure_op = laplacian.tolil()
        for pin in pins:
            pressure_op.rows[pin] = [int(pin)]
            pressure_op.data[pin] = [1.0]
        project_solve = splu(pressure_op.tocsc()).solve

        for _ in range(num_steps):
            if self.body_force > 0.0:
                vr = np.where(fluid, vr + self.dt * self.body_force * dir_r, 0.0)
                vc = np.where(fluid, vc + self.dt * self.body_force * dir_c, 0.0)
            if force_field is not None:
                vr = np.where(fluid, vr + self.dt * force_field[..., 0], 0.0)
                vc = np.where(fluid, vc + self.dt * force_field[..., 1], 0.0)

            vr, vc = self._advect(vr, vc, fluid)
            if implicit_solve is not None:
                vr[fluid] = implicit_solve(vr[fluid])
                vc[fluid] = implicit_solve(vc[fluid])
            vr, vc, _ = self._project(vr, vc, fluid, q, project_solve, pins)
            vr, vc = self._seal_domain(vr, vc, fluid)

        return np.stack([vr, vc], axis=-1)

    def compute_flow_field(
        self, velocity_field: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        """Unit-normalize the velocity into a flow-direction field.

        Mirrors :meth:`Eikonal.compute_gradient_map`: (n, n, 2) unit vectors
        pointing along the local flow (toward the goal), zero where the fluid is
        stagnant or blocked.
        """
        v = np.array(velocity_field, dtype=float)
        mag = np.hypot(v[..., 0], v[..., 1])
        out = np.zeros_like(v)
        moving = (grid == 0) & (mag > 1e-9)
        out[..., 0] = np.where(moving, v[..., 0] / np.where(moving, mag, 1.0), 0.0)
        out[..., 1] = np.where(moving, v[..., 1] / np.where(moving, mag, 1.0), 0.0)
        return out

    def compute_flow_direction(
        self,
        grid: np.ndarray,
        goal: tuple[int, int] = None,
        force_field: np.ndarray = None,
        num_steps: int = None,
    ) -> np.ndarray:
        """Solve for the steady velocity and return its unit flow direction.

        Convenience wrapper around :meth:`compute_velocity_field` followed by
        :meth:`compute_flow_field`; the ``force_field`` and the configured
        ``drag_gain`` are applied inside the solve.
        """
        velocity = self.compute_velocity_field(
            grid, goal=goal, num_steps=num_steps, force_field=force_field
        )
        return self.compute_flow_field(velocity, grid)

    @staticmethod
    def boundary_taper(shape: tuple[int, int], margin: float) -> np.ndarray:
        """Smoothstep weight in [0, 1]: 0 at the domain edge, 1 at ``margin`` cells in.

        Used to blend an edge band of a field back toward a reference flow (e.g.
        the reversed one-cell ring the injected repulsion leaves at the map wall).
        """
        if margin <= 0.0:
            return np.ones(shape, dtype=float)
        axes = []
        for size in shape:
            idx = np.arange(size)
            axes.append(np.minimum(idx, idx[::-1]))
        dist = axes[0][:, None]
        for ax in axes[1:]:
            dist = np.minimum(dist, ax[None, :])
        t = np.clip(dist.astype(float) / margin, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def to_display_map(
        self, velocity_field: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        v = np.asarray(velocity_field, dtype=float)
        speed = np.hypot(v[..., 0], v[..., 1])
        speed[grid == 1] = -1.0
        return speed

    def compute_navigation_field(
        self, velocity_field: np.ndarray, grid: np.ndarray, gamma: float = 1.0
    ) -> np.ndarray:
        """Flow direction scaled by a pressure-based speed (slow near the goal).

        The raw fluid is fastest at the goal (flux converging into the sink),
        which is backwards for navigation. The steady pressure ``self.pressure_``
        is a natural potential that is lowest at the goal and rises outward, so
        scaling the unit flow direction by the normalized pressure yields a field
        that is (near) zero at the goal and speeds up with distance -- a smooth
        arrival profile taken straight from the solved field rather than an
        ad-hoc remap. Requires a prior :meth:`compute_velocity_field` call.

        Args:
            velocity_field: (n, n, 2) field from :meth:`compute_velocity_field`.
            grid: (n, n) numpy array where 0=free, 1=obstacle.
            gamma: Speed-profile exponent (>1 lingers slow near the goal longer,
                <1 ramps up faster).

        Returns:
            field: (n, n, 2) direction vectors scaled to speed in [0, 1].
        """
        if getattr(self, "pressure_", None) is None:
            raise RuntimeError("call compute_velocity_field() first")
        direction = self.compute_flow_field(velocity_field, grid)
        fluid = grid == 0
        speed = np.zeros(grid.shape, dtype=float)
        pf = self.pressure_[fluid]
        pf = pf - pf.min()
        pmax = pf.max()
        if pmax > 0:
            speed[fluid] = (pf / pmax) ** gamma
        return direction * speed[..., None]
