"""Incompressible Navier-Stokes flow solver (3D) for volumetric grid navigation.

3D counterpart of :mod:`gridflow.algorithms.navier_stokes`. The lattice
``(row, col)`` becomes a voxel lattice ``(i, j, k)``, the 5-point Laplacian
becomes the 7-point (6-neighbor) Laplacian and every ``(n, n)`` array becomes
``(n, n, n)``. Same "Stable Fluids" scheme and water-to-a-plughole driving
(uniform source everywhere, goal the only outlet).

Direct sparse factorization (``splu``) is impractical at ``61^3`` because of 3D
fill-in, so the pressure Poisson and viscous solves use conjugate gradient (CG)
instead, warm-started from the previous step -- near steady state the field
barely changes between steps, so CG converges in a handful of iterations.

Inherits :class:`BreadthFirstSearch3D` only to reuse its grid generators.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

from .bfs import BreadthFirstSearch3D


class NavierStokes3D(BreadthFirstSearch3D):
    """Solve incompressible Navier-Stokes on a 3D voxel grid (Stable Fluids)."""

    def __init__(
        self,
        viscosity: float = 0.1,
        dt: float = 0.2,
        source_rate: float = 0.05,
        body_force: float = 0.0,
        num_steps: int = 300,
        cg_tol: float = 1e-4,
        cg_maxiter: int = 400,
        drag_gain: float = 0.0,
        drag_influence: float = 3.0,
    ):
        """Configure the solver (see the 2D class; ``cg_*`` tune the solves)."""
        super().__init__()
        self.viscosity = viscosity
        self.dt = dt
        self.source_rate = source_rate
        self.body_force = body_force
        self.num_steps = num_steps
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self.drag_gain = drag_gain
        self.drag_influence = drag_influence
        self.pressure_ = None  # steady-state pressure, set by compute_velocity_field

    # ------------------------------------------------------------------
    # Voxel / neighbor helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _neighbor(field, di, dj, dk, fluid, mode):
        """``(i+di, j+dj, k+dk)`` neighbor of ``field`` (see 2D ``_neighbor``)."""
        nb = np.roll(field, shift=(-di, -dj, -dk), axis=(0, 1, 2))
        valid = np.roll(fluid, shift=(-di, -dj, -dk), axis=(0, 1, 2)).copy()
        if di == 1:
            valid[-1, :, :] = False
        elif di == -1:
            valid[0, :, :] = False
        if dj == 1:
            valid[:, -1, :] = False
        elif dj == -1:
            valid[:, 0, :] = False
        if dk == 1:
            valid[:, :, -1] = False
        elif dk == -1:
            valid[:, :, 0] = False
        fill = 0.0 if mode == "zero" else field
        return np.where(valid, nb, fill), valid

    @staticmethod
    def _trilinear(field, ii, jj, kk):
        """Trilinearly sample ``field`` at floating-point voxel positions."""
        n = field.shape[0]
        ii = np.clip(np.nan_to_num(ii), 0.0, n - 1.0)
        jj = np.clip(np.nan_to_num(jj), 0.0, n - 1.0)
        kk = np.clip(np.nan_to_num(kk), 0.0, n - 1.0)
        i0, j0, k0 = np.floor(ii).astype(int), np.floor(jj).astype(int), np.floor(kk).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        j1 = np.minimum(j0 + 1, n - 1)
        k1 = np.minimum(k0 + 1, n - 1)
        fi, fj, fk = ii - i0, jj - j0, kk - k0
        out = 0.0
        for wi, ia in ((1 - fi, i0), (fi, i1)):
            for wj, ja in ((1 - fj, j0), (fj, j1)):
                for wk, ka in ((1 - fk, k0), (fk, k1)):
                    out = out + wi * wj * wk * field[ia, ja, ka]
        return out

    # ------------------------------------------------------------------
    # Navier-Stokes sub-steps
    # ------------------------------------------------------------------
    def _advect(self, vi, vj, vk, fluid):
        """Semi-Lagrangian advection (inertia)."""
        n = vi.shape[0]
        gi, gj, gk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        i_back = gi - self.dt * vi
        j_back = gj - self.dt * vj
        k_back = gk - self.dt * vk
        vi_n = np.where(fluid, self._trilinear(vi, i_back, j_back, k_back), 0.0)
        vj_n = np.where(fluid, self._trilinear(vj, i_back, j_back, k_back), 0.0)
        vk_n = np.where(fluid, self._trilinear(vk, i_back, j_back, k_back), 0.0)
        return vi_n, vj_n, vk_n

    def _build_laplacian(self, fluid, idx, cells):
        """Sparse 7-point Neumann Laplacian over the fluid voxels (see 2D)."""
        n = fluid.shape[0]
        N = len(cells)
        i, j, k = cells[:, 0], cells[:, 1], cells[:, 2]
        deg = np.zeros(N)
        rows, colv, vals = [], [], []
        for di, dj, dk in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            ni, nj, nk = i + di, j + dj, k + dk
            inb = (ni >= 0) & (ni < n) & (nj >= 0) & (nj < n) & (nk >= 0) & (nk < n)
            nb_fluid = np.zeros(N, dtype=bool)
            nb_fluid[inb] = fluid[ni[inb], nj[inb], nk[inb]]
            si = np.nonzero(nb_fluid)[0]
            rows.append(si)
            colv.append(idx[ni[si], nj[si], nk[si]])
            vals.append(np.ones(si.size))
            deg += nb_fluid
        rows.append(np.arange(N))
        colv.append(np.arange(N))
        vals.append(-deg)
        return csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(colv))),
            shape=(N, N),
        )

    def _divergence(self, vi, vj, vk, fluid):
        """Backward-difference velocity divergence (walls contribute no flux)."""
        vi_u, _ = self._neighbor(vi, -1, 0, 0, fluid, "zero")
        vj_u, _ = self._neighbor(vj, 0, -1, 0, fluid, "zero")
        vk_u, _ = self._neighbor(vk, 0, 0, -1, fluid, "zero")
        return (vi - vi_u) + (vj - vj_u) + (vk - vk_u)

    def _project(self, vi, vj, vk, fluid, q, a_p, x0):
        """Pressure projection enforcing ``div u = q`` via warm-started CG."""
        b = (q - self._divergence(vi, vj, vk, fluid))[fluid]
        p_flat, _ = cg(a_p, b, x0=x0, rtol=self.cg_tol, maxiter=self.cg_maxiter)
        p = np.zeros_like(vi)
        p[fluid] = p_flat

        p_i, _ = self._neighbor(p, 1, 0, 0, fluid, "self")
        p_j, _ = self._neighbor(p, 0, 1, 0, fluid, "self")
        p_k, _ = self._neighbor(p, 0, 0, 1, fluid, "self")
        vi = np.where(fluid, vi - (p_i - p), 0.0)
        vj = np.where(fluid, vj - (p_j - p), 0.0)
        vk = np.where(fluid, vk - (p_k - p), 0.0)
        return vi, vj, vk, p_flat

    @staticmethod
    def _seal_domain(vi, vj, vk, fluid):
        """Zero the wall-normal velocity at every wall face (no-penetration)."""
        solid = ~fluid
        vi[0, :, :] = 0.0
        vi[-1, :, :] = 0.0
        vj[:, 0, :] = 0.0
        vj[:, -1, :] = 0.0
        vk[:, :, 0] = 0.0
        vk[:, :, -1] = 0.0

        ip = np.zeros_like(fluid)
        ip[:-1, :, :] = solid[1:, :, :]
        im = np.zeros_like(fluid)
        im[1:, :, :] = solid[:-1, :, :]
        jp = np.zeros_like(fluid)
        jp[:, :-1, :] = solid[:, 1:, :]
        jm = np.zeros_like(fluid)
        jm[:, 1:, :] = solid[:, :-1, :]
        kp = np.zeros_like(fluid)
        kp[:, :, :-1] = solid[:, :, 1:]
        km = np.zeros_like(fluid)
        km[:, :, 1:] = solid[:, :, :-1]

        vi = np.where(ip & (vi > 0.0), 0.0, vi)
        vi = np.where(im & (vi < 0.0), 0.0, vi)
        vj = np.where(jp & (vj > 0.0), 0.0, vj)
        vj = np.where(jm & (vj < 0.0), 0.0, vj)
        vk = np.where(kp & (vk > 0.0), 0.0, vk)
        vk = np.where(km & (vk < 0.0), 0.0, vk)
        return vi, vj, vk

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_velocity_field(self, grid, goal=None, num_steps=None, force_field=None):
        """March Navier-Stokes to a (near) steady velocity field.

        Args:
            grid: (n, n, n) array where 0=free, 1=obstacle.
            goal: (i, j, k) outlet the fluid drains toward (default: center).
            num_steps: Override the configured number of time steps.
            force_field: Optional (n, n, n, 3) acceleration added to every fluid
                voxel each step before the pressure projection (kept smooth and
                divergence-free by the projection).

        Returns:
            velocity: (n, n, n, 3) array of ``(v_i, v_j, v_k)`` velocities, zero
                on obstacles and in unreachable pockets.
        """
        n = len(grid)
        if goal is None:
            goal = (n // 2, n // 2, n // 2)
        if num_steps is None:
            num_steps = self.num_steps

        fluid = grid == 0
        if not fluid[goal]:
            raise ValueError("goal must be a free voxel")

        idx = -np.ones((n, n, n), dtype=np.int64)
        cells = np.argwhere(fluid)
        idx[fluid] = np.arange(len(cells))
        laplacian = self._build_laplacian(fluid, idx, cells)

        # Diffusion operator (SPD, well-conditioned) and pressure operator
        # (-L, made strictly SPD by a tiny Tikhonov term so CG is robust on the
        # otherwise-singular Neumann system without per-pocket bookkeeping).
        a = self.viscosity * self.dt
        eye = identity(len(cells), format="csc")
        # Brinkman porous drag sigma(x): strong at walls, ramping to 0 over
        # drag_influence voxels, folded into the implicit velocity solve so the
        # flow diverts around obstacles and keeps clearance.
        sigma_cells = None
        if self.drag_gain > 0.0:
            dist = distance_transform_edt(fluid)
            sigma = (
                self.drag_gain
                * np.clip(1.0 - dist / self.drag_influence, 0.0, 1.0) ** 2
            )
            sigma_cells = sigma[fluid]
        a_diff = None
        if a > 0 or sigma_cells is not None:
            a_diff = eye
            if a > 0:
                a_diff = a_diff - a * laplacian
            if sigma_cells is not None:
                a_diff = a_diff + self.dt * diags(sigma_cells)
            a_diff = a_diff.tocsc()
        a_p = (-laplacian + 1e-6 * eye).tocsc()

        # Uniform source in the goal's connected region, cancelled by a sink at
        # the goal; pockets that cannot reach the goal get no source (stay put).
        ref = int(idx[goal])
        _, labels = connected_components(laplacian, directed=False)
        goal_comp = labels[ref]
        qf = np.where(labels == goal_comp, self.source_rate, 0.0)
        qf[ref] -= qf.sum()
        q = np.zeros((n, n, n))
        q[fluid] = qf

        gi, gj, gk = goal
        ci, cj, ck = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        fi, fj, fk = gi - ci, gj - cj, gk - ck
        fnorm = np.sqrt(fi * fi + fj * fj + fk * fk)
        safe = fnorm > 1e-9
        dir_i = np.where(safe, fi / np.where(safe, fnorm, 1.0), 0.0)
        dir_j = np.where(safe, fj / np.where(safe, fnorm, 1.0), 0.0)
        dir_k = np.where(safe, fk / np.where(safe, fnorm, 1.0), 0.0)

        vi = np.zeros((n, n, n))
        vj = np.zeros((n, n, n))
        vk = np.zeros((n, n, n))
        p_prev = np.zeros(len(cells))

        for _ in range(num_steps):
            if self.body_force > 0.0:
                vi = np.where(fluid, vi + self.dt * self.body_force * dir_i, 0.0)
                vj = np.where(fluid, vj + self.dt * self.body_force * dir_j, 0.0)
                vk = np.where(fluid, vk + self.dt * self.body_force * dir_k, 0.0)
            if force_field is not None:
                vi = np.where(fluid, vi + self.dt * force_field[..., 0], 0.0)
                vj = np.where(fluid, vj + self.dt * force_field[..., 1], 0.0)
                vk = np.where(fluid, vk + self.dt * force_field[..., 2], 0.0)

            vi, vj, vk = self._advect(vi, vj, vk, fluid)
            if a_diff is not None:
                vi[fluid], _ = cg(a_diff, vi[fluid], x0=vi[fluid], rtol=self.cg_tol, maxiter=self.cg_maxiter)
                vj[fluid], _ = cg(a_diff, vj[fluid], x0=vj[fluid], rtol=self.cg_tol, maxiter=self.cg_maxiter)
                vk[fluid], _ = cg(a_diff, vk[fluid], x0=vk[fluid], rtol=self.cg_tol, maxiter=self.cg_maxiter)
            vi, vj, vk, p_prev = self._project(vi, vj, vk, fluid, q, a_p, p_prev)
            vi, vj, vk = self._seal_domain(vi, vj, vk, fluid)

        # Steady-state pressure: low at the goal outlet, high far away.
        pressure = np.zeros((n, n, n))
        pressure[fluid] = p_prev
        self.pressure_ = pressure
        return np.stack([vi, vj, vk], axis=-1)

    def compute_flow_field(self, velocity_field, grid):
        """Unit-normalize the velocity into a (n, n, n, 3) flow-direction field."""
        v = np.array(velocity_field, dtype=float)
        mag = np.sqrt((v * v).sum(axis=-1))
        moving = (grid == 0) & (mag > 1e-9)
        scale = np.where(moving, mag, 1.0)
        out = np.zeros_like(v)
        for c in range(3):
            out[..., c] = np.where(moving, v[..., c] / scale, 0.0)
        return out

    def compute_flow_direction(self, grid, goal=None, force_field=None, num_steps=None):
        """Solve for the steady velocity and return its unit flow direction."""
        velocity = self.compute_velocity_field(
            grid, goal=goal, num_steps=num_steps, force_field=force_field
        )
        return self.compute_flow_field(velocity, grid)

    @staticmethod
    def boundary_taper(shape, margin):
        """Smoothstep weight in [0, 1]: 0 at the domain edge, 1 at ``margin`` voxels in."""
        if margin <= 0.0:
            return np.ones(shape, dtype=float)
        axes = [np.minimum(np.arange(s), np.arange(s)[::-1]) for s in shape]
        dist = axes[0][:, None, None]
        dist = np.minimum(dist, axes[1][None, :, None])
        dist = np.minimum(dist, axes[2][None, None, :])
        t = np.clip(dist.astype(float) / margin, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def to_display_map(self, velocity_field, grid):
        """Speed magnitude ``|u|`` with obstacles marked -1."""
        v = np.asarray(velocity_field, dtype=float)
        speed = np.sqrt((v * v).sum(axis=-1))
        speed[grid == 1] = -1.0
        return speed

    def compute_navigation_field(self, velocity_field, grid, gamma=1.0):
        """Flow direction scaled by a pressure-based speed (slow near the goal).

        The raw fluid is fastest at the goal (flux converging into the sink); the
        steady pressure ``self.pressure_`` is lowest at the goal and rises
        outward, so scaling the unit direction by the normalized pressure gives a
        field that is ~zero at the goal and speeds up with distance. Requires a
        prior :meth:`compute_velocity_field` call. ``gamma`` shapes the profile.
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
