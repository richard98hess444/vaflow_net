"""GPU-batched incompressible Navier-Stokes (drag) solver for fast targets.

A PyTorch port of :class:`gridflow.algorithms.navier_stokes.NavierStokes` (and its
3D sibling) restricted to the ``drag`` clearance mode -- the one the guidance uses
for training targets. The numpy solvers factorize one sparse operator per grid and
solve a single sample at a time (~0.18 s per 61x61 sample, and far worse in 3D),
which is the dataset-generation bottleneck. This module instead marches many grids
at once on the GPU:

  * every sub-step (advect -> diffuse+drag -> project -> seal) is a batched tensor
    op over ``B`` grids;
  * the diffusion+drag and pressure solves use a batched conjugate-gradient (the
    same iterative approach the 3D numpy solver already uses), the pressure solve
    warm-started across time steps;
    * the discretization follows the numpy reference (compact 5-/7-point Neumann
        Laplacian, backward-difference divergence + forward-difference pressure
        gradient, no-penetration wall seal), then applies the same navigation-speed
        mapping as :meth:`VectorFieldGuidance.ns_method` (``clearance="drag"``).

The per-sample preprocessing that is awkward on the GPU -- picking the goal's
connected fluid component and the distance-to-obstacle drag profile -- is done once
on the CPU with SciPy (cheap) and handed to the batched solver as source/sink and
drag fields.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label


def _slice_zero(y: torch.Tensor, dim: int, index: int) -> torch.Tensor:
    """Zero one hyper-slice of ``y`` along ``dim`` (in place; ``y`` is fresh)."""
    idx = [slice(None)] * y.dim()
    idx[dim] = index
    y[tuple(idx)] = 0.0
    return y


class _NSTorch:
    """Batched Stable-Fluids drag solver shared by the 2D and 3D fronts.

    ``dirs`` are the axis-aligned neighbor offsets (4 in 2D, 6 in 3D); every field
    is ``(B, *spatial)`` and the spatial rank is inferred from ``dirs``.
    """

    def __init__(
        self,
        viscosity: float = 0.9,
        dt: float = 0.1,
        source_rate: float = 0.05,
        num_steps: int = 50,
        drag_gain: float = 3.7,
        drag_influence: float = 3.0,
        diff_iters: int = 30,
        proj_iters: int = 50,
        cg_tol: float = 1e-4,
        pressure_reg: float = 1e-3,
        use_advection: bool = True,
        device: str | torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.viscosity = viscosity
        self.dt = dt
        self.source_rate = source_rate
        self.num_steps = num_steps
        self.drag_gain = drag_gain
        self.drag_influence = drag_influence
        self.diff_iters = diff_iters
        self.proj_iters = proj_iters
        self.cg_tol = cg_tol
        self.pressure_reg = pressure_reg
        self.use_advection = use_advection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Neighbor shifts and discrete operators (dimension-agnostic)
    # ------------------------------------------------------------------
    def _shift(self, x: torch.Tensor, offset: tuple[int, ...]) -> torch.Tensor:
        """Value of ``x`` at the ``offset`` neighbor; wrapped edges zeroed."""
        dims = tuple(range(1, x.dim()))
        y = torch.roll(x, shifts=tuple(-o for o in offset), dims=dims)
        for axis, o in enumerate(offset):
            if o == 1:
                _slice_zero(y, axis + 1, -1)
            elif o == -1:
                _slice_zero(y, axis + 1, 0)
        return y

    def _build_laplacian(self, fluid: torch.Tensor):
        """Prepare the matrix-free Neumann Laplacian for a batch of grids.

        The CPU solver assembles a sparse matrix. Building one matrix per sample
        would prevent efficient GPU batching, so this returns the fluid-neighbor
        masks and degree used by :meth:`_laplacian` to apply the same stencil.
        """
        neighbor_masks = [self._shift(fluid, offset) for offset in self._dirs]
        degree = sum(neighbor_masks)
        return neighbor_masks, degree

    def _laplacian(self, x, nf, deg, fluid):
        """Compact Neumann Laplacian ``sum_nbr(x_nbr - x)`` over fluid neighbors."""
        lx = -deg * x
        for offset, nb in zip(self._dirs, nf):
            lx = lx + nb * self._shift(x, offset)
        return lx * fluid

    def _divergence(self, vel, fluid):
        """Backward-difference divergence (solid neighbors contribute 0 flux)."""
        div = torch.zeros_like(fluid)
        for axis, offset in enumerate(self._back_dirs):
            v = vel[..., axis]
            div = div + (v - self._shift(v, offset))
        return div * fluid

    def _subtract_gradient(self, vel, p, nf_fwd, fluid):
        """Subtract the forward-difference pressure gradient (Neumann walls)."""
        out = vel.clone()
        for axis, (offset, nb) in enumerate(zip(self._fwd_dirs, nf_fwd)):
            p_fwd = nb * self._shift(p, offset) + (1.0 - nb) * p
            out[..., axis] = (vel[..., axis] - (p_fwd - p)) * fluid
        return out

    @staticmethod
    def _bdims(x):
        return (-1,) + (1,) * (x.dim() - 1)

    def _cg(self, matvec, b, x0, iters, diag=None):
        """Batched (optionally Jacobi-preconditioned) CG for SPD ``matvec``.

        ``diag`` is the operator diagonal (B, *spatial); passing it enables
        diagonal preconditioning, which greatly speeds the pressure solve.
        """
        x = x0.clone()
        r = b - matvec(x)
        z = r / diag if diag is not None else r
        p = z.clone()
        rz = (r * z).flatten(1).sum(1)
        bnorm = (b * b).flatten(1).sum(1).clamp_min(1e-30)
        tol2 = self.cg_tol * self.cg_tol
        for _ in range(iters):
            Ap = matvec(p)
            pAp = (p * Ap).flatten(1).sum(1).clamp_min(1e-30)
            alpha = (rz / pAp.clamp_min(1e-30)).view(self._bdims(x))
            x = x + alpha * p
            r = r - alpha * Ap
            rs = (r * r).flatten(1).sum(1)
            if torch.all(rs <= tol2 * bnorm):
                break
            z = r / diag if diag is not None else r
            rz_new = (r * z).flatten(1).sum(1)
            beta = (rz_new / rz.clamp_min(1e-30)).view(self._bdims(x))
            p = z + beta * p
            rz = rz_new
        return x

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def compute_velocity_field(self, fluid, q, sigma):
        """March Navier-Stokes to a batched steady velocity field.

        Args:
            fluid: (B, *spatial) float mask, 1 = free, 0 = solid.
            q: (B, *spatial) source/sink (zero-mean over each active component).
            sigma: (B, *spatial) Brinkman drag coefficient.

        Returns:
            Velocity tensor shaped ``(B, *spatial, ndim)`` on the solver device.
        """
        ndim = len(self._dirs[0])
        fluid = fluid.to(self.device, self.dtype)
        q = q.to(self.device, self.dtype)
        sigma = sigma.to(self.device, self.dtype)

        nf, deg = self._build_laplacian(fluid)
        nf_fwd = [self._shift(fluid, o) for o in self._fwd_dirs]

        a = self.viscosity * self.dt
        eps = self.pressure_reg

        def diffuse_op(u):
            return fluid * (
                (1.0 + self.dt * sigma) * u - a * self._laplacian(u, nf, deg, fluid)
            )

        def pressure_op(p):
            return fluid * (-self._laplacian(p, nf, deg, fluid) + eps * p)

        # Jacobi (diagonal) preconditioners; 1 on solids to avoid div-by-zero.
        solid = 1.0 - fluid
        diff_diag = fluid * (1.0 + self.dt * sigma + a * deg) + solid
        pres_diag = fluid * (deg + eps) + solid

        shape = fluid.shape
        vel = torch.zeros(shape + (ndim,), device=self.device, dtype=self.dtype)
        p_warm = torch.zeros(shape, device=self.device, dtype=self.dtype)

        for _ in range(self.num_steps):
            if self.use_advection:
                vel = self._advect(vel, fluid)
            if a > 0.0 or self.drag_gain > 0.0:
                for axis in range(ndim):
                    vel[..., axis] = self._cg(
                        diffuse_op,
                        fluid * vel[..., axis],
                        vel[..., axis].clone(),
                        self.diff_iters,
                        diff_diag,
                    )
            rhs = fluid * (q - self._divergence(vel, fluid))
            p_warm = self._cg(pressure_op, rhs, p_warm, self.proj_iters, pres_diag)
            vel = self._subtract_gradient(vel, p_warm, nf_fwd, fluid)
            vel = self._seal(vel, fluid)
            # Guard against a rare under-converged step amplifying into inf/nan.
            vel = torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)

        return vel

    def _compute_flow_field_tensor(self, velocity_field, fluid):
        """Unit-normalize a batched velocity field without leaving the device."""
        velocity = torch.as_tensor(
            velocity_field, dtype=self.dtype, device=self.device
        )
        fluid = torch.as_tensor(fluid, device=self.device).bool()
        magnitude = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        moving = fluid.unsqueeze(-1) & (magnitude > 1e-9)
        return torch.where(
            moving,
            velocity / magnitude.clamp_min(1e-30),
            torch.zeros_like(velocity),
        )

    def compute_flow_field(self, velocity_field, fluid):
        """Unit-normalize batched velocity fields, matching the CPU API stage."""
        return self._compute_flow_field_tensor(velocity_field, fluid).cpu().numpy()

    def compute_navigation_field(self, velocity_field, fluid, lo=0.5, hi=2.4):
        """Scale flow directions by per-sample min-max speed in ``[lo, hi]``.

        This is the batched GPU counterpart of the CPU Navier-Stokes method.
        Min/max values are computed independently for each sample over moving
        fluid cells; obstacles and stagnant cells remain zero.
        """
        velocity = torch.as_tensor(
            velocity_field, dtype=self.dtype, device=self.device
        )
        fluid = torch.as_tensor(fluid, device=self.device).bool()
        direction = self._compute_flow_field_tensor(velocity, fluid)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        moving = fluid & (speed > 1e-9)

        flat_speed = speed.flatten(1)
        flat_moving = moving.flatten(1)
        has_moving = flat_moving.any(dim=1)
        speed_min = flat_speed.masked_fill(~flat_moving, torch.inf).min(dim=1).values
        speed_max = flat_speed.masked_fill(~flat_moving, -torch.inf).max(dim=1).values
        speed_min = torch.where(has_moving, speed_min, torch.zeros_like(speed_min))
        speed_max = torch.where(has_moving, speed_max, torch.zeros_like(speed_max))

        view_shape = (speed.shape[0],) + (1,) * (speed.dim() - 1)
        speed_min = speed_min.view(view_shape)
        speed_max = speed_max.view(view_shape)
        speed_range = speed_max - speed_min
        normalized = (speed - speed_min) / speed_range.clamp_min(1e-30)
        scaled = lo + normalized * (hi - lo)
        scaled = torch.where(speed_range > 0.0, scaled, torch.full_like(scaled, lo))
        scaled = torch.where(moving, scaled, torch.zeros_like(scaled))
        return (direction * scaled.unsqueeze(-1)).cpu().numpy()

    def solve(self, fluid, q, sigma):
        """Compatibility alias for :meth:`compute_velocity_field`."""
        return self.compute_velocity_field(fluid, q, sigma)

    def flow_direction(self, fluid, q, sigma):
        """Compatibility wrapper returning unit-normalized steady flow."""
        velocity = self.compute_velocity_field(fluid, q, sigma)
        return self.compute_flow_field(velocity, fluid)


class NSTorch2D(_NSTorch):
    """2D batched drag solver (fields ``(B, n, n)``)."""

    _dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    _back_dirs = ((-1, 0), (0, -1))  # backward divergence: up, left
    _fwd_dirs = ((1, 0), (0, 1))  # forward gradient: down, right

    def _advect(self, vel, fluid):
        n = vel.shape[1]
        b = vel.shape[0]
        vr, vc = vel[..., 0], vel[..., 1]
        rows = torch.arange(n, device=self.device, dtype=self.dtype)
        gi, gj = torch.meshgrid(rows, rows, indexing="ij")
        r_back = gi.unsqueeze(0) - self.dt * vr
        c_back = gj.unsqueeze(0) - self.dt * vc
        y = 2.0 * r_back / (n - 1) - 1.0
        x = 2.0 * c_back / (n - 1) - 1.0
        grid = torch.stack([x, y], dim=-1)  # (B, n, n, 2)
        inp = torch.stack([vr, vc], dim=1)  # (B, 2, n, n)
        out = torch.nn.functional.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        vr_n = out[:, 0] * fluid
        vc_n = out[:, 1] * fluid
        return torch.stack([vr_n, vc_n], dim=-1)

    def _seal(self, vel, fluid):
        vr, vc = vel[..., 0].clone(), vel[..., 1].clone()
        solid = 1.0 - fluid
        vr[:, 0, :] = 0.0
        vr[:, -1, :] = 0.0
        vc[:, :, 0] = 0.0
        vc[:, :, -1] = 0.0
        east = torch.zeros_like(fluid)
        east[:, :, :-1] = solid[:, :, 1:]
        west = torch.zeros_like(fluid)
        west[:, :, 1:] = solid[:, :, :-1]
        down = torch.zeros_like(fluid)
        down[:, :-1, :] = solid[:, 1:, :]
        up = torch.zeros_like(fluid)
        up[:, 1:, :] = solid[:, :-1, :]
        vc = torch.where((east > 0) & (vc > 0), torch.zeros_like(vc), vc)
        vc = torch.where((west > 0) & (vc < 0), torch.zeros_like(vc), vc)
        vr = torch.where((down > 0) & (vr > 0), torch.zeros_like(vr), vr)
        vr = torch.where((up > 0) & (vr < 0), torch.zeros_like(vr), vr)
        return torch.stack([vr * fluid, vc * fluid], dim=-1)


class NSTorch3D(_NSTorch):
    """3D batched drag solver (fields ``(B, n, n, n)``)."""

    _dirs = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    _back_dirs = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    _fwd_dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def _advect(self, vel, fluid):
        n = vel.shape[1]
        vi, vj, vk = vel[..., 0], vel[..., 1], vel[..., 2]
        rng = torch.arange(n, device=self.device, dtype=self.dtype)
        gi, gj, gk = torch.meshgrid(rng, rng, rng, indexing="ij")
        i_back = gi.unsqueeze(0) - self.dt * vi
        j_back = gj.unsqueeze(0) - self.dt * vj
        k_back = gk.unsqueeze(0) - self.dt * vk
        z = 2.0 * i_back / (n - 1) - 1.0  # D axis
        y = 2.0 * j_back / (n - 1) - 1.0  # H axis
        x = 2.0 * k_back / (n - 1) - 1.0  # W axis
        grid = torch.stack([x, y, z], dim=-1)  # (B, n, n, n, 3)
        inp = torch.stack([vi, vj, vk], dim=1)  # (B, 3, n, n, n)
        out = torch.nn.functional.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return torch.stack([out[:, c] * fluid for c in range(3)], dim=-1)

    def _seal(self, vel, fluid):
        vi, vj, vk = (vel[..., c].clone() for c in range(3))
        solid = 1.0 - fluid
        vi[:, 0, :, :] = 0.0
        vi[:, -1, :, :] = 0.0
        vj[:, :, 0, :] = 0.0
        vj[:, :, -1, :] = 0.0
        vk[:, :, :, 0] = 0.0
        vk[:, :, :, -1] = 0.0

        def face(shifted_axis, lo):
            m = torch.zeros_like(fluid)
            if shifted_axis == 0 and not lo:
                m[:, :-1, :, :] = solid[:, 1:, :, :]
            elif shifted_axis == 0 and lo:
                m[:, 1:, :, :] = solid[:, :-1, :, :]
            elif shifted_axis == 1 and not lo:
                m[:, :, :-1, :] = solid[:, :, 1:, :]
            elif shifted_axis == 1 and lo:
                m[:, :, 1:, :] = solid[:, :, :-1, :]
            elif shifted_axis == 2 and not lo:
                m[:, :, :, :-1] = solid[:, :, :, 1:]
            else:
                m[:, :, :, 1:] = solid[:, :, :, :-1]
            return m > 0

        vi = torch.where(face(0, False) & (vi > 0), torch.zeros_like(vi), vi)
        vi = torch.where(face(0, True) & (vi < 0), torch.zeros_like(vi), vi)
        vj = torch.where(face(1, False) & (vj > 0), torch.zeros_like(vj), vj)
        vj = torch.where(face(1, True) & (vj < 0), torch.zeros_like(vj), vj)
        vk = torch.where(face(2, False) & (vk > 0), torch.zeros_like(vk), vk)
        vk = torch.where(face(2, True) & (vk < 0), torch.zeros_like(vk), vk)
        return torch.stack([vi * fluid, vj * fluid, vk * fluid], dim=-1)


# ----------------------------------------------------------------------
# CPU preprocessing + batched target construction
# ----------------------------------------------------------------------
def _structure(ndim: int) -> np.ndarray:
    """Face-connectivity structuring element (4-conn in 2D, 6-conn in 3D)."""
    st = np.zeros((3,) * ndim, dtype=bool)
    center = (1,) * ndim
    st[center] = True
    for axis in range(ndim):
        for off in (-1, 1):
            idx = list(center)
            idx[axis] += off
            st[tuple(idx)] = True
    return st


def _prep_sample(grid, goal, source_rate, drag_gain, drag_influence):
    """Per-sample source/sink ``q`` and drag ``sigma`` (matches the numpy solver)."""
    fluid = grid == 0
    ndim = grid.ndim
    lbl, _ = label(fluid, structure=_structure(ndim))
    goal_comp = lbl[goal]
    comp = lbl == goal_comp
    q = np.zeros(grid.shape, dtype=np.float32)
    q[comp] = source_rate
    q[goal] -= float(q.sum())  # zero-mean sink at the goal (only outlet)
    sigma = np.zeros(grid.shape, dtype=np.float32)
    if drag_gain > 0.0:
        dist = distance_transform_edt(fluid)
        sigma = (
            drag_gain * np.clip(1.0 - dist / drag_influence, 0.0, 1.0) ** 2
        ).astype(np.float32)
    return fluid.astype(np.float32), q, sigma


def build_targets(
    grids, goals, ndim=2, batch_size=64, device=None, progress=False, fill_stagnant=True, **solver_kwargs
):
    """Batched drag-flow targets for many ``(grid, goal)`` samples.

    Args:
        grids: sequence of (n, ...) int grids (0=free, 1=obstacle), all same size.
        goals: sequence of goal index tuples, one per grid.
        ndim: 2 or 3.
        batch_size: number of grids solved together on the GPU.
        device: torch device (defaults to cuda if available).
        progress: when True, print after each GPU sub-batch is solved.
        fill_stagnant: when True, inpaint zero-flow cells via GPU fill before zeroing goal.
        **solver_kwargs: forwarded to :class:`NSTorch2D` / :class:`NSTorch3D`
            (``viscosity``, ``dt``, ``source_rate``, ``num_steps``, ``drag_gain``,
            ``drag_influence``, ``diff_iters``, ``proj_iters``, ``cg_tol``).

    Returns:
        (K, n, ..., ndim) float32 navigation fields whose raw velocity magnitudes
        are min-max mapped to [0.8, 2.4], with the goal cell zeroed.
    """
    solver_cls = NSTorch2D if ndim == 2 else NSTorch3D
    solver = solver_cls(device=device, **solver_kwargs)
    source_rate = solver.source_rate
    drag_gain = solver.drag_gain
    drag_influence = solver.drag_influence

    total = len(grids)
    out = []
    for start in range(0, total, batch_size):
        gb = grids[start : start + batch_size]
        ob = goals[start : start + batch_size]
        fluids, qs, sigmas = [], [], []
        for grid, goal in zip(gb, ob):
            f, q, s = _prep_sample(
                np.asarray(grid), tuple(goal), source_rate, drag_gain, drag_influence
            )
            fluids.append(f)
            qs.append(q)
            sigmas.append(s)
        fluid_t = torch.from_numpy(np.stack(fluids))
        q_t = torch.from_numpy(np.stack(qs))
        sigma_t = torch.from_numpy(np.stack(sigmas))
        velocity = solver.compute_velocity_field(fluid_t, q_t, sigma_t)
        field = solver.compute_navigation_field(
            velocity, fluid_t, lo=0.8, hi=2.4
        )
        
        if fill_stagnant:
            grids_batch = np.stack(gb).astype(np.float32)
            field = fill_stagnant_flow_torch(field, grids_batch, iters=4, normalize=False, device=device)
        
        for i, goal in enumerate(ob):
            field[i][tuple(goal)] = 0.0
        out.append(field.astype(np.float32))
        if progress:
            done = min(start + batch_size, total)
            print(f"    GPU solved {done}/{total} targets...", flush=True)
    return np.concatenate(out, axis=0)


def _sample_grids(
    num_samples,
    grid_size,
    obstacle_prob,
    ndim,
    seed,
    abs_seed,
    check_reachable,
    max_attempts,
    first_goal,
):
    """Collect valid, reachable ``(grid, goal)`` pairs (same rules as the datasets)."""
    if ndim == 2:
        from gridflow.algorithms import AStar, BreadthFirstSearch

        bfs = BreadthFirstSearch()
        astar = AStar()
        reachable = lambda g, c, go: bool(astar.compute_path(g, c, go))
    else:
        from gridflow.algorithms_3d import BreadthFirstSearch3D

        bfs = BreadthFirstSearch3D()
        reachable = lambda g, c, go: bfs.is_reachable(g, c, go)

    rng = np.random.default_rng(seed)
    center = (grid_size // 2,) * ndim
    if max_attempts is None:
        max_attempts = max(1000, 50 * num_samples)

    grids, goals = [], []
    attempt = 0
    while len(grids) < num_samples and attempt < max_attempts:
        grid = bfs.generate_sparse_grid(
            n=grid_size,
            obstacle_prob=obstacle_prob,
            seed=attempt if abs_seed is None else abs_seed,
        )
        goal = _random_free_cell(grid, rng, ndim)
        if abs_seed is not None and len(grids) == 0 and first_goal is not None:
            goal = first_goal
        attempt += 1

        if grid[center] != 0 or grid[goal] != 0:
            continue
        if check_reachable and not reachable(grid, center, goal):
            continue

        grids.append(grid)
        goals.append(goal)
    return grids, goals


def _random_free_cell(grid, rng, ndim):
    n = len(grid)
    while True:
        cell = tuple(int(rng.integers(0, n)) for _ in range(ndim))
        if grid[cell] == 0:
            return cell


def generate_flow_dataset(
    num_samples,
    grid_size,
    obstacle_prob,
    ndim=2,
    seed=0,
    abs_seed=None,
    check_reachable=True,
    batch_size=256,
    device=None,
    max_attempts=None,
    first_goal=None,
    save_path=None,
    **solver_kwargs,
):
    """Generate a full drag-flow dataset on the GPU (grids, goals, targets).

    Samples valid, reachable ``(grid, goal)`` pairs with the same rules as
    :class:`FlowFieldDataset` / :class:`FlowFieldDataset3D`, then computes all
    ground-truth targets in GPU batches with :func:`build_targets`.

    Args:
        num_samples, grid_size, obstacle_prob, seed, abs_seed, check_reachable,
            max_attempts: as in the dataset classes.
        ndim: 2 or 3.
        batch_size: grids solved together on the GPU.
        device: torch device (defaults to cuda if available).
        first_goal: optional fixed goal for the first sample (abs_seed runs).
        save_path: optional ``.npz`` path; saved in the
            :meth:`FlowFieldDataset.load` layout (grids/goals/targets/grid_size).
        **solver_kwargs: forwarded to the solver (viscosity, dt, source_rate,
            num_steps, drag_gain, drag_influence, diff_iters, proj_iters, cg_tol).

    Returns:
        (grids, goals, targets) numpy arrays.
    """
    grids, goals = _sample_grids(
        num_samples,
        grid_size,
        obstacle_prob,
        ndim,
        seed,
        abs_seed,
        check_reachable,
        max_attempts,
        first_goal,
    )
    if len(grids) < num_samples:
        print(
            f"Warning: collected only {len(grids)} / {num_samples} samples "
            f"(obstacle_prob={obstacle_prob} may be too high)."
        )
    targets = build_targets(
        grids, goals, ndim=ndim, batch_size=batch_size, device=device, **solver_kwargs
    )
    grids = np.stack(grids).astype(np.int8)
    goals = np.array(goals, dtype=np.int64)

    if save_path is not None:
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path, grids=grids, goals=goals, targets=targets, grid_size=grid_size
        )
    return grids, goals, targets


def fill_stagnant_flow_torch(field, grid, iters=4, normalize=False, device=None):
    """Fill zero-flow free cells with mean of moving neighbors' vectors (batched torch).

    GPU-accelerated version of :func:`gridflow.utils.grid.fill_stagnant_flow`. Accepts
    batched inputs and returns the same shape. Incompressible flow has genuine
    stagnation points (domain corners, pocket interiors); this inpaints them by
    averaging moving neighbors' vectors, repeated ``iters`` passes to reach into
    2-3 cell pockets. Obstacles and fully isolated pockets are left at zero.

    Works for 2D ``(B, n, n, 2)`` and 3D ``(B, n, n, n, 3)`` field tensors.

    Args:
        field: (B, *spatial, D) flow/direction field (torch tensor or numpy array).
        grid: (B, *spatial) or (*spatial) obstacle grid (0=free, non-zero=obstacle).
            If (*spatial) without batch, broadcasted to all B samples.
        iters: Number of propagation passes.
        normalize: If True, renormalize filled cells to unit length (for unit-direction
            fields); if False, keep averaged magnitude (for magnitude-carrying fields).
        device: torch device; if None, infers from field or defaults to cuda/cpu.

    Returns:
        (B, *spatial, D) float32 numpy array with stagnation holes filled.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Convert to torch, ensure float32, move to device
    field = torch.as_tensor(field, dtype=torch.float32, device=device)
    grid = torch.as_tensor(grid, dtype=torch.float32, device=device)

    # Handle grid broadcasting: if grid lacks batch dimension, add it
    # field is (B, *spatial, D), so field.shape[:-1] is (B, *spatial)
    # If grid is (*spatial) only, it has one fewer dimension than field.shape[:-1]
    if grid.dim() < len(field.shape[:-1]):
        grid = grid.unsqueeze(0).expand(field.shape[:-1])
    
    ndim = field.shape[-1]
    spatial_ndim = field.dim() - 2  # e.g., 2 for (B, n, n, 2)
    free = grid == 0.0

    # Build 3^D neighborhood kernel (center excluded)
    kernel_shape = (3,) * spatial_ndim
    kernel = torch.ones(kernel_shape, dtype=torch.float32, device=device)
    center_idx = tuple(1 for _ in range(spatial_ndim))
    kernel[center_idx] = 0.0

    for iteration in range(iters):
        mag = torch.linalg.vector_norm(field, dim=-1)  # (B, *spatial)
        moving = free & (mag > 1e-9)
        stagnant = free & ~moving
        free_f = free.to(torch.float32)
        moving_f = moving.to(torch.float32)

        # Convolve to count moving neighbors: sum of moving_f in each 3^D neighborhood
        if spatial_ndim == 2:
            den = torch.nn.functional.conv_transpose2d(
                moving_f.unsqueeze(1),  # (B, 1, *spatial)
                kernel.unsqueeze(0).unsqueeze(0),  # (1, 1, 3, 3)
                padding=1,
            ).squeeze(1)  # (B, *spatial)
        else:  # 3D
            den = torch.nn.functional.conv_transpose3d(
                moving_f.unsqueeze(1),
                kernel.unsqueeze(0).unsqueeze(0),  # (1, 1, 3, 3, 3)
                padding=1,
            ).squeeze(1)

        fillable = stagnant & (den > 0.5)
        if not fillable.any():
            break

        out = field.clone()
        for c in range(ndim):
            # Convolve each component separately
            if spatial_ndim == 2:
                contrib = torch.nn.functional.conv_transpose2d(
                    (field[..., c] * moving_f).unsqueeze(1),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=1,
                ).squeeze(1)
            else:
                contrib = torch.nn.functional.conv_transpose3d(
                    (field[..., c] * moving_f).unsqueeze(1),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=1,
                ).squeeze(1)
            avg_c = contrib / torch.where(den > 0, den, torch.ones_like(den))
            out[..., c] = torch.where(fillable, avg_c, field[..., c])

        if normalize:
            nm = torch.linalg.vector_norm(out, dim=-1)
            renorm = fillable & (nm > 1e-9)
            for c in range(ndim):
                out[..., c] = torch.where(
                    renorm,
                    out[..., c] / torch.where(nm > 1e-9, nm, torch.ones_like(nm)),
                    out[..., c],
                )
        field = out

    return field.cpu().numpy().astype(np.float32)
