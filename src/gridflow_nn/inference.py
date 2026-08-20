"""Inference helpers for the neural vector-field model.

This module is intentionally torch-only (no matplotlib, no dataloader, no
training loop) so a consumer that just wants to run a trained model can do::

    from gridflow_nn import load_model, predict_full_field

without pulling in the training/visualization dependencies.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .vector_field_net import VectorFieldNet
from .integration import integrate_path


def load_model(ckpt_path, device=None):
    """Rebuild a trained VectorFieldNet from a checkpoint and load its weights.

    Args:
        ckpt_path: path to a ``.pt`` checkpoint saved by the trainer. The
            checkpoint must contain ``model_state_dict`` and a ``config`` dict
            (with at least ``model_channels`` and ``grid_size``).
        device: torch device (or string). Defaults to CUDA when available.

    Returns:
        (model, config): the loaded, eval-mode model and its stored config dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    model = VectorFieldNet(
        base_channels=config.get("model_channels", 32),
        fourier_bands=config.get("query_fourier_bands", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    return model, config


def predict_full_field(
    model,
    grid_t: torch.Tensor,
    goal_t: torch.Tensor,
    grid_size: int,
    device: torch.device,
    step: float = 1.0,
):
    """Predict the flow vector on a (possibly sub-cell) grid of query points.

    Args:
        model: trained VectorFieldNet
        grid_t: (1, n, n) obstacle grid tensor
        goal_t: (2,) normalized goal coordinates
        grid_size: n
        device: torch device
        step: spacing between query points in cell units. 1.0 queries integer
            cells (0, 1, ..., n-1); 0.5 queries half-cell offsets, etc.

    Returns:
        field: (R, R, 2) numpy array of predicted vectors, where
            R = len(positions).
        positions: (R,) numpy array of the query coordinates (in cell units)
            along each axis, e.g. [0, 0.5, 1.0, ...].
    """
    n = grid_size
    model.eval()

    grid_b = grid_t.unsqueeze(0).to(device)  # (1, 1, n, n)
    goal_b = goal_t.unsqueeze(0).to(device)  # (1, 2)

    with torch.no_grad():
        dense = model(grid_b, goal_b)  # (1, n, n, 2)

        if step == 1.0:
            positions = np.arange(n)
            return dense.squeeze(0).cpu().numpy(), positions

        positions = np.arange(0, n - 1 + 1e-9, step)
        res = len(positions)
        coords = [[r, c] for r in positions for c in positions]
        query = torch.tensor(coords, dtype=torch.float32) / (n - 1)
        query = query.to(device)

        # Bilinearly resample the dense field at the sub-cell query points.
        dense_chw = dense.permute(0, 3, 1, 2)  # (1, 2, n, n)
        sample_grid = query[..., [1, 0]] * 2.0 - 1.0  # (R*R, 2) as (col, row)
        sample_grid = sample_grid.view(1, res * res, 1, 2)
        sampled = F.grid_sample(
            dense_chw,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, 2, R*R, 1)
        pred = sampled.squeeze(-1).squeeze(0).transpose(0, 1)  # (R*R, 2)

    return pred.reshape(res, res, 2).cpu().numpy(), positions


def predict_query_field(
    model: VectorFieldNet,
    grid_t: torch.Tensor,
    goal_t: torch.Tensor,
    grid_size: int,
    device: torch.device,
    positions,
):
    """Predict the flow vector(s) at continuous query location(s) via the model.

    Args:
        model: trained VectorFieldNet (with a query head).
        grid_t: (1, n, n) obstacle grid tensor.
        goal_t: (2,) normalized goal coordinates.
        grid_size: n (kept for signature parity with predict_full_field).
        device: torch device.
        positions: (2,) or (M, 2) array of (row, col) query coordinates in cell
            units (0 .. n-1); fractional values are allowed.

    Returns:
        A (2,) vector when a single position is given, else an (M, 2) array.
    """
    model.eval()

    pos = np.asarray(positions, dtype=np.float32)
    single = pos.ndim == 1
    pos = np.atleast_2d(pos)  # (M, 2)

    grid_b = grid_t.unsqueeze(0).to(device)  # (1, 1, n, n)
    goal_b = goal_t.unsqueeze(0).to(device)  # (1, 2)
    queries = torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        vecs = model.forward_query(grid_b, goal_b, queries)  # (1, M, 2)

    vecs = vecs.squeeze(0).cpu().numpy()
    return vecs[0] if single else vecs


def predict_path(
    model: VectorFieldNet,
    grid_t: torch.Tensor,
    goal_t: torch.Tensor,
    grid_size: int,
    device: torch.device,
    start,
    goal=None,
    method: str = "rk4",
    step_size: float = 0.5,
    max_steps: int = 200,
    stop_radius: float = 1.0,
    downsample_factor: int = 1,
):
    """Trace a path by following the model's flow field from ``start`` to ``goal``.

    Args:
        model, grid_t, goal_t, grid_size, device: as for ``predict_query_field``.
        start: (2,) start point in full-resolution cell units.
        goal: (2,) goal in full-resolution cell units for the stop test.
            Defaults to the goal implied by ``goal_t`` and the grid resolution.
        method: "rk4" or "euler".
        step_size, stop_radius: in the (downsampled) grid's cell units.
        max_steps: iteration cap.
        downsample_factor: run the model on a coarsened ``[::f, ::f]`` grid
            (>= 1) for speed. ``start``/``goal`` and the returned path stay in
            full-resolution cell units; only the resolution the model sees changes.

    Returns:
        (M, 2) numpy array of trajectory points in full-resolution cell units.
    """
    df = max(1, int(downsample_factor))
    full_spatial = np.array(grid_t.shape[-2:], dtype=float)  # full-res (H, W)
    if df > 1:
        grid_t = grid_t[..., ::df, ::df]  # coarse obstacle map

    spatial = np.array(grid_t.shape[-2:], dtype=float)  # (H, W) of the grid used

    def _is_inside(p, size):
        p = np.asarray(p, dtype=float)
        return np.all((p >= 0) & (p < size))

    if goal is None or not _is_inside(goal, full_spatial):
        goal = goal_t.detach().cpu().numpy() * (spatial - 1.0)  # coarse cell coords
    else:
        goal = np.asarray(goal, dtype=float) / df

    start = np.asarray(start, dtype=float) / df

    # encode the (coarse) grid once; every query along the path reuses this feature
    grid_b = grid_t.unsqueeze(0).to(device)  # (1, 1, H, W)
    goal_b = goal_t.unsqueeze(0).to(device)  # (1, 2)
    model.eval()
    with torch.no_grad():
        feat = model.encode(grid_b, goal_b)

    def field_fn(p):
        q = torch.tensor(
            np.atleast_2d(p), dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, 1, 2)
        with torch.no_grad():
            vecs = model.query(feat, q, goal_b)  # (1, 1, 2)
        return vecs[0, 0].cpu().numpy()  # (2,)

    path = integrate_path(
        field_fn,
        start,
        goal,
        method=method,
        step_size=step_size,
        max_steps=max_steps,
        stop_radius=stop_radius,
    )
    return path * df  # coarse -> full-resolution cell units
