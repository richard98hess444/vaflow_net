"""Inference helpers for the 3D neural vector-field model.

The 3D counterpart of :mod:`gridflow_nn.inference`. Intentionally torch-only so
a consumer that just wants to run a trained 3D model can do::

    from gridflow_nn import load_model_3d, predict_full_field_3d

without pulling in the training/visualization dependencies.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .vector_field_net_3d import VectorFieldNet3D


def load_model_3d(ckpt_path, device=None):
    """Rebuild a trained VectorFieldNet3D from a checkpoint and load its weights.

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

    model = VectorFieldNet3D(
        base_channels=config.get("model_channels", 32),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config


def predict_full_field_3d(
    model,
    grid_t: torch.Tensor,
    goal_t: torch.Tensor,
    grid_size: int,
    device: torch.device,
    step: float = 1.0,
):
    """Predict the flow vector on a (possibly sub-voxel) grid of query points.

    Args:
        model: trained VectorFieldNet3D
        grid_t: (1, n, n, n) obstacle grid tensor
        goal_t: (3,) normalized goal coordinates
        grid_size: n
        device: torch device
        step: spacing between query points in voxel units. 1.0 queries integer
            voxels (0, 1, ..., n-1); 0.5 queries half-voxel offsets, etc.

    Returns:
        field: (R, R, R, 3) numpy array of predicted vectors, where
            R = len(positions).
        positions: (R,) numpy array of the query coordinates (in voxel units)
            along each axis, e.g. [0, 0.5, 1.0, ...].
    """
    n = grid_size
    model.eval()

    grid_b = grid_t.unsqueeze(0).to(device)  # (1, 1, n, n, n)
    goal_b = goal_t.unsqueeze(0).to(device)  # (1, 3)

    with torch.no_grad():
        dense = model(grid_b, goal_b)  # (1, n, n, n, 3)

        if step == 1.0:
            positions = np.arange(n)
            return dense.squeeze(0).cpu().numpy(), positions

        positions = np.arange(0, n - 1 + 1e-9, step)
        res = len(positions)
        coords = [
            [i, j, k] for i in positions for j in positions for k in positions
        ]
        query = torch.tensor(coords, dtype=torch.float32) / (n - 1)
        query = query.to(device)

        # Trilinearly resample the dense field at the sub-voxel query points.
        dense_cdhw = dense.permute(0, 4, 1, 2, 3)  # (1, 3, n, n, n)
        # grid_sample expects (x, y, z) = (width, height, depth) = (k, j, i).
        sample_grid = query[..., [2, 1, 0]] * 2.0 - 1.0  # (R^3, 3) as (k, j, i)
        sample_grid = sample_grid.view(1, res * res * res, 1, 1, 3)
        sampled = F.grid_sample(
            dense_cdhw,
            sample_grid,
            mode="bilinear",  # trilinear for 5D input
            padding_mode="border",
            align_corners=True,
        )  # (1, 3, R^3, 1, 1)
        pred = sampled.view(3, res * res * res).transpose(0, 1)  # (R^3, 3)

    return pred.reshape(res, res, res, 3).cpu().numpy(), positions
