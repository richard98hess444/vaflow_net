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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
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
