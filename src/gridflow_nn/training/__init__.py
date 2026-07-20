"""Training subpackage for the neural vector-field model.

This is the *heavy* tier (torch + matplotlib + dataloader). It is deliberately
NOT imported by ``gridflow_nn/__init__`` so that lightweight consumers can do
``from gridflow_nn import load_model, predict_full_field`` without pulling in
matplotlib or the training loop.
"""

from .trainer import Trainer
from .configs import Config
from .dataloader import FlowFieldDataset
from gridflow.utils import plot_prediction_vs_target

__all__ = [
    "Trainer",
    "Config",
    "FlowFieldDataset",
    "plot_prediction_vs_target",
]
