"""gridflow_nn - Neural vector-field model for the gridflow project.

This package holds the torch-based model definition so it can be imported
cleanly (``from gridflow_nn import VectorFieldNet``) from the training scripts
and the simulation, without any ``sys.path`` hacks. It is kept separate from
the ``gridflow`` core package, which stays torch-free (numpy + matplotlib only).
"""

from .vector_field_net import VectorFieldNet, ConvBlock, GlobalAttentionBlock, QueryHead
from .vector_field_net_3d import (
    VectorFieldNet3D,
    ConvBlock3D,
    GlobalAttentionBlock3D,
    QueryHead3D,
)
from .inference import load_model, predict_full_field, predict_query_field, predict_path
from .inference_3d import (
    load_model_3d,
    predict_full_field_3d,
    predict_query_field_3d,
    predict_path_3d,
)
from .integration import eul_integrate, rk4_integrate, integrate_path
from .ns_torch import NSTorch2D, NSTorch3D, build_targets, generate_flow_dataset

__all__ = [
    "VectorFieldNet",
    "ConvBlock",
    "GlobalAttentionBlock",
    "QueryHead",
    "VectorFieldNet3D",
    "ConvBlock3D",
    "GlobalAttentionBlock3D",
    "QueryHead3D",
    "load_model",
    "predict_full_field",
    "predict_query_field",
    "predict_path",
    "load_model_3d",
    "predict_full_field_3d",
    "predict_query_field_3d",
    "predict_path_3d",
    "eul_integrate",
    "rk4_integrate",
    "integrate_path",
    "NSTorch2D",
    "NSTorch3D",
    "build_targets",
    "generate_flow_dataset",
]
