"""gridflow_nn - Neural vector-field model for the gridflow project.

This package holds the torch-based model definition so it can be imported
cleanly (``from gridflow_nn import VectorFieldNet``) from the training scripts
and the simulation, without any ``sys.path`` hacks. It is kept separate from
the ``gridflow`` core package, which stays torch-free (numpy + matplotlib only).
"""

from .vector_field_net import VectorFieldNet, ConvBlock, QueryHead
from .vector_field_net_3d import VectorFieldNet3D, ConvBlock3D
from .inference import load_model, predict_full_field, predict_query_field
from .inference_3d import load_model_3d, predict_full_field_3d

__all__ = [
    "VectorFieldNet",
    "ConvBlock",
    "QueryHead",
    "VectorFieldNet3D",
    "ConvBlock3D",
    "load_model",
    "predict_full_field",
    "predict_query_field",
    "load_model_3d",
    "predict_full_field_3d",
]
