"""gridflow_nn - Neural vector-field model for the gridflow project.

This package holds the torch-based model definition so it can be imported
cleanly (``from gridflow_nn import VectorFieldNet``) from the training scripts
and the simulation, without any ``sys.path`` hacks. It is kept separate from
the ``gridflow`` core package, which stays torch-free (numpy + matplotlib only).
"""

from .vector_field_net import VectorFieldNet, ConvBlock
from .inference import load_model, predict_full_field

__all__ = ["VectorFieldNet", "ConvBlock", "load_model", "predict_full_field"]
