"""Load a trained neural vector-field model and visualize its predictions.

Loads a saved checkpoint (.pt), generates a few fresh samples, and plots the
predicted flow field against the ground truth. Works for both 2D and 3D
checkpoints: the ``dataset`` key stored in the checkpoint config selects the
matching model, dataset and visualization.

Run from the repo root:
    python scripts/run_visualizer.py
    python scripts/run_visualizer.py path/to/model.pt
"""

import os
import sys
import glob

import torch

from gridflow_nn import (
    VectorFieldNet,
    VectorFieldNet3D,
    predict_full_field_3d,
)
from gridflow_nn.training import (
    Trainer,
    FlowFieldDataset,
    FlowFieldDataset3D,
    plot_prediction_vs_target,
)
from gridflow.utils import plot_vector_field_3d


def _latest_checkpoint():
    """Return the most recently modified .pt file under models/."""
    checkpoints = glob.glob("models/*.pt") + glob.glob("*.pt")
    if not checkpoints:
        raise FileNotFoundError(
            "No .pt checkpoint found. Train a model first "
            "(python scripts/run_trainer.py)."
        )
    return max(checkpoints, key=os.path.getmtime)


def load_model(ckpt_path, device):
    """Rebuild a saved model from a checkpoint and load its weights.

    Chooses ``VectorFieldNet`` or ``VectorFieldNet3D`` based on the ``dataset``
    key stored in the checkpoint config ("2d" by default for older 2D models).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    dataset_key = config.get("dataset", "2d")

    net_cls = VectorFieldNet3D if dataset_key == "3d" else VectorFieldNet
    model = net_cls(
        base_channels=config.get("model_channels", 32),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config, dataset_key


def _visualize_2d(model, dataset, grid_size, device):
    """Plot prediction vs ground truth for each 2D sample."""
    for idx in range(len(dataset)):
        grid, goal, target = dataset.samples[idx]

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (grid_size - 1)

        # step=1.0 -> integer cells; use 0.5 (or smaller) for sub-cell resolution.
        pred_field, pred_positions = Trainer.predict_full_field(
            model, grid_t, goal_t, grid_size, device, step=0.5
        )

        plot_prediction_vs_target(
            pred_field,
            target,
            grid,
            goal=goal,
            suptitle=f"Sample {idx}",
            pred_positions=pred_positions,
        )


def _visualize_3d(model, dataset, grid_size, device):
    """Plot prediction and ground truth (separately) for each 3D sample."""
    for idx in range(len(dataset)):
        grid, goal, target = dataset.samples[idx]

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (grid_size - 1)

        pred_field, _ = predict_full_field_3d(
            model, grid_t, goal_t, grid_size, device, step=1.0
        )

        plot_vector_field_3d(
            pred_field,
            grid,
            source=goal,
            fast_obstacles=True,
            stride=4,
            title=f"Sample {idx} - Prediction",
        )
        plot_vector_field_3d(
            target,
            grid,
            source=goal,
            fast_obstacles=True,
            stride=4,
            title=f"Sample {idx} - Ground truth",
        )


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else _latest_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {ckpt_path}")

    model, config, dataset_key = load_model(ckpt_path, device)
    grid_size = config["grid_size"]
    print(f"Detected {dataset_key.upper()} model (grid_size={grid_size}).")

    num_examples = 3
    dataset_cls = FlowFieldDataset3D if dataset_key == "3d" else FlowFieldDataset
    dataset = dataset_cls(
        num_samples=num_examples,
        grid_size=grid_size,
        obstacle_prob=0.2,
        seed=1,
        abs_seed=1,
    )

    if dataset_key == "3d":
        _visualize_3d(model, dataset, grid_size, device)
    else:
        _visualize_2d(model, dataset, grid_size, device)


if __name__ == "__main__":
    main()
