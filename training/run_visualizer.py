"""Load a trained neural vector-field model and visualize its predictions.

Loads a saved checkpoint (.pt), generates a few fresh WaterFlow samples, and
plots the predicted flow field against the ground truth.

Run from the repo root:
    python training/run_visualizer.py
    python training/run_visualizer.py path/to/model.pt
"""

import os
import sys
import glob

import torch

from trainer import Trainer
from modules import VectorFieldNet
from dataloader import FlowFieldDataset
from visualizer import plot_prediction_vs_target


def _latest_checkpoint():
    """Return the most recently modified .pt file under models/."""
    checkpoints = glob.glob("models/*.pt") + glob.glob("*.pt")
    if not checkpoints:
        raise FileNotFoundError(
            "No .pt checkpoint found. Train a model first "
            "(python training/run_train_vfn.py)."
        )
    return max(checkpoints, key=os.path.getmtime)


def load_model(ckpt_path, device):
    """Rebuild a VectorFieldNet from a checkpoint and load its weights."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]

    model = VectorFieldNet(
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, config


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else _latest_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {ckpt_path}")

    model, config = load_model(ckpt_path, device)
    grid_size = config["grid_size"]

    # Generate fresh samples to visualize predictions on.
    num_examples = 3
    dataset = FlowFieldDataset(
        num_samples=num_examples,
        grid_size=grid_size,
        obstacle_prob=0.2,
        threshold=0.05,
        seed=1,  # different seed -> grids the model didn't train on
        abs_seed=1,  # ensure reproducibility of the random grids
    )

    for idx in range(len(dataset)):
        grid, goal, target = dataset.samples[idx]

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (grid_size - 1)

        # step=1.0 -> integer cells; use 0.5 (or smaller) for sub-cell resolution.
        pred_field, pred_positions = Trainer.predict_full_field(
            model, grid_t, goal_t, grid_size, device, step=0.25
        )

        plot_prediction_vs_target(
            pred_field,
            target,
            grid,
            goal=goal,
            suptitle=f"Sample {idx}",
            pred_positions=pred_positions,
        )


if __name__ == "__main__":
    main()
