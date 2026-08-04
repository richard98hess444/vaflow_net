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
    predict_full_field,
    predict_full_field_3d,
)
from gridflow_nn.training import (
    FlowFieldDataset,
    FlowFieldDataset3D,
)
from gridflow.utils import plot_prediction_vs_target, plot_prediction_vs_target_3d


def _raw_sample(dataset, idx):
    getter = getattr(dataset, "get_raw", None)
    if getter is not None:
        return getter(idx)
    return dataset.samples[idx]


def visualize_prediction_2d(
    model,
    dataset,
    indices,
    grid_size,
    device,
    step=0.5,
    label="Sample",
):
    for idx in indices:
        grid, goal, target = _raw_sample(dataset, idx)

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (grid_size - 1)

        pred_field, pred_positions = predict_full_field(
            model, grid_t, goal_t, grid_size, device, step=step
        )
        plot_prediction_vs_target(
            pred_field,
            target,
            grid,
            goal=goal,
            suptitle=f"{label} {idx}",
            pred_positions=pred_positions,
        )


def visualize_prediction_3d(
    model,
    dataset,
    indices,
    grid_size,
    device,
    step=1.0,
    stride=1,
    label="Sample",
):
    for idx in indices:
        grid, goal, target = _raw_sample(dataset, idx)

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (grid_size - 1)

        pred_field, _ = predict_full_field_3d(
            model, grid_t, goal_t, grid_size, device, step=step
        )
        plot_prediction_vs_target_3d(
            pred_field,
            target,
            grid,
            source=goal,
            stride=stride,
            fast_obstacles=True,
            suptitle=f"{label} {idx}",
        )


def _latest_checkpoint():
    folders = [d for d in glob.glob("models/*") if os.path.isdir(d)]
    if not folders:
        raise FileNotFoundError(
            "No model folder found under models/. Train a model first "
            "(python scripts/run_trainer.py)."
        )
    latest_folder = max(folders, key=os.path.getmtime)

    checkpoints = glob.glob(os.path.join(latest_folder, "*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoint found in {latest_folder}. Train a model first "
            "(python scripts/run_trainer.py)."
        )
    return max(checkpoints, key=os.path.getmtime)


def load_model(ckpt_path, device):
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
    visualize_prediction_2d(
        model, dataset, range(len(dataset)), grid_size, device, step=1.0
    )


def _visualize_3d(model, dataset, grid_size, device):
    visualize_prediction_3d(
        model, dataset, range(len(dataset)), grid_size, device, step=1.0, stride=4
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
