"""Train the neural vector-field model, then visualize predictions.

Set ``config.dataset`` to ``"2d"`` or ``"3d"`` to choose which dataset/model to
train. The visualization branches on the same key.

Run from the repo root:
    python scripts/run_trainer.py
"""

import os
import datetime

import torch

from gridflow_nn.training import Trainer, Config, plot_prediction_vs_target
from gridflow.utils import plot_vector_field_3d


def _visualize_2d(trainer, model, val_set, grid_size, device, num_examples=3):
    for idx in list(val_set.indices)[:num_examples]:
        grid, goal, target = val_set.dataset.samples[idx]

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (grid_size - 1)

        pred_field, pred_positions = trainer.predict_full_field(
            model, grid_t, goal_t, grid_size, device, step=0.25
        )
        plot_prediction_vs_target(
            pred_field,
            target,
            grid,
            goal=goal,
            suptitle=f"Val sample {idx}",
            pred_positions=pred_positions,
        )


def _visualize_3d(trainer, model, val_set, grid_size, device, num_examples=2):
    for idx in list(val_set.indices)[:num_examples]:
        grid, goal, target = val_set.dataset.get_raw(idx)

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (grid_size - 1)

        pred_field, _ = trainer.predict_full_field(
            model, grid_t, goal_t, grid_size, device, step=1.0
        )
        plot_vector_field_3d(
            pred_field,
            grid,
            source=goal,
            fast_obstacles=True,
            title=f"Val {idx}: prediction",
        )
        plot_vector_field_3d(
            target,
            grid,
            source=goal,
            fast_obstacles=True,
            title=f"Val {idx}: ground truth",
        )


def main():
    config = Config()
    config.dataset = "3d"

    os.makedirs("models", exist_ok=True)
    model_name = f"models/vfn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    trainer = Trainer(config=config, save_path=model_name)
    model, val_set, device = trainer.train()

    grid_size = val_set.dataset.grid_size

    if trainer.dataset_key == "3d":
        _visualize_3d(trainer, model, val_set, grid_size, device)
    else:
        _visualize_2d(trainer, model, val_set, grid_size, device)


if __name__ == "__main__":
    main()
