"""Train the neural vector-field model, then visualize predictions.

Run from the repo root:
    python training/run_train_vfn.py
"""

import os
import datetime

import torch

from gridflow_nn.training import Trainer, Config, plot_prediction_vs_target


def main():
    config = Config()
    os.makedirs("models", exist_ok=True)
    model_name = f"models/vfn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    trainer = Trainer(config=config, save_path=model_name)
    model, val_set, device = trainer.train()

    grid_size = val_set.dataset.grid_size

    # Visualize predictions on a few validation samples
    num_examples = 3
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


if __name__ == "__main__":
    main()
