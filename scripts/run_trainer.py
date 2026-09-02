"""Train the neural vector-field model, then visualize predictions.

Set ``config.dataset`` to ``"2d"`` or ``"3d"`` to choose which dataset/model to
train. The visualization branches on the same key.

Run from the repo root:
    python scripts/run_trainer.py
"""

import os
import argparse
import datetime

from gridflow_nn.training import Trainer, Config
from run_visualizer import visualize_prediction_2d, visualize_prediction_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip the post-training visualization (no plot windows).",
    )
    args = parser.parse_args()

    config = Config()

    os.makedirs("models", exist_ok=True)
    dataset_dim = config.dataset
    model_name = f"models/vfn_{dataset_dim}_{config.method}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    trainer = Trainer(config=config, save_path=model_name)
    model, val_set, device = trainer.train()

    if args.headless:
        return

    grid_size = val_set.dataset.grid_size

    if trainer.dataset_key == "3d":
        indices = list(val_set.indices)[:2]
        visualize_prediction_3d(
            model, val_set.dataset, indices, grid_size, device, step=1.0, label="Val"
        )
    else:
        indices = list(val_set.indices)[:3]
        visualize_prediction_2d(
            model, val_set.dataset, indices, grid_size, device, step=1.0, label="Val"
        )


if __name__ == "__main__":
    main()
