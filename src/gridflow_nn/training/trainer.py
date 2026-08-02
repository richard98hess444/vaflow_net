"""Train a dense neural vector-field representation of water flow.

The model predicts the full 2D flow field from the obstacle grid and goal,
which preserves spatial obstacle geometry throughout the prediction path.
"""

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from gridflow_nn import VectorFieldNet, VectorFieldNet3D
from gridflow_nn.inference import predict_full_field
from gridflow_nn.inference_3d import predict_full_field_3d
from .dataloader import FlowFieldDataset, FlowFieldDataset3D
from .configs import Config

DATASETS = {
    "2d": {
        "dataset": FlowFieldDataset,
        "model": VectorFieldNet,
        "predict": predict_full_field,
    },
    "3d": {
        "dataset": FlowFieldDataset3D,
        "model": VectorFieldNet3D,
        "predict": predict_full_field_3d,
    },
}


class Trainer:
    """Train and evaluate the neural vector-field model.

    Bundles the model, optimizer, data and training loop into a single object.
    """

    def __init__(self, config: Config, save_path="vfn_model.pt"):
        self.config = config
        self.save_path = Path(save_path)

        dataset_key = getattr(config, "dataset", "2d")
        if dataset_key not in DATASETS:
            raise ValueError(
                f"Unknown config.dataset {dataset_key!r}; "
                f"expected one of {sorted(DATASETS)}."
            )
        self.dataset_key = dataset_key
        spec = DATASETS[dataset_key]
        self.dataset_cls = spec["dataset"]
        self.model_cls = spec["model"]
        self._predict_fn = spec["predict"]

        torch.manual_seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = self.model_cls(
            base_channels=config.model_channels,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")

    def predict_full_field(
        self,
        model,
        grid_t: torch.Tensor,
        goal_t: torch.Tensor,
        grid_size: int,
        device: torch.device,
        step: float = 1.0,
    ):
        """Predict the flow field on a (possibly sub-cell) grid of query points.

        Delegates to the 2D or 3D prediction helper selected by
        ``config.dataset`` (see the ``DATASETS`` registry).
        """
        return self._predict_fn(model, grid_t, goal_t, grid_size, device, step)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    def _dataset_cache_path(self):
        """Build a cache filename keyed to the dataset-defining config values.

        Different grid sizes / obstacle densities / sample counts / seeds must
        not collide, so all of them go into the filename.
        """
        config = self.config
        name = (
            f"flow_ds_{self.dataset_key}_n{config.grid_size}_p{config.obstacle_prob}"
            f"_s{config.num_samples}_seed{config.seed}"
            f"_reach{int(getattr(config, 'check_reachable', True))}.npz"
        )
        return Path(config.dataset_cache_dir) / name

    def _load_or_build_dataset(self):
        """Return the dataset, loading a cached copy when one exists.

        When caching is enabled and a matching cache file is present it is
        reused (skipping the slow grid generation, A* reachability check and
        ground-truth computation). Otherwise the dataset is generated and, if
        caching is enabled, written to disk for next time.
        """
        config = self.config

        cache_path = None
        if getattr(config, "use_dataset_cache", False):
            cache_path = self._dataset_cache_path()
            if cache_path.exists():
                print(f"Loading cached dataset from {cache_path}")
                return self.dataset_cls.load(cache_path)

        print(f"Generating dataset ({self.dataset_key})...")
        dataset = self.dataset_cls(
            num_samples=config.num_samples,
            grid_size=config.grid_size,
            obstacle_prob=config.obstacle_prob,
            seed=config.seed,
            check_reachable=getattr(config, "check_reachable", True),
        )

        if cache_path is not None:
            dataset.save(cache_path)
            print(f"Saved dataset cache to {cache_path}")

        return dataset

    def _build_loaders(self):
        """Build the dataset and return (train_loader, val_loader, val_set)."""
        config = self.config

        dataset = self._load_or_build_dataset()

        # Split on the dataset's actual length: reachability filtering (or a
        # loaded cache) may yield fewer than config.num_samples samples.
        total = len(dataset)
        val_size = int(total * config.val_fraction)
        train_size = total - val_size
        generator = torch.Generator().manual_seed(config.seed)
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        print(f"Train samples: {train_size}, Val samples: {val_size}")

        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
        return train_loader, val_loader, val_set

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------
    def evaluate(self, loader):
        """Compute average loss over a data loader (no gradient updates)."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for grid_t, goal_t, target_t in loader:
                grid_t = grid_t.to(self.device)
                goal_t = goal_t.to(self.device)
                target_t = target_t.to(self.device)

                pred = self.model(grid_t, goal_t)
                loss = self.criterion(pred, target_t)
                total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------
    def train(self):
        config = self.config

        train_loader, val_loader, val_set = self._build_loaders()

        best_state = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")

        start_time = time.perf_counter()

        for epoch in range(config.epochs):
            self.model.train()
            epoch_loss = 0.0

            for grid_t, goal_t, target_t in train_loader:
                grid_t = grid_t.to(self.device)
                goal_t = goal_t.to(self.device)
                target_t = target_t.to(self.device)

                pred = self.model(grid_t, goal_t)  # (B, n, n, 2)
                loss = self.criterion(pred, target_t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            train_loss = epoch_loss / max(len(train_loader), 1)
            val_loss = self.evaluate(val_loader)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
                elapsed = time.perf_counter() - start_time
                print(
                    f"Epoch [{epoch + 1}/{config.epochs}] "
                    f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}  "
                    f"elapsed: {elapsed:.1f}s"
                )

        print(f"Best val loss: {self.best_val_loss:.6f}")
        self._plot_history()

        # Restore and save the best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.save()

        return self.model, val_set, self.device

    def _plot_history(self):
        plt.plot(self.train_loss_history, label="train")
        plt.plot(self.val_loss_history, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

    def save(self):
        config = self.config
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "architecture": "dense_cnn",
                    "model_channels": config.model_channels,
                    "grid_size": config.grid_size,
                    "dataset": self.dataset_key,
                },
                "best_val_loss": self.best_val_loss,
            },
            self.save_path,
        )
        print(f"Saved model to {self.save_path.resolve()}")
