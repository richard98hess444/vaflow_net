"""Train a dense neural vector-field representation of water flow.

The model predicts the full 2D flow field from the obstacle grid and goal,
which preserves spatial obstacle geometry throughout the prediction path.
"""

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from gridflow_nn import VectorFieldNet, VectorFieldNet3D
from gridflow_nn.inference import predict_full_field
from gridflow_nn.inference_3d import predict_full_field_3d
from .dataloader import (
    FlowFieldDataset,
    FlowFieldDataset3D,
    ChunkedFlowFieldDataset3D,
    ChunkAwareSampler,
)
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

        self.ckpt_dir = self.save_path.with_suffix("")
        self.tb_log_dir = self.ckpt_dir / "tensorboard"
        self.save_model_freq = 10  # epochs

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
        self.display_freq = 1  # epochs

    def predict_full_field(
        self,
        model,
        grid_t: torch.Tensor,
        goal_t: torch.Tensor,
        grid_size: int,
        device: torch.device,
        step: float = 1.0,
    ):
        return self._predict_fn(model, grid_t, goal_t, grid_size, device, step)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    def _dataset_dir(self):
        """Return the per-config directory that holds this dataset's file(s).

        The directory name encodes every dataset-defining value (grid size,
        obstacle density, sample count, seed, reachability) so different configs
        never collide. 2D datasets live under ``dataset_dir_2d`` and 3D under
        ``dataset_dir_3d``; returns ``None`` when the relevant base dir is unset.
        """
        config = self.config
        base = (
            getattr(config, "dataset_dir_2d", None)
            if self.dataset_key == "2d"
            else getattr(config, "dataset_dir_3d", None)
        )
        if not base:
            return None
        return Path(base) / config.dataset_subdir(self.dataset_key)

    def _load_or_build_dataset(self):
        """Return the dataset, loading a cached copy when one exists."""
        config = self.config
        ds_dir = self._dataset_dir()
        use_cache = getattr(config, "use_dataset_cache", False)

        if (
            self.dataset_key == "3d"
            and ds_dir is not None
            and any(ds_dir.glob("*-*.npz"))
        ):
            print(f"Loading chunked dataset (lazy, from disk) from {ds_dir}")
            return ChunkedFlowFieldDataset3D(
                ds_dir, cache_chunks=getattr(config, "cache_chunks", 1)
            )

        cache_file = None
        if ds_dir is not None:
            cache_file = ds_dir / f"0-{config.num_samples}.npz"
            if self.dataset_key == "2d" and use_cache and cache_file.exists():
                print(f"Loading cached dataset from {cache_file}")
                return self.dataset_cls.load(cache_file)

        print(f"Generating dataset ({self.dataset_key})...")

        if self.dataset_key == "3d" and use_cache and ds_dir is not None:
            FlowFieldDataset3D(
                num_samples=config.num_samples,
                grid_size=config.grid_size,
                obstacle_prob=config.obstacle_prob,
                seed=config.seed,
                check_reachable=getattr(config, "check_reachable", True),
                save_dir=ds_dir,
                chunk_size=config.num_samples,
            )
            return ChunkedFlowFieldDataset3D(
                ds_dir, cache_chunks=getattr(config, "cache_chunks", 1)
            )

        dataset = self.dataset_cls(
            num_samples=config.num_samples,
            grid_size=config.grid_size,
            obstacle_prob=config.obstacle_prob,
            seed=config.seed,
            check_reachable=getattr(config, "check_reachable", True),
        )

        if self.dataset_key == "2d" and use_cache and cache_file is not None:
            dataset.save(cache_file)
            print(f"Saved dataset to {cache_file}")

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

        num_workers = getattr(config, "num_workers", 0)
        pin_memory = getattr(config, "pin_memory", False)
        if isinstance(dataset, ChunkedFlowFieldDataset3D):
            train_loader = DataLoader(
                train_set,
                batch_size=config.batch_size,
                sampler=ChunkAwareSampler(train_set, shuffle=True, seed=config.seed),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=config.batch_size,
                sampler=ChunkAwareSampler(val_set, shuffle=False, seed=config.seed),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
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

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(self.tb_log_dir))
        print(f"Logging tensorboard to {self.tb_log_dir.resolve()}")

        best_state = None
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")

        start_time = time.perf_counter()

        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()
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
            epoch_time = time.perf_counter() - epoch_start
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            if epoch % self.save_model_freq == 0 or epoch == config.epochs - 1:
                self.save(self.ckpt_dir / f"epoch_{epoch}.pt", epoch=epoch)

            if (epoch + 1) % self.display_freq == 0 or epoch == config.epochs - 1:
                elapsed = time.perf_counter() - start_time
                print(
                    f"Epoch [{epoch + 1}/{config.epochs}] "
                    f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}  "
                    f"epoch_time: {epoch_time:.1f}s  elapsed: {elapsed:.1f}s"
                )

        print(f"Best val loss: {self.best_val_loss:.6f}")
        writer.close()
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

    def save(self, path=None, epoch=None):
        config = self.config
        path = Path(path) if path is not None else self.save_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "architecture": "dense_cnn",
                "model_channels": config.model_channels,
                "grid_size": config.grid_size,
                "dataset": self.dataset_key,
            },
            "best_val_loss": self.best_val_loss,
        }
        if epoch is not None:
            payload["epoch"] = epoch
        torch.save(payload, path)
        print(f"Saved model to {path.resolve()}")
