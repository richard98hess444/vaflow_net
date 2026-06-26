"""Train a neural vector-field representation of water flow.

The model is a coordinate-based neural field with two components:

1. GridEncoder (CNN): encodes the obstacle grid into a latent vector.
2. FieldMLP: takes [latent, goal, query] and predicts the flow vector at the
   query cell.

Target: the (reversed) flow vector field produced by the WaterFlow algorithm,
i.e. for a query cell the model must regress the 2D flow vector at that cell.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from modules import VectorFieldNet
from dataloader import FlowFieldDataset
from configs import Config


class Trainer:
    """Train and evaluate the neural vector-field model.

    Bundles the model, optimizer, data and training loop into a single object.
    """

    def __init__(self, config: Config, save_path="vfn_model.pt"):
        self.config = config
        self.save_path = Path(save_path)

        torch.manual_seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = VectorFieldNet(
            latent_dim=config.latent_dim, hidden_dim=config.hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------------
    # Query sampling
    # ------------------------------------------------------------------------
    @staticmethod
    def sample_queries(target_field, num_queries, grid_size):
        """Sample random query cells and gather their target vectors.

        Args:
            target_field: (B, n, n, 2) target vector field
            num_queries: number of query points per sample
            grid_size: n

        Returns:
            query_coords: (B, Q, 2) normalized query coordinates
            target_vecs: (B, Q, 2) target vectors at those coordinates
        """
        batch_size = target_field.shape[0]
        n = grid_size

        rows = torch.randint(0, n, (batch_size, num_queries))
        cols = torch.randint(0, n, (batch_size, num_queries))

        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, num_queries)
        target_vecs = target_field[batch_idx, rows, cols]  # (B, Q, 2)

        query_coords = torch.stack([rows, cols], dim=-1).float() / (n - 1)  # (B, Q, 2)
        return query_coords, target_vecs

    @staticmethod
    def predict_full_field(model, grid_t, goal_t, grid_size, device, step=1.0):
        """Predict the flow vector on a (possibly sub-cell) grid of query points.

        Args:
            model: trained VectorFieldNet
            grid_t: (1, n, n) obstacle grid tensor
            goal_t: (2,) normalized goal coordinates
            grid_size: n
            device: torch device
            step: spacing between query points in cell units. 1.0 queries integer
                cells (0, 1, ..., n-1); 0.5 queries half-cell offsets, etc.

        Returns:
            field: (R, R, 2) numpy array of predicted vectors, where
                R = len(positions).
            positions: (R,) numpy array of the query coordinates (in cell units)
                along each axis, e.g. [0, 0.5, 1.0, ...].
        """
        n = grid_size
        model.eval()

        positions = np.arange(0, n - 1 + 1e-9, step)
        res = len(positions)
        coords = [[r, c] for r in positions for c in positions]
        query = torch.tensor(coords, dtype=torch.float32) / (
            n - 1
        )  # (R*R, 2), normalize because model expects [0,1] coordinates
        query = query.unsqueeze(0).to(device)  # (1, R*R, 2)

        grid_b = grid_t.unsqueeze(0).to(device)  # (1, 1, n, n)
        goal_b = goal_t.unsqueeze(0).to(device)  # (1, 2)

        with torch.no_grad():
            pred = model(grid_b, goal_b, query)  # (1, R*R, 2)

        return pred.reshape(res, res, 2).cpu().numpy(), positions

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    def _build_loaders(self):
        """Build the dataset and return (train_loader, val_loader, val_set)."""
        config = self.config

        dataset = FlowFieldDataset(
            num_samples=config.num_samples,
            grid_size=config.grid_size,
            obstacle_prob=config.obstacle_prob,
            threshold=config.threshold,
            seed=config.seed,
        )

        val_size = int(config.num_samples * config.val_fraction)
        train_size = config.num_samples - val_size
        generator = torch.Generator().manual_seed(config.seed)
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        print(f"Train samples: {train_size}, Val samples: {val_size}")

        train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=config.batch_size, shuffle=False
        )
        return train_loader, val_loader, val_set

    # ------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------
    def evaluate(self, loader):
        """Compute average loss over a data loader (no gradient updates)."""
        config = self.config
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for grid_t, goal_t, target_t in loader:
                grid_t = grid_t.to(self.device)
                goal_t = goal_t.to(self.device)
                target_t = target_t.to(self.device)

                query_coords, target_vecs = self.sample_queries(
                    target_t, config.num_queries, config.grid_size
                )
                query_coords = query_coords.to(self.device)
                target_vecs = target_vecs.to(self.device)

                pred = self.model(grid_t, goal_t, query_coords)
                loss = self.criterion(pred, target_vecs)
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

        for epoch in range(config.epochs):
            self.model.train()
            epoch_loss = 0.0

            for grid_t, goal_t, target_t in train_loader:
                grid_t = grid_t.to(self.device)
                goal_t = goal_t.to(self.device)
                target_t = target_t.to(self.device)

                query_coords, target_vecs = self.sample_queries(
                    target_t, config.num_queries, config.grid_size
                )
                query_coords = query_coords.to(self.device)
                target_vecs = target_vecs.to(self.device)

                pred = self.model(grid_t, goal_t, query_coords)  # (B, Q, 2)
                loss = self.criterion(pred, target_vecs)

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
                print(
                    f"Epoch [{epoch + 1}/{config.epochs}] "
                    f"train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}"
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
                    "latent_dim": config.latent_dim,
                    "hidden_dim": config.hidden_dim,
                    "grid_size": config.grid_size,
                },
                "best_val_loss": self.best_val_loss,
            },
            self.save_path,
        )
        print(f"Saved model to {self.save_path.resolve()}")
