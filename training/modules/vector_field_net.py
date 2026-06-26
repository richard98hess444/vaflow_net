"""Neural network modules for the vector-field model.

VectorFieldNet: takes [latent, goal, query] and predicts the flow vector at
the query cell.
"""

import torch
import torch.nn as nn

from modules.grid_encoder import GridEncoder


class VectorFieldNet(nn.Module):
    """Neural field: (grid, goal, query) -> predicted flow vector."""

    def __init__(self, latent_dim=64, hidden_dim=128, coord_dim=2, out_dim=2):
        super().__init__()
        self.encoder = GridEncoder(latent_dim)
        in_dim = latent_dim + coord_dim + coord_dim  # latent + goal + query
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, grid, goal, query):
        """
        Args:
            grid: (B, 1, n, n) obstacle grids
            goal: (B, 2) normalized goal coordinates
            query: (B, Q, 2) normalized query coordinates

        Returns:
            (B, Q, 2) predicted flow vectors at the query cells.
        """
        latent = self.encoder(grid)  # (B, latent_dim)
        num_queries = query.shape[1]

        latent = latent.unsqueeze(1).expand(-1, num_queries, -1)
        goal = goal.unsqueeze(1).expand(-1, num_queries, -1)

        feat = torch.cat([latent, goal, query], dim=-1)  # (B, Q, latent+4)
        return self.mlp(feat)  # (B, Q, 2)
