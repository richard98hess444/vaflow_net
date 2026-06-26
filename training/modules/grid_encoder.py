"""Neural network modules for the vector-field model.

GridEncoder (CNN): encodes the obstacle grid into a latent vector.
"""

import torch.nn as nn


class GridEncoder(nn.Module):
    """CNN that embeds an obstacle grid into a latent vector."""

    def __init__(self, latent_dim=64, pooled=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pooled, pooled)),  # decouple from grid size
        )
        self.fc = nn.Linear(32 * pooled * pooled, latent_dim)

    def forward(self, grid):
        x = self.conv(grid)  # (B, 32, pooled, pooled)
        x = x.flatten(1)  # (B, 32 * pooled * pooled)
        return self.fc(x)  # (B, latent_dim)
