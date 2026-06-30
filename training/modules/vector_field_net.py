"""Neural network modules for vector-field prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class VectorFieldNet(nn.Module):
    """Dense obstacle-aware vector-field predictor.

    Predicts the full ``(B, H, W, 2)`` flow field directly from the obstacle grid and goal.
    """

    def __init__(
        self,
        latent_dim=64,
        hidden_dim=128,
        base_channels=32,
        out_dim=2,
    ):
        super().__init__()
        del latent_dim, hidden_dim

        in_channels = 6  # obstacle, goal map, row/col coords, row/col offsets
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = nn.Conv2d(base_channels, out_dim, kernel_size=1)

    @staticmethod
    def _build_input_features(grid, goal):
        batch_size, _, height, width = grid.shape
        device = grid.device
        dtype = grid.dtype

        row_coords = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        col_coords = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        row_grid = row_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        col_grid = col_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)

        goal_row = goal[:, 0].view(batch_size, 1, 1, 1)
        goal_col = goal[:, 1].view(batch_size, 1, 1, 1)
        row_offset = row_grid - goal_row
        col_offset = col_grid - goal_col

        goal_map = torch.zeros(
            (batch_size, 1, height, width), device=device, dtype=dtype
        )
        goal_rows = torch.round(goal[:, 0] * (height - 1)).long().clamp(0, height - 1)
        goal_cols = torch.round(goal[:, 1] * (width - 1)).long().clamp(0, width - 1)
        goal_map[
            torch.arange(batch_size, device=device),
            0,
            goal_rows,
            goal_cols,
        ] = 1.0

        return torch.cat(
            [grid, goal_map, row_grid, col_grid, row_offset, col_offset], dim=1
        )

    def forward_dense(self, grid, goal):
        features = self._build_input_features(grid, goal)

        enc1 = self.enc1(features)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))

        up2 = F.interpolate(
            bottleneck,
            size=enc2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = F.interpolate(
            dec2,
            size=enc1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        dense = self.head(dec1)
        return dense.permute(0, 2, 3, 1)  # (B, 1, H, W) -> (B, H, W, 2)

    def forward(self, grid, goal):
        dense_field = self.forward_dense(grid, goal)
        return dense_field
