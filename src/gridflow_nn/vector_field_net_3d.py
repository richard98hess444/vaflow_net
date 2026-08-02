"""3D neural network modules for volumetric vector-field prediction.

The 3D counterpart of :mod:`gridflow_nn.vector_field_net`. Every 2D operation
becomes volumetric: ``Conv2d`` -> ``Conv3d``, ``MaxPool2d`` -> ``MaxPool3d``,
bilinear -> trilinear interpolation, and the coordinate/offset feature planes
gain a third axis. The model predicts the full ``(B, D, H, W, 3)`` flow field
directly from the obstacle volume and goal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class VectorFieldNet3D(nn.Module):
    """Dense obstacle-aware 3D vector-field predictor.

    Predicts the full ``(B, D, H, W, 3)`` flow field directly from the obstacle
    volume and goal.
    """

    def __init__(
        self,
        base_channels=32,
        out_dim=3,
    ):
        super().__init__()

        # obstacle, goal map, i/j/k coords, i/j/k offsets.
        in_channels = 8
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock3D(
            base_channels * 4 + base_channels * 2, base_channels * 2
        )
        self.dec1 = ConvBlock3D(base_channels * 2 + base_channels, base_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.head = nn.Conv3d(base_channels, out_dim, kernel_size=1)

    @staticmethod
    def _build_input_features(grid, goal):
        batch_size, _, depth, height, width = grid.shape
        device = grid.device
        dtype = grid.dtype

        i_coords = torch.linspace(0.0, 1.0, depth, device=device, dtype=dtype)
        j_coords = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        k_coords = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        i_grid = i_coords.view(1, 1, depth, 1, 1).expand(
            batch_size, 1, depth, height, width
        )
        j_grid = j_coords.view(1, 1, 1, height, 1).expand(
            batch_size, 1, depth, height, width
        )
        k_grid = k_coords.view(1, 1, 1, 1, width).expand(
            batch_size, 1, depth, height, width
        )

        goal_i = goal[:, 0].view(batch_size, 1, 1, 1, 1)
        goal_j = goal[:, 1].view(batch_size, 1, 1, 1, 1)
        goal_k = goal[:, 2].view(batch_size, 1, 1, 1, 1)
        i_offset = i_grid - goal_i
        j_offset = j_grid - goal_j
        k_offset = k_grid - goal_k

        goal_map = torch.zeros(
            (batch_size, 1, depth, height, width), device=device, dtype=dtype
        )
        goal_is = torch.round(goal[:, 0] * (depth - 1)).long().clamp(0, depth - 1)
        goal_js = torch.round(goal[:, 1] * (height - 1)).long().clamp(0, height - 1)
        goal_ks = torch.round(goal[:, 2] * (width - 1)).long().clamp(0, width - 1)
        goal_map[
            torch.arange(batch_size, device=device),
            0,
            goal_is,
            goal_js,
            goal_ks,
        ] = 1.0

        return torch.cat(
            [grid, goal_map, i_grid, j_grid, k_grid, i_offset, j_offset, k_offset],
            dim=1,
        )

    def forward_dense(self, grid, goal):
        features = self._build_input_features(grid, goal)

        enc1 = self.enc1(features)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))

        up2 = F.interpolate(
            bottleneck,
            size=enc2.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = F.interpolate(
            dec2,
            size=enc1.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        dense = self.head(dec1)  # (B, 3, D, H, W)
        return dense.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

    def forward(self, grid, goal):
        dense_field = self.forward_dense(grid, goal)
        return dense_field
