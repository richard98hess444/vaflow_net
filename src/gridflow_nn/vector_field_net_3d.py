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


class QueryHead3D(nn.Module):
    def __init__(
        self, feat_channels, out_dim=3, hidden=128, fourier_bands=0, coord_dim=6
    ):
        super().__init__()
        self.fourier_bands = fourier_bands
        enc_dim = coord_dim * (1 + 2 * fourier_bands)
        self.mlp = nn.Sequential(
            nn.Linear(feat_channels + enc_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def positional_encode(self, coords):
        """coords (B, Q, D) -> (B, Q, D*(1+2F)) sinusoidal encoding (F = fourier_bands)."""
        if self.fourier_bands == 0:
            return coords
        freqs = (
            2.0
            ** torch.arange(
                self.fourier_bands, device=coords.device, dtype=coords.dtype
            )
        ) * torch.pi
        scaled = coords.unsqueeze(-1) * freqs  # (B, Q, D, F)
        enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (B, Q, D, 2F)
        enc = enc.reshape(*coords.shape[:-1], -1)  # (B, Q, D*2F)
        return torch.cat([coords, enc], dim=-1)  # (B, Q, D*(1+2F))

    def forward(self, feat, queries, goal):
        """feat (B,C,D,H,W); queries (B,Q,3) as (i,j,k) voxel units; goal (B,3) in [0,1]."""
        _, _, d, h, w = feat.shape
        denom = torch.tensor(
            [d - 1, h - 1, w - 1], device=feat.device, dtype=feat.dtype
        )
        q_norm = queries / denom  # (B, Q, 3) as (i, j, k) in [0, 1]
        # grid_sample wants (x, y, z) = (k, j, i) in [-1, 1]
        sample_grid = (q_norm[..., [2, 1, 0]] * 2.0 - 1.0).view(
            feat.shape[0], -1, 1, 1, 3
        )  # (B, Q, 1, 1, 3)
        sampled = F.grid_sample(
            feat,
            sample_grid,
            mode="bilinear",  # trilinear for 5D input
            padding_mode="border",
            align_corners=True,
        )  # (B, C, Q, 1, 1)
        sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, Q, C)

        offset = q_norm - goal.unsqueeze(1)  # (B, Q, 3) goal-relative coords
        coords = torch.cat([q_norm, offset], dim=-1)  # (B, Q, 6)
        mlp_in = torch.cat([sampled, self.positional_encode(coords)], dim=-1)
        return self.mlp(mlp_in)  # (B, Q, 3)


class VectorFieldNet3D(nn.Module):
    """Dense obstacle-aware 3D vector-field predictor.

    Predicts the full ``(B, D, H, W, 3)`` flow field directly from the obstacle
    volume and goal.
    """

    def __init__(
        self,
        base_channels=32,
        out_dim=3,
        query_hidden=128,
        fourier_bands=0,
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

        self.query_head = QueryHead3D(
            base_channels,
            out_dim=out_dim,
            hidden=query_hidden,
            fourier_bands=fourier_bands,
        )

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
        dense = self.head(self._encode(grid, goal))  # (B, 3, D, H, W)
        return dense.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

    def _encode(self, grid, goal):
        """Run the U-Net trunk and return the shared feature map (B, C, D, H, W)."""
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
        return dec1

    def forward_query(self, grid, goal, queries):
        """Predict the flow vector at arbitrary continuous query coordinates."""
        feat = self._encode(grid, goal)  # (B, C, D, H, W)
        return self.query_head(feat, queries, goal)

    def forward(self, grid, goal):
        dense_field = self.forward_dense(grid, goal)
        return dense_field
