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


class QueryHead(nn.Module):
    def __init__(
        self, feat_channels, out_dim=2, hidden=128, fourier_bands=0, coord_dim=4
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
        """feat (B,C,H,W); queries (B,Q,2) as (row,col) cell units; goal (B,2) in [0,1]."""
        _, _, h, w = feat.shape
        denom = torch.tensor([h - 1, w - 1], device=feat.device, dtype=feat.dtype)
        q_norm = queries / denom  # (B, Q, 2) as (row, col) in [0, 1]
        # grid_sample wants (x, y) = (col, row) in [-1, 1]
        sample_grid = (q_norm[..., [1, 0]] * 2.0 - 1.0).unsqueeze(2)  # (B, Q, 1, 2)
        sampled = F.grid_sample(
            feat,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B, C, Q, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, Q, C)

        offset = q_norm - goal.unsqueeze(1)  # (B, Q, 2) goal-relative coords
        coords = torch.cat([q_norm, offset], dim=-1)  # (B, Q, 4)
        mlp_in = torch.cat([sampled, self.positional_encode(coords)], dim=-1)
        return self.mlp(mlp_in)  # (B, Q, 2)


class VectorFieldNet(nn.Module):
    """Dense obstacle-aware vector-field predictor.

    Predicts the full ``(B, H, W, 2)`` flow field directly from the obstacle grid and goal.
    """

    def __init__(
        self,
        base_channels=32,
        out_dim=2,
        query_hidden=128,
        fourier_bands=0,
    ):
        super().__init__()

        in_channels = 6  # obstacle, goal map, row/col coords, row/col offsets
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = nn.Conv2d(base_channels, out_dim, kernel_size=1)

        self.query_head = QueryHead(
            base_channels,
            out_dim=out_dim,
            hidden=query_hidden,
            fourier_bands=fourier_bands,
        )

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

    def _encode(self, grid, goal):
        """Run the U-Net trunk and return the shared feature map (B, C, H, W)."""
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
        return dec1

    def forward_dense(self, grid, goal):
        dense = self.head(self._encode(grid, goal))
        return dense.permute(0, 2, 3, 1)  # (B, 1, H, W) -> (B, H, W, 2)

    def encode(self, grid, goal):
        """Run the U-Net trunk once; reuse the returned feature for many queries."""
        return self._encode(grid, goal)

    def query(self, feat, queries, goal):
        """Flow at continuous coords from a precomputed feature map (see ``encode``)."""
        return self.query_head(feat, queries, goal)

    def forward_query(self, grid, goal, queries):
        """Predict the flow vector at arbitrary continuous query coordinates."""
        return self.query(self.encode(grid, goal), queries, goal)

    def forward_dense_query(self, grid, goal, queries):
        """Dense field and query-head prediction from a single shared encode."""
        feat = self._encode(grid, goal)  # (B, C, H, W)
        dense = self.head(feat).permute(0, 2, 3, 1)  # (B, H, W, 2)
        return dense, self.query_head(feat, queries, goal)

    def forward(self, grid, goal):
        dense_field = self.forward_dense(grid, goal)
        return dense_field
