import torch
from torch.utils.data import Dataset
import numpy as np

from gridflow.algorithms import BreadthFirstSearch, VectorFieldGuidance


class FlowFieldDataset(Dataset):
    """Precomputed (grid, goal, target vector field) samples."""

    def __init__(self, num_samples, grid_size, obstacle_prob, seed=0, abs_seed=None):
        self.grid_size = grid_size
        self.samples = []

        bfs = BreadthFirstSearch()  # used to generate the grids
        guide = VectorFieldGuidance()  # shared ground-truth construction
        rng = np.random.default_rng(seed)

        for i in range(num_samples):
            grid = bfs.generate_sparse_grid(
                n=grid_size,
                obstacle_prob=obstacle_prob,
                seed=i if abs_seed is None else abs_seed,
            )
            goal = self._random_free_cell(grid, rng)
            if abs_seed is not None and i == 0:
                goal = (0, 17)  # fixed goal for first sample

            target = guide.bfs_method(grid, goal)
            self.samples.append((grid, goal, target))

    @staticmethod
    def _random_free_cell(grid, rng):
        n = len(grid)
        while True:
            r = int(rng.integers(0, n))
            c = int(rng.integers(0, n))
            if grid[r, c] == 0:
                return (r, c)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grid, goal, target = self.samples[idx]
        n = self.grid_size

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # (1, n, n)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (n - 1)  # (2,) in [0,1]
        target_t = torch.tensor(target, dtype=torch.float32)  # (n, n, 2)

        return grid_t, goal_t, target_t
