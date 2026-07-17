import torch
from torch.utils.data import Dataset
import numpy as np

from gridflow.algorithms import AStar, BreadthFirstSearch, VectorFieldGuidance


class FlowFieldDataset(Dataset):
    """Precomputed (grid, goal, target vector field) samples."""

    def __init__(
        self,
        num_samples,
        grid_size,
        obstacle_prob,
        seed=0,
        abs_seed=None,
        check_reachable=True,
        max_attempts=None,
    ):
        self.grid_size = grid_size
        self.samples = []

        bfs = BreadthFirstSearch()  # used to generate the grids
        guide = VectorFieldGuidance()  # shared ground-truth construction
        astar = AStar()  # reachability check (center -> goal)
        rng = np.random.default_rng(seed)
        center = (grid_size // 2, grid_size // 2)

        if max_attempts is None:
            max_attempts = max(1000, 50 * num_samples)
        attempt = 0
        fail_count = 0
        while len(self.samples) < num_samples and attempt < max_attempts:
            grid = bfs.generate_sparse_grid(
                n=grid_size,
                obstacle_prob=obstacle_prob,
                seed=attempt if abs_seed is None else abs_seed,
            )
            goal = self._random_free_cell(grid, rng)
            if abs_seed is not None and len(self.samples) == 0:
                goal = (0, 17)  # fixed goal for the first sample
            attempt += 1

            if not self._is_sample_valid(grid, goal):
                fail_count += 1
                continue

            if check_reachable and not astar.compute_path(grid, center, goal):
                fail_count += 1
                continue

            target = guide.bfs_method(grid, goal)
            self.samples.append((grid, goal, target))

            if len(self.samples) % 1000 == 0:
                print(f"Collected {len(self.samples)} / {num_samples} samples...")

        if len(self.samples) < num_samples:
            print(
                f"Warning: FlowFieldDataset collected only {len(self.samples)} "
                f"/ {num_samples} samples after {attempt} attempts "
                f"(obstacle_prob={obstacle_prob} may be too high)."
            )

    @staticmethod
    def _is_sample_valid(grid, goal):
        """True if the sample is usable for training.

        Both the grid center (where the robot stands) and the goal cell must be
        free (0 = free, 1 = obstacle).
        """
        n = len(grid)
        center = (n // 2, n // 2)
        if grid[center[0], center[1]] != 0:
            return False
        if grid[goal[0], goal[1]] != 0:
            return False
        return True

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
