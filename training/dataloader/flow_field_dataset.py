import torch
from torch.utils.data import Dataset
import numpy as np

from gridflow.algorithms import Eikonal, SignedDistanceField


class FlowFieldDataset(Dataset):
    """Precomputed (grid, goal, target vector field) samples."""

    def __init__(self, num_samples, grid_size, obstacle_prob, seed=0, abs_seed=None):
        self.grid_size = grid_size
        self.samples = []

        eik = Eikonal()
        sdf = SignedDistanceField()
        rng = np.random.default_rng(seed)

        for i in range(num_samples):
            grid = eik.generate_sparse_grid(
                n=grid_size,
                obstacle_prob=obstacle_prob,
                seed=i if abs_seed is None else abs_seed,
            )
            goal = self._random_free_cell(grid, rng)
            if abs_seed is not None and i == 0:
                goal = (0, 17)  # fixed goal for first sample
            target = self.compute_target_vector_field(eik, sdf, grid, goal)
            self.samples.append((grid, goal, target))

    def compute_target_vector_field(
        self,
        eik: Eikonal,
        sdf: SignedDistanceField,
        grid,
        goal,
    ):
        """Compute the ground-truth guidance field for one sample.

        The target combines the Eikonal shortest-path gradient (attraction
        toward the goal) with the SDF artificial-potential-field repulsion
        (pushing away from obstacles); the goal cell is zeroed.

        Returns:
            target_vectors: np.ndarray of shape (n, n, 2) of (drow, dcol)
                guidance vectors, dtype float32.
        """

        eik_arr_time_map = eik.compute_arrival_time_map(grid, source=goal)
        eik_gradient_map = eik.compute_gradient_map(eik_arr_time_map, grid)

        sdf_map = sdf.compute_sdf_map(grid)
        repulsive_force_map = sdf.compute_repulsive_force_map(
            grid, sdf=sdf_map, influence=3.0, gain=1.0
        )

        final_map = eik_gradient_map + repulsive_force_map
        final_map[goal[0]][goal[1]] = np.array([0.0, 0.0])

        return final_map.astype(np.float32)

    @staticmethod
    def _random_free_cell(grid, rng):
        n = len(grid)
        while True:
            r = int(rng.integers(0, n))
            c = int(rng.integers(0, n))
            if grid[r][c] == 0:
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
