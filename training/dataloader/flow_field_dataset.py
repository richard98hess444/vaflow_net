import torch
from torch.utils.data import Dataset
import numpy as np

from gridflow.algorithms.flow import WaterFlow

class FlowFieldDataset(Dataset):
    """Precomputed (grid, goal, target vector field) samples."""

    def __init__(
        self, num_samples, grid_size, obstacle_prob, threshold, seed=0, abs_seed=None
    ):
        self.grid_size = grid_size
        self.samples = []

        wf = WaterFlow()
        rng = np.random.default_rng(seed)

        for i in range(num_samples):
            grid = wf.generate_random_grid(
                n=grid_size,
                obstacle_prob=obstacle_prob,
                seed=i if abs_seed is None else abs_seed,
            )
            goal = self._random_free_cell(grid, rng)
            if abs_seed is not None and i == 0:
                goal = (0, 17)  # fixed goal for first sample
            target = self.compute_target_vector_field(wf, grid, goal, threshold)
            self.samples.append((grid, goal, target))

    def compute_target_vector_field(self, wf: WaterFlow, grid, goal, threshold):
        """Run the full WaterFlow pipeline and return the flow vector field.

        Returns:
            target_vectors: np.ndarray of shape (n, n, 2), the reversed flow
                vectors (pointing toward the goal) scaled by strength.
        """
        n = len(grid)

        arrival_time_map = wf.compute_arrival_time_map(grid, source=goal)
        strength_map = wf.compute_strength_map(arrival_time_map, source=goal)
        viscosity_map = wf.compute_viscosity_map(grid, source=goal)
        filtered_map = wf.compute_filtered_map(
            strength_map, viscosity_map, grid, source=goal
        )

        thresholded_grid = wf.apply_strength_threshold(
            filtered_map, grid, threshold=threshold, source=goal
        )

        binary_thresholded_grid = [
            [1 if thresholded_grid[r][c] != 0 else 0 for c in range(n)] for r in range(n)
        ]
        binary_thresholded_grid[goal[0]][goal[1]] = 0  # ensure goal is free

        flow_map_t = wf.compute_flow_map(binary_thresholded_grid, source=goal)
        flow_map_r = wf.reverse_flow_map(flow_map_t)
        target_vectors = wf.convert_flow_map_to_vectors(flow_map_r, strength_map)

        return target_vectors.astype(np.float32)

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