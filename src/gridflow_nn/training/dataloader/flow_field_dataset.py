from pathlib import Path

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
        method="eik",
        gpu_batch_size=None,
    ):
        self.grid_size = grid_size
        self.gpu_batch_size = gpu_batch_size
        self.samples = []

        bfs = BreadthFirstSearch()  # used to generate the grids
        guide = VectorFieldGuidance(  # shared ground-truth construction
            apf_gain=0.03,
            apf_influence=5,
            apf_resolution=0.1,
            apf_falloff_power=0.67,
            drag_gain=3.7,
            drag_influence=3.0,
            ns_force_boundary=2.5,
        )
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

            if method == "bfs":
                target = guide.bfs_method(grid, goal)
            elif method == "ns":
                target = None  # filled in one GPU batch after sampling
            else:
                target = guide.eik_method(grid, goal)
            self.samples.append((grid, goal, target))

            if len(self.samples) % 1000 == 0:
                print(f"Collected {len(self.samples)} / {num_samples} samples...")

        if method == "ns":
            self._fill_ns_targets(ndim=2)

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

    def _fill_ns_targets(self, ndim):
        """Fill deferred ns (drag) targets for buffered samples in GPU batches."""
        from gridflow_nn.ns_torch import build_targets

        pending = [i for i, s in enumerate(self.samples) if s[2] is None]
        if not pending:
            return
        grids = [self.samples[i][0] for i in pending]
        goals = [self.samples[i][1] for i in pending]
        batch = self.gpu_batch_size or (256 if ndim == 2 else 32)
        targets = build_targets(
            grids, goals, ndim=ndim, batch_size=batch, progress=True
        )
        for j, i in enumerate(pending):
            g, go, _ = self.samples[i]
            self.samples[i] = (g, go, np.asarray(targets[j], dtype=np.float32))

    def __len__(self):
        return len(self.samples)

    def save(self, path):
        """Save the generated samples to a compressed ``.npz`` file for reuse.

        All samples share the same ``grid_size``, so grids/goals/targets stack
        cleanly into single arrays. Reload with :meth:`load` to skip the (slow)
        grid generation, reachability check and ground-truth computation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        grids = np.stack([grid for grid, _, _ in self.samples])
        goals = np.array([goal for _, goal, _ in self.samples], dtype=np.int64)
        targets = np.stack([target for _, _, target in self.samples])
        np.savez_compressed(
            path,
            grids=grids,
            goals=goals,
            targets=targets,
            grid_size=self.grid_size,
        )

    @classmethod
    def load(cls, path):
        """Reconstruct a dataset previously written by :meth:`save`.

        Bypasses ``__init__`` (and thus all sample generation) and repopulates
        the samples directly from the saved arrays.
        """
        data = np.load(path, allow_pickle=False)
        obj = cls.__new__(cls)
        obj.grid_size = int(data["grid_size"])
        grids = data["grids"]
        goals = data["goals"]
        targets = data["targets"]
        obj.samples = [
            (grids[i], tuple(int(x) for x in goals[i]), targets[i])
            for i in range(len(grids))
        ]
        return obj

    def __getitem__(self, idx):
        grid, goal, target = self.samples[idx]
        n = self.grid_size

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # (1, n, n)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (n - 1)  # (2,) in [0,1]
        target_t = torch.tensor(target, dtype=torch.float32)  # (n, n, 2)

        return grid_t, goal_t, target_t
