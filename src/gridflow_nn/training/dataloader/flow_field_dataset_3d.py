import bisect
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
import numpy as np

from gridflow.algorithms_3d import (
    BreadthFirstSearch3D,
    VectorFieldGuidance3D,
)


class FlowFieldDataset3D(Dataset):
    """Precomputed (voxel grid, goal, target vector field) 3D samples.

    The 3D counterpart of
    :class:`gridflow_nn.training.dataloader.FlowFieldDataset`. Grids are
    ``(n, n, n)`` occupancy volumes, goals are ``(i, j, k)`` voxels and targets
    are ``(n, n, n, 3)`` guidance fields.
    """

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
        save_dir=None,
        chunk_size=1000,
        resume=True,
    ):
        """Generate the dataset, optionally checkpointing to disk in chunks.

        When ``save_dir`` is given, every ``chunk_size`` collected samples are
        written to ``save_dir`` as ``{start}-{end}.npz`` (e.g. ``0-999.npz``,
        ``1000-1999.npz``, ...) together with a ``progress.json`` that records
        exactly where to resume: the next grid seed (``next_attempt``) and the
        goal-RNG state. If the process crashes, re-running with the same
        ``save_dir`` and ``resume=True`` reloads the finished chunks and
        continues from the saved seed instead of restarting from scratch.

        Args:
            save_dir: directory for chunk checkpoints. ``None`` disables saving.
            chunk_size: number of samples per chunk file.
            resume: when True, reuse any chunks/progress already in ``save_dir``.
        """
        self.grid_size = grid_size
        self.samples = []

        bfs = BreadthFirstSearch3D()  # grid generation
        guide = VectorFieldGuidance3D(  # shared ground-truth construction
            gain=0.03,
            influence=5,
            resolution=0.1,
            falloff_power=0.67,
        )
        rng = np.random.default_rng(seed)
        center = (grid_size // 2, grid_size // 2, grid_size // 2)

        if max_attempts is None:
            max_attempts = max(1000, 50 * num_samples)
        attempt = 0

        # Parameters stamped into progress.json so a resume can't silently mix
        # samples generated with a different configuration.
        meta = {
            "grid_size": int(grid_size),
            "obstacle_prob": float(obstacle_prob),
            "method": method,
            "seed": int(seed),
        }

        save_dir = Path(save_dir) if save_dir is not None else None
        saved_count = 0  # number of samples already flushed to chunk files
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            if resume:
                attempt, saved_count = self._resume_from(save_dir, rng, meta)
                if saved_count:
                    print(
                        f"Resuming: {saved_count} samples already on disk, "
                        f"continuing from grid seed (attempt) {attempt}."
                    )

        # `total` counts every accepted sample across the whole run, including
        # those already flushed to disk and freed from RAM. In streaming mode
        # (save_dir set) ``self.samples`` only holds the current, not-yet-saved
        # chunk, so peak memory stays bounded by ``chunk_size`` samples.
        total = saved_count
        display_freq = 10 ** (int(np.log10(num_samples)) - 1) if num_samples >= 10 else 1
        while total < num_samples and attempt < max_attempts:
            grid = bfs.generate_sparse_grid(
                n=grid_size,
                obstacle_prob=obstacle_prob,
                seed=attempt if abs_seed is None else abs_seed,
            )
            goal = self._random_free_cell(grid, rng)
            if abs_seed is not None and total == 0:
                goal = (0, grid_size - 1, grid_size - 1)
            attempt += 1

            if not self._is_sample_valid(grid, goal):
                continue

            if check_reachable and not bfs.is_reachable(grid, center, goal):
                continue

            if method == "bfs":
                target = guide.bfs_method(grid, goal)
            else:
                target = guide.eik_method(grid, goal)
            # Store as float32 up front: training consumes float32 and chunks are
            # saved as float32, so this halves peak buffer RAM vs float64 targets.
            target = np.asarray(target, dtype=np.float32)
            self.samples.append((grid, goal, target))
            total += 1

            # Once a full chunk is buffered, flush it to disk and free the RAM.
            if save_dir is not None and len(self.samples) >= chunk_size:
                self._save_chunk(save_dir, saved_count, attempt, rng, meta)
                saved_count += len(self.samples)
                self.samples = []  # release memory

            if total % display_freq == 0:
                print(f"Collected {total} / {num_samples} samples...")

        # Only whole chunks are persisted; drop any partial tail so resume always
        # restarts on a clean chunk boundary.
        if save_dir is not None and self.samples:
            print(
                f"Discarding {len(self.samples)} trailing samples "
                f"(< chunk_size={chunk_size}); only full chunks are saved."
            )
            self.samples = []

        if total < num_samples:
            print(
                f"Warning: FlowFieldDataset3D collected only {total} "
                f"/ {num_samples} samples after {attempt} attempts "
                f"(obstacle_prob={obstacle_prob} may be too high)."
            )

    # ------------------------------------------------------------------------
    # Chunked checkpointing / resume
    # ------------------------------------------------------------------------
    def _save_chunk(self, save_dir, start, next_attempt, rng, meta):
        """Write the current in-memory buffer (``self.samples``) as one chunk.

        The buffer holds exactly the not-yet-saved samples, so it is written to
        ``{start}-{start+len-1}.npz`` and ``progress.json`` is atomically
        rewritten with the state needed to resume right after this chunk.
        Grids are stored as ``int8`` and targets as ``float32`` to keep files
        (and later RAM when loaded) small.
        """
        chunk = self.samples
        count = len(chunk)
        end = start + count
        grids = np.stack([g for g, _, _ in chunk]).astype(np.int8)
        goals = np.array([goal for _, goal, _ in chunk], dtype=np.int64)
        targets = np.stack([t for _, _, t in chunk]).astype(np.float32)

        path = Path(save_dir) / f"{start}-{end - 1}.npz"
        np.savez_compressed(
            path,
            grids=grids,
            goals=goals,
            targets=targets,
            grid_size=self.grid_size,
            start=start,
            end=end,
        )

        # progress.json: next grid seed + goal-RNG state so resume is exact.
        # Written atomically (temp file + os.replace) so a crash mid-write can't
        # corrupt the resume state.
        progress = {
            "samples_done": end,
            "next_attempt": int(next_attempt),
            "chunk_size": count,
            "rng_state": rng.bit_generator.state,
            **meta,
        }
        tmp = Path(save_dir) / "progress.json.tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f)
        os.replace(tmp, Path(save_dir) / "progress.json")

        print(f"Saved chunk {path.name} ({end} samples total).")

    def _resume_from(self, save_dir, rng, meta):
        """Restore resume state from ``save_dir`` *without* loading samples.

        Returns ``(next_attempt, samples_done)``: the grid seed to continue from
        and how many samples are already persisted in chunk files. Restores
        ``rng`` to the saved state so goal sampling continues deterministically.
        Returns ``(0, 0)`` when there is nothing to resume. Raises if the saved
        configuration differs from the requested one.
        """
        progress_path = Path(save_dir) / "progress.json"
        if not progress_path.exists():
            return 0, 0

        with open(progress_path) as f:
            progress = json.load(f)

        # Refuse to resume into a directory generated with different settings.
        for key, val in meta.items():
            if key in progress and progress[key] != val:
                raise ValueError(
                    f"Cannot resume {save_dir}: saved {key}={progress[key]!r} "
                    f"but requested {val!r}. Use a fresh save_dir."
                )

        rng.bit_generator.state = progress["rng_state"]
        return int(progress["next_attempt"]), int(progress["samples_done"])

    @staticmethod
    def _load_chunk_file(path):
        """Return the list of ``(grid, goal, target)`` tuples in a chunk file."""
        data = np.load(path, allow_pickle=False)
        grids = data["grids"]
        goals = data["goals"]
        targets = data["targets"]
        return [
            (grids[i], tuple(int(x) for x in goals[i]), targets[i])
            for i in range(len(grids))
        ]

    @classmethod
    def load_chunks(cls, save_dir):
        """Rebuild a full dataset from all chunk files in ``save_dir``.

        Use this to assemble the complete dataset after a (possibly resumed)
        chunked generation run, e.g. ``FlowFieldDataset3D.load_chunks("cache/ds")``.
        """
        save_dir = Path(save_dir)
        chunk_files = sorted(
            save_dir.glob("*-*.npz"),
            key=lambda p: int(p.stem.split("-")[0]),
        )
        if not chunk_files:
            raise FileNotFoundError(f"No chunk .npz files found in {save_dir}.")

        obj = cls.__new__(cls)
        obj.samples = []
        for path in chunk_files:
            obj.samples.extend(cls._load_chunk_file(path))
        obj.grid_size = int(np.load(chunk_files[0], allow_pickle=False)["grid_size"])
        return obj


    @staticmethod
    def _is_sample_valid(grid, goal):
        """True if the sample is usable for training.

        Both the grid center (where the robot stands) and the goal voxel must be
        free (0 = free, 1 = obstacle).
        """
        n = len(grid)
        center = (n // 2, n // 2, n // 2)
        if grid[center] != 0:
            return False
        if grid[goal] != 0:
            return False
        return True

    @staticmethod
    def _random_free_cell(grid, rng):
        n = len(grid)
        while True:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            k = int(rng.integers(0, n))
            if grid[i, j, k] == 0:
                return (i, j, k)

    def __len__(self):
        return len(self.samples)

    def get_raw(self, idx):
        """Return the raw ``(grid, goal, target)`` numpy sample at ``idx``.

        Unlike :meth:`__getitem__` (which returns normalized training tensors),
        this returns the plain arrays used for visualization.
        """
        return self.samples[idx]

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

        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # (1,n,n,n)
        goal_t = torch.tensor(goal, dtype=torch.float32) / (n - 1)  # (3,) in [0,1]
        target_t = torch.tensor(target, dtype=torch.float32)  # (n, n, n, 3)

        return grid_t, goal_t, target_t


class ChunkedFlowFieldDataset3D(Dataset):
    """Disk-backed 3D dataset that reads chunk files lazily during training.

    Unlike :meth:`FlowFieldDataset3D.load_chunks` (which loads every sample into
    RAM), this keeps at most ``cache_chunks`` chunk files resident, so a dataset
    far larger than memory can still be trained on. Point it at the ``save_dir``
    produced by a chunked generation run::

        ds = ChunkedFlowFieldDataset3D("cache/dataset_3d")
        loader = DataLoader(ds, batch_size=16, shuffle=True)

    Note: with ``shuffle=True`` random access hops between chunk files, so keep
    ``cache_chunks`` >= 1 (an LRU of decoded chunks). Because a full 61^3 chunk
    is a few GB in RAM, the default cache holds a single chunk; raise it only if
    you have the memory. Using ``num_workers > 0`` gives each worker its own
    cache, which improves throughput.
    """

    def __init__(self, save_dir, cache_chunks=1):
        self.save_dir = Path(save_dir)
        self.chunk_files = sorted(
            self.save_dir.glob("*-*.npz"),
            key=lambda p: int(p.stem.split("-")[0]),
        )
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk .npz files found in {self.save_dir}.")

        # Build a flat index: cumulative sample offset at the start of each file.
        self._offsets = []
        total = 0
        for p in self.chunk_files:
            start, end = (int(x) for x in p.stem.split("-"))
            self._offsets.append(total)
            total += end - start + 1
        self._length = total

        first = np.load(self.chunk_files[0], allow_pickle=False)
        self.grid_size = int(first["grid_size"])

        self._cache_chunks = max(1, int(cache_chunks))
        self._cache = {}  # file index -> (grids, goals, targets)
        self._cache_order = []  # LRU order of file indices

    def __len__(self):
        return self._length

    def get_raw(self, idx):
        """Return the raw ``(grid, goal, target)`` numpy sample at ``idx``."""
        if idx < 0:
            idx += self._length
        if not 0 <= idx < self._length:
            raise IndexError(idx)
        file_idx = bisect.bisect_right(self._offsets, idx) - 1
        local = idx - self._offsets[file_idx]
        grids, goals, targets = self._load_file(file_idx)
        return grids[local], tuple(int(x) for x in goals[local]), targets[local]

    def _load_file(self, file_idx):
        """Return the (grids, goals, targets) arrays for ``file_idx`` (LRU cached)."""
        cached = self._cache.get(file_idx)
        if cached is not None:
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return cached

        data = np.load(self.chunk_files[file_idx], allow_pickle=False)
        arrays = (data["grids"], data["goals"], data["targets"])
        self._cache[file_idx] = arrays
        self._cache_order.append(file_idx)
        if len(self._cache_order) > self._cache_chunks:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        return arrays

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._length
        if not 0 <= idx < self._length:
            raise IndexError(idx)

        file_idx = bisect.bisect_right(self._offsets, idx) - 1
        local = idx - self._offsets[file_idx]
        grids, goals, targets = self._load_file(file_idx)
        n = self.grid_size

        grid_t = torch.tensor(grids[local], dtype=torch.float32).unsqueeze(0)
        goal_t = torch.tensor(goals[local], dtype=torch.float32) / (n - 1)
        target_t = torch.tensor(targets[local], dtype=torch.float32)

        return grid_t, goal_t, target_t

    def chunk_of(self, idx):
        """Return the chunk-file index that sample ``idx`` lives in."""
        return bisect.bisect_right(self._offsets, idx) - 1


class ChunkAwareSampler(Sampler):
    """Sampler that keeps access chunk-local so each chunk is read once/epoch.

    A :class:`ChunkedFlowFieldDataset3D` decompresses a whole chunk file (~GBs)
    on every cache miss. Plain ``shuffle=True`` hops between chunks on every
    sample, so it reloads a full chunk per item and grinds to a halt. This
    sampler instead groups indices by chunk, shuffles the *order of chunks* and
    the *samples within* each chunk, then yields them chunk-by-chunk. With the
    dataset's default 1-chunk cache that means exactly one decompress per chunk
    per epoch, while still giving well-shuffled batches.

    Works over a ``Subset`` of a chunked dataset (as produced by ``random_split``)
    or over the chunked dataset directly.
    """

    def __init__(self, subset, shuffle=True, seed=0):
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Resolve the underlying chunked dataset and the positions to iterate.
        if hasattr(subset, "indices") and hasattr(subset, "dataset"):
            base = subset.dataset
            positions = list(range(len(subset.indices)))
            orig_indices = list(subset.indices)
        else:
            base = subset
            positions = list(range(len(subset)))
            orig_indices = positions

        if not hasattr(base, "chunk_of"):
            raise TypeError(
                "ChunkAwareSampler requires a ChunkedFlowFieldDataset3D "
                "(optionally wrapped in a Subset)."
            )

        # Group DataLoader positions by the chunk their sample belongs to.
        self._groups = {}
        for pos, orig in zip(positions, orig_indices):
            c = base.chunk_of(orig)
            self._groups.setdefault(c, []).append(pos)
        self._length = len(positions)

    def __len__(self):
        return self._length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        chunk_ids = sorted(self._groups)
        if self.shuffle:
            rng.shuffle(chunk_ids)

        order = []
        for c in chunk_ids:
            positions = list(self._groups[c])
            if self.shuffle:
                rng.shuffle(positions)
            order.extend(positions)

        self.epoch += 1
        return iter(order)
