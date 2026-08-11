"""Generate a large 3D flow-field dataset to disk in memory-safe chunks.

Every ``chunk_size`` accepted samples are written to a per-config subfolder of
``dataset_dir_3d`` as ``{start}-{end}.npz`` (e.g. ``0-999.npz``) and then freed
from RAM, so peak memory stays bounded by one chunk regardless of
``num_samples``.

The dataset-defining values (grid_size, obstacle_prob, num_samples, seed,
check_reachable, dataset_dir_3d) are read from the shared training ``Config`` so
this script and the Trainer can never drift out of sync: the subfolder name
(see ``dataset_subdir``) is built from those values, so training with the same
``Config`` finds the chunks automatically. Only the generation-only knobs
(method, chunk_size, resume) live in ``GeneratorCfg`` below.

Progress is recorded in ``progress.json`` after every chunk; re-running resumes
from the last completed chunk instead of restarting.

Edit ``Config`` (and the ``GeneratorCfg`` knobs below) to control generation,
then run from the repo root (foreground):
    python scripts/generate_dataset_3d.py

Run overnight in the background, logging to a file:
    nohup python scripts/generate_dataset_3d.py > cache/dataset_3d/gen.log 2>&1 &

Later, train from these chunks by leaving ``Config`` unchanged; the Trainer
lazy-loads them via ``ChunkedFlowFieldDataset3D`` without filling RAM.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gridflow_nn.training.configs import Config
from gridflow_nn.training.dataloader import FlowFieldDataset3D


@dataclass
class GeneratorCfg:
    method: Literal["eik", "bfs"] = "eik"
    chunk_size: int = 1000
    resume: bool = True


gen = GeneratorCfg()


def main():
    config = Config()
    save_dir = Path(config.dataset_dir_3d) / config.dataset_subdir("3d")
    print(
        f"Generating {config.num_samples} samples "
        f"(grid_size={config.grid_size}, obstacle_prob={config.obstacle_prob}, "
        f"method={gen.method}) -> {save_dir}, "
        f"chunk_size={gen.chunk_size}."
    )

    FlowFieldDataset3D(
        num_samples=config.num_samples,
        grid_size=config.grid_size,
        obstacle_prob=config.obstacle_prob,
        seed=config.seed,
        check_reachable=config.check_reachable,
        method=gen.method,
        save_dir=save_dir,
        chunk_size=gen.chunk_size,
        resume=gen.resume,
    )

    print("Done.")


if __name__ == "__main__":
    main()
