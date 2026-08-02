"""Generate a large 3D flow-field dataset to disk in memory-safe chunks.

Every ``--chunk-size`` accepted samples are written to ``--save-dir`` as
``{start}-{end}.npz`` (e.g. ``0-999.npz``) and then freed from RAM, so peak
memory stays bounded by one chunk regardless of ``--num-samples``. Progress is
recorded in ``progress.json`` after every chunk; re-running with the same
``--save-dir`` resumes from the last completed chunk instead of restarting.

Run from the repo root (foreground):
    python scripts/generate_dataset_3d.py

Run overnight in the background, logging to a file:
    nohup python scripts/generate_dataset_3d.py > cache/dataset_3d/gen.log 2>&1 &

Later, train from the chunks without loading everything into RAM:
    from gridflow_nn.training import ChunkedFlowFieldDataset3D
    ds = ChunkedFlowFieldDataset3D("cache/dataset_3d")
"""

import argparse

from gridflow_nn.training.dataloader import FlowFieldDataset3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--obstacle-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["eik", "bfs"], default="eik")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--save-dir", default="cache/dataset_3d")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing chunks/progress in --save-dir.",
    )
    args = parser.parse_args()

    print(
        f"Generating {args.num_samples} samples "
        f"(grid_size={args.grid_size}, obstacle_prob={args.obstacle_prob}, "
        f"method={args.method}) -> {args.save_dir}, "
        f"chunk_size={args.chunk_size}."
    )

    # Streaming mode: samples are flushed to disk and freed every chunk, so this
    # returns an (essentially empty) object; the data lives in the chunk files.
    FlowFieldDataset3D(
        num_samples=args.num_samples,
        grid_size=args.grid_size,
        obstacle_prob=args.obstacle_prob,
        seed=args.seed,
        method=args.method,
        save_dir=args.save_dir,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
    )

    print("Done.")


if __name__ == "__main__":
    main()
