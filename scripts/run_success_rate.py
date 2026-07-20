"""Headless success-rate benchmark for the autonomous flow-following robot.

For each sample a random (grid_seed, goal_seed) pair in [seed_min, seed_max] is
drawn (pairs are unique). A fresh world + goal are built and the robot follows
the learned flow field toward the goal with NO rendering. We record whether the
robot reaches the goal and write the per-sample results plus the overall success
rate to a CSV file.

Run from the repo root (with the venv active):
    python scripts/run_success_rate.py
"""

import argparse
import csv
import glob
import os
import random

import numpy as np

from gridflow_sim import GridFlowSimulator
from gridflow_sim.configs import Config
from run_sim import FlowSimRunner

# Directory holding the trained checkpoints; the default model is the most
# recently modified *.pt file in here (resolved at runtime).
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DEFAULT_CSV_PATH = os.path.join(
    REPO_ROOT, "outputs", "success_rate", "success_rate.csv"
)


def latest_model_path(models_dir=MODELS_DIR):
    """Return the most recently modified *.pt checkpoint in `models_dir`."""
    checkpoints = glob.glob(os.path.join(models_dir, "*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No *.pt checkpoints found in {os.path.abspath(models_dir)}"
        )
    return max(checkpoints, key=os.path.getmtime)


def sample_seed_pairs(num_samples, seed_min, seed_max, rng):
    """Draw `num_samples` unique (grid_seed, goal_seed) pairs in range."""
    span = seed_max - seed_min + 1
    max_pairs = span * span
    if num_samples > max_pairs:
        raise ValueError(
            f"Requested {num_samples} unique pairs but only {max_pairs} exist "
            f"for seeds in [{seed_min}, {seed_max}]."
        )
    pairs = set()
    while len(pairs) < num_samples:
        g = rng.randint(seed_min, seed_max)
        s = rng.randint(seed_min, seed_max)
        pairs.add((g, s))
    return list(pairs)


def run_episode(
    grid_seed,
    goal_seed,
    model,
    device,
    start_pos,
    max_steps,
    world_size,
    local_size,
    obstacle_prob,
):
    """Run a single autonomous episode headlessly; return True on success."""
    sim = GridFlowSimulator(
        world_size=world_size,
        local_size=local_size,
        obstacle_prob=obstacle_prob,
        grid_seed=grid_seed,
    )
    sim.set_start_pos(*start_pos)
    sim.generate_goal(seed=goal_seed)
    sim.model = model
    sim.device = device

    center = sim.local_size // 2
    steps = 0
    while not sim.reached_goal() and steps < max_steps:
        steps += 1
        pred_field, _ = sim.predict_local_flow(step=1.0)
        if pred_field is None:
            break
        vec = pred_field[center, center]
        degree = np.arctan2(-vec[0], vec[1]) * 180 / np.pi
        cmd = FlowSimRunner.ang_2_cmd(degree)
        sim.move_dir(FlowSimRunner.KEYS[cmd])
    return sim.reached_goal()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Headless success-rate benchmark for the autonomous robot."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model checkpoint to use (default: newest *.pt in models/).",
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=100)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for drawing the (grid_seed, goal_seed) pairs (reproducible).",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--world-size", type=int, default=128)
    parser.add_argument("--local-size", type=int, default=21)
    parser.add_argument("--obstacle-prob", type=float, default=0.2)
    parser.add_argument("--start-row", type=int, default=20)
    parser.add_argument("--start-col", type=int, default=20)
    cli = parser.parse_args(argv)

    model_path = cli.model_path or latest_model_path()

    config = Config()

    # Load the model once (into a throwaway sim) and reuse it for every sample.
    bootstrap = GridFlowSimulator(
        local_size=config.local_size,
        world_size=config.world_size,
        obstacle_prob=cli.obstacle_prob,
        grid_seed=cli.seed_min,
    )
    print(f"Loading flow model from: {os.path.abspath(model_path)}")
    model = bootstrap.load_model(model_path)
    device = bootstrap.device

    rng = random.Random(cli.sample_seed)
    pairs = sample_seed_pairs(cli.num_samples, cli.seed_min, cli.seed_max, rng)

    results = []
    num_success = 0
    for i, (grid_seed, goal_seed) in enumerate(pairs, start=1):
        success = run_episode(
            grid_seed=grid_seed,
            goal_seed=goal_seed,
            model=model,
            device=device,
            start_pos=(cli.start_row, cli.start_col),
            max_steps=cli.max_steps,
            world_size=cli.world_size,
            local_size=cli.local_size,
            obstacle_prob=cli.obstacle_prob,
        )
        num_success += int(success)
        results.append((grid_seed, goal_seed, int(success)))
        if i % 50 == 0 or i == len(pairs):
            print(
                f"  [{i}/{len(pairs)}] running success rate: " f"{num_success / i:.4f}"
            )

    success_rate = num_success / len(pairs) if pairs else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(cli.csv_path)), exist_ok=True)
    with open(cli.csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_seed", "goal_seed", "success"])
        for grid_seed, goal_seed, success in results:
            writer.writerow([grid_seed, goal_seed, success])
        # Summary section.
        writer.writerow([])
        writer.writerow(["successes", num_success])
        writer.writerow(["total", len(pairs)])
        writer.writerow(["success_rate", f"{success_rate:.4f}"])

    print(
        f"\nDone. {num_success}/{len(pairs)} reached the goal "
        f"(success rate: {success_rate:.4f})."
    )
    print(f"Results written to: {os.path.abspath(cli.csv_path)}")


if __name__ == "__main__":
    main()
