# Vacancy Flow Navigation

## Quick Start

From your workspace:

```bash
# 1. Create virtual environment (one-time)
uv venv env_vfn --python 3.11

# 2. Activate virtual environment
source env_vfn/bin/activate

# 3. Install the library in editable mode (from repo root)
uv pip install -e .

# 4. Run a demo
python scripts/run_flow.py
```

## Project Structure

- `pyproject.toml`: Packaging and dependency configuration
- `src/gridflow/`: The installable library (lean: numpy + matplotlib only)
  - `src/gridflow/algorithms/`: BFS, A*, water-flow, Eikonal, and SDF algorithms
  - `src/gridflow/utils/`: Plotting and utility helpers
- `scripts/`: Runnable library demos (outside the package)
- `training/`: Neural vector-field training project (outside the package; needs torch)
- `models/`: Saved model checkpoints (`.pt`)

## Requirements

- Python `>=3.11`
- Library: `numpy`, `matplotlib`
- Training: `torch` (find the version that matches your device)

## Running Scripts

### Library demos

```bash
python scripts/run_bfs.py
python scripts/run_flow.py
python scripts/run_astar.py
python scripts/run_eikonal.py
python scripts/run_sdf.py
```

### Training

Run from the repo root:

```bash
python training/run_trainer.py     # train + visualize, saves a .pt model
python training/run_visualizer.py    # load latest .pt and visualize
python training/run_visualizer.py models/your_checkpoint.pt    # visualize a pre-trained model
```

### Simulation
```bash
python simulation/run_sim.py -h  # shows all the arguments
python simulation/run_sim.py --model-path models/your_checkpoint.pt --save  # run autonomous mode and save a video
python simulation/run_sim.py --model-path models/your_checkpoint.pt --mode teleo  # run teleoperation
```

### Success rate recording
```bash
python simulation/run_success_rate.py -h  # shows all the arguments
python simulation/run_success_rate.py --model-path models/your_checkpoint.pt
```

### Known caveat: diagonal corner behavior

If the predicted flow appears to pass diagonally around a blocked side cell,
that may come from the target-generation rules, not only from the neural model.
In the current `WaterFlow` implementation, diagonal motion is blocked only when
both adjacent orthogonal cells are blocked. That means the target field can
still permit some diagonal corner-sliding cases, and the model may learn to
imitate them.

## Important Notes

- The library (`src/gridflow`) never imports `torch`; only the `training/` project does.
- Always activate the virtual environment before running commands.
- If you use system Python instead of the venv Python, imports may fail or dependencies may not match.
