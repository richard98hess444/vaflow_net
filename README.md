# Vecancy Flow Navigation

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
  - `src/gridflow/algorithms/`: BFS, A*, and water-flow algorithms
  - `src/gridflow/utils/`: Plotting and utility helpers
- `scripts/`: Runnable library demos (outside the package)
- `training/`: Neural vector-field training project (outside the package; needs torch)
- `notebooks/`: Jupyter notebooks
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
```

### Training project

Run from the repo root:

```bash
python training/run_trainer.py     # train + visualize, saves a .pt model
python training/run_visualizer.py    # load latest .pt and visualize
```

## Dense Flow Model Update

The training model was changed from a query-based MLP predictor to a dense CNN
predictor.

### What changed

- Old model: encode the full obstacle grid into one latent vector, then predict
  the flow at a queried cell from `[latent, goal, query]`.
- New model: predict the full `(n, n, 2)` vector field directly from the
  obstacle grid and goal using a small encoder-decoder CNN.
- Training loss now compares the predicted full field against the target full
  field, instead of supervising only a random subset of queried cells.
- Checkpoints save an `architecture` field set to `dense_cnn`.

### Why this helped

- Obstacle geometry is preserved through the network instead of being collapsed
  into one global latent.
- Local obstacle effects are easier to learn because the model predicts with
  spatial feature maps.
- Dense supervision makes optimization more stable than random query sampling,
  so training and validation curves are usually smoother.

### Files changed

- `training/modules/vector_field_net.py`: the predictor is a dense CNN.
- `training/trainer.py`: switched training and evaluation from random-query MSE
  to full-field dense MSE.
- `training/configs/config.py`: added `model_channels` for the dense model
  width.
- `training/run_visualizer.py`: loads checkpoints.

### How to train the new model

If you use the `gvf` conda environment:

```bash
conda activate gvf
python training/run_trainer.py
```

This will train the dense model from scratch and save a new checkpoint under
`models/`.

### How to inspect the result

```bash
conda activate gvf
python training/run_visualizer.py
```

The visualizer loads the newest checkpoint by default. It can also load a
specific `.pt` file:

```bash
conda activate gvf
python training/run_visualizer.py models/your_checkpoint.pt
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
