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

# 4. Install torch (we have tested on this version)
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128


# 5. Run a demo
python scripts/run_flow.py
```

## Project Structure

The code is split into three installable packages under `src/`, layered by
dependency weight so a consumer only pulls in what it needs:

- `pyproject.toml`: Packaging and dependency configuration
- `src/gridflow/`: Torch-free core library (numpy + matplotlib only)
  - `src/gridflow/algorithms/`: BFS, A*, water-flow, Eikonal, Dijkstra, and SDF algorithms
  - `src/gridflow/utils/`: Plotting and utility helpers
- `src/gridflow_nn/`: Neural model + inference (needs `torch`)
  - `vector_field_net.py`: the `VectorFieldNet` model
  - `inference.py`: `load_model` + `predict_full_field` (lightweight, torch only)
  - `training/`: full training stack — `Trainer`, dataloader, configs, visualizer
    (heavy tier; not imported unless you actually train)
- `src/gridflow_sim/`: Grid-world simulation (needs `torch`)
  - `simulator.py`: `GridFlowSimulator` (world / local-scan / flow-following logic)
  - `render.py`: `VideoRecorder` and rendering helpers (imported explicitly)
  - `configs/`: simulation `Config`
- `scripts/`: All runnable entry points (demos, training, simulation) — outside the packages
- `models/`: Saved model checkpoints (`.pt`)

## Requirements

- Python `>=3.11`
- Core (`gridflow`): `numpy`, `matplotlib`
- Model / training / simulation (`gridflow_nn`, `gridflow_sim`): `torch`
  (find the version that matches your device)

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
python scripts/run_trainer.py     # train + visualize, saves a .pt model
python scripts/run_visualizer.py    # load latest .pt and visualize
python scripts/run_visualizer.py models/your_checkpoint.pt    # visualize a pre-trained model
```

### Simulation
```bash
python scripts/run_sim.py -h  # shows all the arguments
python scripts/run_sim.py --model-path models/your_checkpoint.pt --save  # run autonomous mode and save a video
python scripts/run_sim.py --model-path models/your_checkpoint.pt --mode teleo  # run teleoperation
```

### Success rate recording
```bash
python scripts/run_success_rate.py -h  # shows all the arguments
python scripts/run_success_rate.py --model-path models/your_checkpoint.pt
```

### Known caveat: diagonal corner behavior

If the predicted flow appears to pass diagonally around a blocked side cell,
that may come from the target-generation rules, not only from the neural model.
In the current `WaterFlow` implementation, diagonal motion is blocked only when
both adjacent orthogonal cells are blocked. That means the target field can
still permit some diagonal corner-sliding cases, and the model may learn to
imitate them.

## Important Notes

- The core library (`src/gridflow`) never imports `torch`; only `gridflow_nn` and `gridflow_sim` do.
- Importing `gridflow_nn` for inference does not pull in matplotlib or the training code.
- Always activate the virtual environment before running commands.
- If you use system Python instead of the venv Python, imports may fail or dependencies may not match.
