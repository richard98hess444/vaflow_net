"""A simple teleoperated robot simulation on a grid world.

The robot lives in a large world map (default 128x128) and carries a small
local map (default 21x21) that represents what it can scan around itself. The
local map matches the grid size the vector-field model was trained on.

Coordinate convention: (row, col), 0 = free cell, 1 = obstacle.

Because the local map is `local_size` wide (an odd number) and centered on the
robot, the robot/goal must stay far enough from the world border for the local
window to fit. With a 128x128 world and a 21x21 local map, the valid range for
both row and col is [10, 117] (i.e. < 118), which is why goals and start
positions are constrained to "within 118x118".
"""

import random

import numpy as np
import matplotlib.pyplot as plt

from gridflow.algorithms import BreadthFirstSearch
from gridflow.utils import setup_axes, overlay_obstacles_and_source

# 8-neighbor movement deltas, keyed by an arrow / name for convenient teleop.
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "up_left": (-1, -1),
    "up_right": (-1, 1),
    "down_left": (1, -1),
    "down_right": (1, 1),
}


class GridFlowSimulator:
    """Teleoperated robot on a grid world with a rolling local scan map."""

    def __init__(
        self,
        world_size=128,
        local_size=21,
        obstacle_prob=0.2,
        seed=None,
    ):
        if local_size % 2 == 0:
            raise ValueError("local_size must be odd so the robot sits at its center")
        if world_size < local_size:
            raise ValueError("world_size must be at least local_size")

        self.world_size = world_size
        self.local_size = local_size
        self.half = local_size // 2  # distance from robot to local-map edge

        # Valid positions keep the whole local window inside the world.
        self.min_pos = self.half
        self.max_pos = world_size - self.half - 1

        self.bfs = BreadthFirstSearch()
        self.world = self._generate_world(world_size, obstacle_prob, seed)

        self.robot_pos = None  # (row, col)
        self.goal = None  # (row, col)
        self.local_map = None  # (local_size, local_size) numpy array

        # Matplotlib handles for the live local-map view (created lazily).
        self._fig = None
        self._ax = None

        # Pre-trained flow model (loaded on demand via load_model()).
        self.model = None
        self.device = None
        self._torch = None

    # ------------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------------
    def _generate_world(self, size, obstacle_prob, seed):
        """Build a `size`x`size` world from `generate_random_grid`.

        `generate_random_grid` requires an odd grid size, so for an even world
        we generate one cell larger and crop back down to `size`.
        """
        gen_n = size if size % 2 == 1 else size + 1
        grid = self.bfs.generate_random_grid(
            n=gen_n, obstacle_prob=obstacle_prob, seed=seed
        )
        world = np.array(grid, dtype=np.int8)
        return world[:size, :size]

    # ------------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------------
    def _is_valid_pos(self, r, c):
        """True if (r, c) is in range AND not an obstacle."""
        in_range = (
            self.min_pos <= r <= self.max_pos and self.min_pos <= c <= self.max_pos
        )
        return in_range and self.world[r, c] == 0

    def _check_in_range(self, r, c, what):
        if not (
            self.min_pos <= r <= self.max_pos and self.min_pos <= c <= self.max_pos
        ):
            raise ValueError(
                f"{what} ({r}, {c}) out of valid range "
                f"[{self.min_pos}, {self.max_pos}] for both axes."
            )

    # ------------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------------
    def set_goal(self, r, c):
        """Place the goal at (r, c). The cell is forced free so it's reachable."""
        self._check_in_range(r, c, "goal")
        self.world[r, c] = 0  # ensure the goal stands on free ground
        self.goal = (r, c)
        return self.goal

    def generate_goal(self, seed=None):
        """Pick a random free goal within the valid range."""
        rng = random.Random(seed)
        while True:
            r = rng.randint(self.min_pos, self.max_pos)
            c = rng.randint(self.min_pos, self.max_pos)
            if self.world[r, c] == 0:
                self.goal = (r, c)
                return self.goal

    # ------------------------------------------------------------------------
    # Robot placement and movement
    # ------------------------------------------------------------------------
    def set_start_pos(self, r, c):
        """Place the robot at (r, c). The cell is forced free, then we scan."""
        self._check_in_range(r, c, "start position")
        self.world[r, c] = 0  # the robot must stand on free ground
        self.robot_pos = (r, c)
        self._update_local_map()
        return self.robot_pos

    def move(self, dr, dc):
        """Move the robot by (dr, dc) and refresh the local map.

        Returns True if the move succeeded. The move is rejected (position
        unchanged) if it would leave the valid range or step into an obstacle.
        """
        if self.robot_pos is None:
            raise RuntimeError("Call set_start_pos() before move().")

        r, c = self.robot_pos
        nr, nc = r + dr, c + dc

        if not self._is_valid_pos(nr, nc):
            return False

        self.robot_pos = (nr, nc)
        self._update_local_map()
        return True

    def move_dir(self, name, steps=1):
        """Convenience teleop: move by a named direction (see DIRECTIONS)."""
        if name not in DIRECTIONS:
            raise ValueError(
                f"Unknown direction '{name}'. Options: {sorted(DIRECTIONS)}"
            )
        dr, dc = DIRECTIONS[name]
        return self.move(dr * steps, dc * steps)

    # ------------------------------------------------------------------------
    # Local map (robot's scan)
    # ------------------------------------------------------------------------
    def _update_local_map(self):
        """Extract the `local_size`x`local_size` window centered on the robot."""
        r, c = self.robot_pos
        h = self.half
        self.local_map = self.world[r - h : r + h + 1, c - h : c + h + 1].copy()

    def get_local_map(self):
        """Return the current local scan map (robot at the center cell)."""
        if self.local_map is None:
            raise RuntimeError("No local map yet. Call set_start_pos() first.")
        return self.local_map

    def goal_in_local_frame(self):
        """Goal position relative to the local map's top-left corner.

        Returns (row, col) within [0, local_size) if the goal currently falls
        inside the robot's scan window, otherwise None.
        """
        if self.robot_pos is None or self.goal is None:
            return None
        r, c = self.robot_pos
        gr, gc = self.goal
        lr, lc = gr - (r - self.half), gc - (c - self.half)
        if 0 <= lr < self.local_size and 0 <= lc < self.local_size:
            return (lr, lc)
        return None

    def goal_edge_cell(self):
        """Local-frame cell where the goal marker should be drawn.

        If the goal is inside the local window, returns its true local cell.
        Otherwise it projects the robot->goal ray onto the border of the local
        map and returns the cell where that ray crosses the edge. This gives a
        rough "the goal is that way" indicator that slides along the border as
        the robot moves.
        """
        if self.robot_pos is None or self.goal is None:
            return None

        inside = self.goal_in_local_frame()
        if inside is not None:
            return inside

        half = self.half
        last = 2 * half  # max local index (local_size - 1)
        r, c = self.robot_pos
        gr, gc = self.goal
        dy, dx = gr - r, gc - c  # direction from robot (local center) to goal

        # Walk a ray from the center (half, half) and find the smallest t > 0
        # at which it exits the [0, last] x [0, last] square.
        best_t = None
        for delta, edges in ((dy, (0, last)), (dx, (0, last))):
            if delta == 0:
                continue
            for edge in edges:
                t = (edge - half) / delta
                if t <= 0:
                    continue
                row = half + t * dy
                col = half + t * dx
                if -1e-9 <= row <= last + 1e-9 and -1e-9 <= col <= last + 1e-9:
                    best_t = t if best_t is None else min(best_t, t)

        if best_t is None:
            return (0, 0)  # degenerate (e.g. robot already on the goal)

        lr = int(round(half + best_t * dy))
        lc = int(round(half + best_t * dx))
        lr = max(0, min(last, lr))
        lc = max(0, min(last, lc))
        return (lr, lc)

    # ------------------------------------------------------------------------
    # Goal check
    # ------------------------------------------------------------------------
    def reached_goal(self, tol=0):
        """True if the robot is at (or within `tol` cells of) the goal."""
        if self.robot_pos is None or self.goal is None:
            return False
        r, c = self.robot_pos
        gr, gc = self.goal
        return abs(r - gr) <= tol and abs(c - gc) <= tol

    # ------------------------------------------------------------------------
    # Pre-trained flow model
    # ------------------------------------------------------------------------
    def load_model(self, ckpt_path, device=None):
        """Load a trained dense VectorFieldNet checkpoint for local inference.

        The model's grid size must match this simulator's local_size, since we
        feed the local scan map straight into it.
        """
        import os
        import sys
        import torch

        # The training package isn't installed; add it to the path so we can
        # import the model definition.
        training_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "training")
        )
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        from modules import VectorFieldNet

        self._torch = torch
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        ckpt = torch.load(ckpt_path, map_location=self.device)
        config = ckpt["config"]
        model_grid = config.get("grid_size")
        if model_grid is not None and model_grid != self.local_size:
            raise ValueError(
                f"Model grid_size {model_grid} != local_size {self.local_size}; "
                "they must match to run inference on the local map."
            )

        model = VectorFieldNet(base_channels=config.get("model_channels", 32))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        self.model = model
        return model

    def predict_local_flow(self, step=1.0):
        """Run the model on the current local map + local goal.

        The goal handed to the model is `goal_edge_cell()` (the true cell if the
        goal is in view, otherwise its projection onto the local border).

        Args:
            step: spacing between arrows in cell units. 1.0 gives one arrow per
                cell; 0.5 gives a 2x denser grid (2 arrows per cell per axis),
                0.25 a 4x denser grid, etc. Sub-cell points are bilinearly
                resampled from the model's dense output.

        Returns (field, positions) where field is (R, R, 2) of (drow, dcol)
        vectors and positions is (R,) coordinates in cell units, or
        (None, None) if no model/goal/local map is available.
        """
        if self.model is None or self.local_map is None or self.goal is None:
            return None, None

        torch = self._torch
        n = self.local_size
        goal_cell = self.goal_edge_cell()

        grid_t = torch.tensor(self.local_map, dtype=torch.float32)  # (n, n)
        goal_t = torch.tensor(goal_cell, dtype=torch.float32) / (n - 1)  # (2,)

        grid_b = grid_t.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, n, n)
        goal_b = goal_t.unsqueeze(0).to(self.device)  # (1, 2)

        with torch.no_grad():
            dense = self.model(grid_b, goal_b)  # (1, n, n, 2)

            if step == 1.0:
                field = dense.squeeze(0).cpu().numpy()
                return field, np.arange(n)

            import torch.nn.functional as F

            positions = np.arange(0, n - 1 + 1e-9, step)
            res = len(positions)
            coords = [[r, c] for r in positions for c in positions]
            query = torch.tensor(coords, dtype=torch.float32) / (n - 1)
            query = query.to(self.device)

            # Bilinearly resample the dense field at the sub-cell query points.
            dense_chw = dense.permute(0, 3, 1, 2)  # (1, 2, n, n)
            sample_grid = query[..., [1, 0]] * 2.0 - 1.0  # (R*R, 2) as (col, row)
            sample_grid = sample_grid.view(1, res * res, 1, 2)
            sampled = F.grid_sample(
                dense_chw,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (1, 2, R*R, 1)
            pred = sampled.squeeze(-1).squeeze(0).transpose(0, 1)  # (R*R, 2)

        field = pred.reshape(res, res, 2).cpu().numpy()
        return field, positions

    # ------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------
    def render(self, pause=0.001, show_flow=True, step=1.0):
        """Show the current local scan map, styled like `plot_grid`.

        - obstacles are black, the robot (always the center cell) is red,
          and the goal is green.
        - if the goal is outside the local scan window, it is projected onto
          the border of the local map (robot->goal ray intersection), so the
          green cell slides along the edge pointing toward the real goal.
        - if a model is loaded (see load_model) and `show_flow` is True, the
          predicted flow field is overlaid as blue arrows. `step` controls the
          arrow density (1.0 = one per cell, 0.5 = 2x denser, etc.).

        The same figure is reused, so calling this each step animates the
        robot's view in place.
        """
        if self.local_map is None:
            raise RuntimeError("No local map yet. Call set_start_pos() first.")

        n = self.local_size
        local = self.local_map.tolist()

        # Goal in the local frame, or its projection onto the border if it is
        # outside the current scan window.
        local_goal = self.goal_in_local_frame()
        goal_cell = self.goal_edge_cell()

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))

        ax = self._ax
        ax.clear()
        ax.imshow(
            np.zeros((n, n)),
            cmap="gray_r",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        setup_axes(ax, n)
        # `overlay_obstacles_and_source` draws the center cell red (our robot)
        # and the `source` cell green (our goal).
        overlay_obstacles_and_source(ax, local, source=goal_cell)

        # Overlay the predicted flow field as blue arrows.
        flow_shown = False
        if show_flow and self.model is not None:
            field, positions = self.predict_local_flow(step=step)
            if field is not None:
                rows, cols = np.meshgrid(positions, positions, indexing="ij")
                u = field[..., 1]  # dcol -> x (rightward)
                v = field[..., 0]  # drow -> y (downward)
                ax.quiver(
                    cols,
                    rows,
                    u,
                    v,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    color="blue",
                    zorder=4,
                )
                flow_shown = True

        in_view = local_goal is not None
        title = (
            f"robot {self.robot_pos} | goal {self.goal} "
            f"({'in view' if in_view else 'projected to edge'})"
        )
        if flow_shown:
            title += " | predicted flow"
        ax.set_title(title)
        self._fig.canvas.draw_idle()
        plt.pause(pause)

    def keep_open(self):
        """Block until the user closes the figure (press 'q').

        Use this after the goal is reached so the final frame stays on screen
        instead of the program exiting immediately. 'q' is matplotlib's default
        quit shortcut.
        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            return
        plt.ioff()
        plt.show()
