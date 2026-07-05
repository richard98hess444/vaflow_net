"""Teleoperated demo of the GridFlowSimulator.

Builds a 128x128 world, drops the robot somewhere valid, sets a goal, and lets
you drive the robot with keyboard-style commands. After each move it reports
the robot position, whether the goal is in view, and whether the goal is
reached, and redraws the robot's local scan map.

Run from the repo root (with the venv active):
    python simulation/run_sim.py
"""

import os

from grid_flow_simulator import GridFlowSimulator

# Pre-trained dense flow model used to predict the local flow field.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")


class FlowSimRunner:
    """Keyboard-driven teleoperation session for a GridFlowSimulator."""

    # Single-character teleop keys -> direction names.
    KEYS = {
        "w": "up",
        "s": "down",
        "a": "left",
        "d": "right",
        "q": "up_left",
        "e": "up_right",
        "z": "down_left",
        "c": "down_right",
    }

    def __init__(
        self,
        model_path=MODEL_PATH,
        world_size=128,
        local_size=21,
        obstacle_prob=0.2,
        grid_seed=42,
        start_pos=(20, 20),
        goal_seed=6,
        flow_step=0.5,
        flow_plot_mode="compare",
        flow_method="eikonal",
    ):
        self.model_path = model_path
        self.start_pos = start_pos
        self.goal_seed = goal_seed
        # Flow arrow density: 1.0 = one arrow per cell, 0.5 = 2x denser, etc.
        self.flow_step = flow_step
        # Render mode: "overlay" (single map) or "compare" (ground truth vs pred).
        self.flow_plot_mode = flow_plot_mode
        # Flow computation method: "bfs" or "eikonal".
        self.flow_method = flow_method

        self.sim = GridFlowSimulator(
            world_size=world_size,
            local_size=local_size,
            obstacle_prob=obstacle_prob,
            grid_seed=grid_seed,
        )

    def setup(self):
        self.sim.set_start_pos(*self.start_pos)
        self.sim.generate_goal(seed=self.goal_seed)

        print(f"Loading flow model from: {os.path.abspath(self.model_path)}")
        self.sim.load_model(self.model_path)

    def print_status(self):
        sim = self.sim
        if sim.robot_pos is None or sim.goal is None:
            raise RuntimeError(
                "Robot position and goal must be set before status print"
            )

        robot_r, robot_c = sim.robot_pos
        goal_r, goal_c = sim.goal
        local_goal = sim.goal_in_local_frame()
        print(f"  robot at ({robot_r}, {robot_c}) | goal at ({goal_r}, {goal_c})")
        if local_goal is not None:
            print(f"  goal visible in local map at {local_goal}")
        else:
            print(f"  goal projected to local edge at {sim.goal_edge_cell()}")

        center = sim.local_size // 2
        pred_field, _ = sim.predict_local_flow(step=1.0)
        if pred_field is not None:
            pred_vec = pred_field[center, center]
            print(f"  pred flow (drow, dcol): ({pred_vec[0]:.4f}, {pred_vec[1]:.4f})")

        target_field = None
        if self.flow_method == "bfs":
            target_field, _ = sim.bfs_gt_local_flow()
        elif self.flow_method == "eikonal":
            target_field, _ = sim.eik_gt_local_flow()
        
        if target_field is not None:
            target_vec = target_field[center, center]
            print(f"  gt flow (drow, dcol): ({target_vec[0]:.4f}, {target_vec[1]:.4f})")

        print(f"  reached goal: {sim.reached_goal()}")

    def render(self):
        self.sim.render(step=self.flow_step, plot_mode=self.flow_plot_mode, method=self.flow_method)
        
    def ang_2_cmd(self, degree):
        """Convert a flow angle (degrees) to a teleop command string."""
        if degree < -157.5 or degree >= 157.5:
            return "a"
        elif -157.5 <= degree < -112.5:
            return "z"
        elif -112.5 <= degree < -67.5:
            return "s"
        elif -67.5 <= degree < -22.5:
            return "c"
        elif -22.5 <= degree < 22.5:
            return "d"
        elif 22.5 <= degree < 67.5:
            return "e"
        elif 67.5 <= degree < 112.5:
            return "w"
        elif 112.5 <= degree < 157.5:
            return "q"

    def run_teleopertation(self):
        """Drive the robot from the keyboard (WASD + QEZC for diagonals)."""
        self.setup()

        print("Teleop keys: w/a/s/d = up/left/down/right, q/e/z/c = diagonals.")
        print("Type a key (optionally repeated, e.g. 'ww') then Enter. 'x' to quit.\n")
        self.print_status()
        self.render()

        while True:
            cmd = input("move> ").strip().lower()
            if cmd in ("x", "quit", "exit"):
                break
            if not cmd:
                continue

            key = cmd[0]
            if key not in self.KEYS:
                print(f"  unknown key '{key}'. Use {sorted(self.KEYS)} or 'x'.")
                continue

            moved_any = False
            if self.sim.move_dir(self.KEYS[key]):
                moved_any = True
            self.render()  # show the local map after each cell of movement
            if not moved_any:
                print("  (blocked by obstacle or world edge)")

            self.print_status()
            if self.sim.reached_goal():
                print("\nGoal reached! Close the window or press 'q' to exit.")
                self.sim.keep_open()
                break

    def run_autonomous(self):
        """Run the robot autonomously to the goal using the learned flow model."""
        self.setup()

        print("Autonomous mode: robot will follow the predicted flow field.")
        self.print_status()
        self.render()
        
        print("Press Enter to start autonomous navigation, or 'x' to quit.")
        cmd = input("start> ").strip().lower()
        if cmd in ("x", "quit", "exit"):
            return

        while not self.sim.reached_goal():
            moved_any = False
            pred_field, _ = self.sim.predict_local_flow(step=1.0)
            if pred_field is not None:
                center = self.sim.local_size // 2
                pred_vec = pred_field[center, center]
                degree = np.arctan2(-pred_vec[0], pred_vec[1]) * 180 / np.pi
                cmd = self.ang_2_cmd(degree)
                if self.sim.move_dir(self.KEYS[cmd]):
                    moved_any = True
                    print(f"Moved '{cmd}' (angle: {degree:.4f} deg)")
                else:
                    print(f"Blocked when trying to move '{cmd}' (angle: {degree:.4f} deg)")

            self.render()  # show the local map after each cell of movement
            if not moved_any:
                print("  (blocked by obstacle or world edge)")

            self.print_status()

        print("\nGoal reached! Close the window or press 'q' to exit.")
        self.sim.keep_open()

def main(argv=None):

if __name__ == "__main__":
    FlowSimRunner().run()
