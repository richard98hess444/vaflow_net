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

# Flow arrow density: 1.0 = one arrow per cell, 0.5 = 2x denser, 0.25 = 4x, etc.
FLOW_STEP = 1.0


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


def _print_status(sim: GridFlowSimulator):
    r, c = sim.robot_pos
    gr, gc = sim.goal
    local_goal = sim.goal_in_local_frame()
    print(f"  robot at ({r}, {c}) | goal at ({gr}, {gc})")
    if local_goal is not None:
        print(f"  goal visible in local map at {local_goal}")
    else:
        print(f"  goal projected to local edge at {sim.goal_edge_cell()}")
    print(f"  reached goal: {sim.reached_goal()}")


def interactive_demo():
    """Drive the robot from the keyboard (WASD + QEZC for diagonals)."""
    sim = GridFlowSimulator(world_size=128, local_size=21, obstacle_prob=0.2, seed=42)
    sim.set_start_pos(20, 20)
    sim.generate_goal(seed=7)

    print(f"Loading flow model from: {os.path.abspath(MODEL_PATH)}")
    sim.load_model(MODEL_PATH)

    print("Teleop keys: w/a/s/d = up/left/down/right, q/e/z/c = diagonals.")
    print("Type a key (optionally repeated, e.g. 'ww') then Enter. 'x' to quit.\n")
    _print_status(sim)
    sim.render(step=FLOW_STEP)

    while True:
        cmd = input("move> ").strip().lower()
        if cmd in ("x", "quit", "exit"):
            break
        if not cmd:
            continue
        key = cmd[0]
        if key not in KEYS:
            print(f"  unknown key '{key}'. Use {sorted(KEYS)} or 'x'.")
            continue

        steps = len(cmd)  # 'ww' moves up twice
        moved_any = False
        for _ in range(steps):
            if sim.move_dir(KEYS[key]):
                moved_any = True
            sim.render(step=FLOW_STEP)  # show the local map after each cell of movement
        if not moved_any:
            print("  (blocked by obstacle or world edge)")

        _print_status(sim)
        if sim.reached_goal():
            print("\nGoal reached! Close the window or press 'q' to exit.")
            sim.keep_open()
            break


if __name__ == "__main__":
    interactive_demo()
