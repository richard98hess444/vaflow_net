"""Demo script for the classic A* algorithm.

Shows the obstacle grid, then the same grid with the A* path overlaid.
"""

from gridflow.algorithms.astar import AStar
from gridflow.utils import plot_grid, plot_path_on_thresholded_grid


def main():
    n = 21

    astar = AStar(heuristic_weight=1.0)

    grid = astar.generate_random_grid(
        n=n,
        obstacle_prob=0.2,
        seed=1,
    )

    # start at the center (drawn red), goal at a corner (drawn green)
    start = (n // 2, n // 2)
    goal = (0, n - 1)

    # ensure start and goal are free
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    path = astar.compute_path(grid, start, goal)

    if path:
        print(f"Path found with {len(path)} steps.")
    else:
        print("No path found.")

    # 1) show the grid map
    plot_grid(grid, source=goal)

    # 2) show the grid with the path on it
    #    (center=red=start, source=green=goal, orange dots/line=path)
    plot_path_on_thresholded_grid(path, grid, source=goal)


if __name__ == "__main__":
    main()