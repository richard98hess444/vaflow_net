"""Main script for BFS algorithm demonstration."""

from gridflow.algorithms import BreadthFirstSearch
from gridflow.utils import print_grid, print_flow, plot_grid, plot_flow

def main():
    """Run BFS algorithm on a sample grid."""
    bfs = BreadthFirstSearch()

    # Generate or define a grid
    n = 9
    grid = bfs.generate_random_grid(
        n=n,
        obstacle_prob=0.2,
        seed=42,
    )

    # Compute flow
    flow = bfs.compute_flow_map(grid)

    # Print results
    print("Obstacle grid (0=free, 1=obstacle):")
    print_grid(grid)

    print("\nFlow map (S=source, #=obstacle, X=unreachable):")
    print_flow(flow)

    # Plot results
    print("\nPlotting grid...")
    plot_grid(grid)

    print("Plotting flow...")
    plot_flow(flow)


if __name__ == "__main__":
    main()
