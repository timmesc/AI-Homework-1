"""
experiments.py - Run experiments for Part 2 (tie-breaking comparison)
"""

import numpy as np
import time
import random
from GirdWorld import GridWorld
from search_algos import repeated_forward_a_star


def pick_random_unblocked(grid, seed):
    """Pick a random unblocked cell from the grid"""
    rng = random.Random(seed)
    size = grid.shape[0]
    unblocked = list(zip(*np.where(grid == 0)))
    return rng.choice(unblocked)


def run_part2():
    print("Part 2: Tie-Breaking Comparison (Larger g vs Smaller g)")
    print("=" * 60)

    results = []

    for i in range(50):
        gw = GridWorld()
        gw.load(f'grids/grid_{i:02d}.pkl')

        # Pick random start and goal (same for both so its a fair comparison)
        start = pick_random_unblocked(gw.grid, seed=i * 100)
        goal = pick_random_unblocked(gw.grid, seed=i * 100 + 1)

        while start == goal:
            goal = pick_random_unblocked(gw.grid, seed=i * 100 + random.randint(2, 1000))

        # Run with larger g tie-breaking
        t0 = time.time()
        reached_lg, expanded_lg, traj_lg, searches_lg = repeated_forward_a_star(
            gw.grid, start, goal, tie_break='larger_g'
        )
        time_lg = time.time() - t0

        # Run with smaller g tie-breaking
        t0 = time.time()
        reached_sg, expanded_sg, traj_sg, searches_sg = repeated_forward_a_star(
            gw.grid, start, goal, tie_break='smaller_g'
        )
        time_sg = time.time() - t0

        results.append({
            'grid': i,
            'larger_g_expanded': expanded_lg,
            'smaller_g_expanded': expanded_sg,
            'larger_g_time': time_lg,
            'smaller_g_time': time_sg,
            'larger_g_reached': reached_lg,
            'smaller_g_reached': reached_sg,
        })

        status_lg = "reached" if reached_lg else "BLOCKED"
        status_sg = "reached" if reached_sg else "BLOCKED"
        print(f"Grid {i:2d} | Larger g: {expanded_lg:6d} expanded ({status_lg}) | "
              f"Smaller g: {expanded_sg:6d} expanded ({status_sg})")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    both_reached = [r for r in results if r['larger_g_reached'] and r['smaller_g_reached']]

    if both_reached:
        avg_lg = np.mean([r['larger_g_expanded'] for r in both_reached])
        avg_sg = np.mean([r['smaller_g_expanded'] for r in both_reached])

        lg_wins = sum(1 for r in both_reached if r['larger_g_expanded'] < r['smaller_g_expanded'])
        sg_wins = sum(1 for r in both_reached if r['smaller_g_expanded'] < r['larger_g_expanded'])

        print(f"Grids where both reached goal: {len(both_reached)}/50")
        print(f"Avg expanded (larger g):  {avg_lg:.1f}")
        print(f"Avg expanded (smaller g): {avg_sg:.1f}")
        print(f"Larger g won: {lg_wins} grids")
        print(f"Smaller g won: {sg_wins} grids")

        if avg_lg > 0:
            print(f"Smaller g expands {avg_sg / avg_lg:.2f}x more cells on average")

    unsolvable = [r for r in results if not r['larger_g_reached']]
    if unsolvable:
        print(f"Unsolvable grids: {len(unsolvable)}")


if __name__ == "__main__":
    run_part2()
