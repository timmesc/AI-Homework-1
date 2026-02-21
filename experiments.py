"""
experiments.py - Run experiments for Part 2 (tie-breaking comparison)
"""

import numpy as np
import time
import random
from GirdWorld import GridWorld
from search_algos import repeated_forward_a_star, repeated_backward_a_star, adaptive_a_star



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






def run_part3():
    print("Part 3: Forward vs Backward (Repeated A*)")
    print("=" * 60)

    results = []

    for i in range(50):
        gw = GridWorld()
        gw.load(f'grids/grid_{i:02d}.pkl')

        # Pick random start and goal (same for both for fairness)
        start = pick_random_unblocked(gw.grid, seed=i * 100)
        goal = pick_random_unblocked(gw.grid, seed=i * 100 + 1)

        while start == goal:
            goal = pick_random_unblocked(gw.grid, seed=i * 100 + random.randint(2, 1000))

        # Forward (must use larger_g tie-breaking for Part 3)
        t0 = time.time()
        reached_f, expanded_f, traj_f, searches_f = repeated_forward_a_star(
            gw.grid, start, goal, tie_break='larger_g'
        )
        time_f = time.time() - t0

        # Backward (same tie-breaking)
        t0 = time.time()
        reached_b, expanded_b, traj_b, searches_b = repeated_backward_a_star(
            gw.grid, start, goal, tie_break='larger_g'
        )
        time_b = time.time() - t0

        results.append({
            'grid': i,
            'forward_expanded': expanded_f,
            'backward_expanded': expanded_b,
            'forward_time': time_f,
            'backward_time': time_b,
            'forward_reached': reached_f,
            'backward_reached': reached_b
        })

        status_f = "reached" if reached_f else "BLOCKED"
        status_b = "reached" if reached_b else "BLOCKED"
        print(f"Grid {i:2d} | Forward:  {expanded_f:6d} expanded ({status_f}) | "
              f"Backward: {expanded_b:6d} expanded ({status_b})")

    # Summary (only compare runs where both reached)
    print()
    print("=" * 60)
    print("SUMMARY (both reached)")
    print("=" * 60)

    both_reached = [r for r in results if r['forward_reached'] and r['backward_reached']]
    if both_reached:
        avg_f = np.mean([r['forward_expanded'] for r in both_reached])
        avg_b = np.mean([r['backward_expanded'] for r in both_reached])
        avg_tf = np.mean([r['forward_time'] for r in both_reached])
        avg_tb = np.mean([r['backward_time'] for r in both_reached])

        print(f"Grids where both reached goal: {len(both_reached)}/50")
        print(f"Avg expanded (Forward):  {avg_f:.1f}")
        print(f"Avg expanded (Backward): {avg_b:.1f}")
        print(f"Avg time (Forward):  {avg_tf:.4f}s")
        print(f"Avg time (Backward): {avg_tb:.4f}s")
    else:
        print("No grids where both algorithms reached the goal.")



#part5
def run_part5():
    print("Part 5: Repeated Forward A* vs Adaptive A*")
    print("=" * 60)

    results = []

    for i in range(50):
        gw = GridWorld()
        gw.load(f'grids/grid_{i:02d}.pkl')

        start = pick_random_unblocked(gw.grid, seed=i * 100)
        goal = pick_random_unblocked(gw.grid, seed=i * 100 + 1)
        while start == goal:
            goal = pick_random_unblocked(gw.grid, seed=i * 100 + random.randint(2, 1000))

        t0 = time.time()
        r_reached, r_exp, _, _ = repeated_forward_a_star(gw.grid, start, goal, tie_break='larger_g')
        r_time = time.time() - t0

        t0 = time.time()
        a_reached, a_exp, _, _ = adaptive_a_star(gw.grid, start, goal, tie_break='larger_g')
        a_time = time.time() - t0

        results.append((r_reached, r_exp, r_time, a_reached, a_exp, a_time))

        print(f"Grid {i:2d} | Repeated: {r_exp:6d} ({'reached' if r_reached else 'BLOCKED'}) | "
              f"Adaptive: {a_exp:6d} ({'reached' if a_reached else 'BLOCKED'})")

    both = [x for x in results if x[0] and x[3]]
    print()
    print("=" * 60)
    print("SUMMARY (both reached)")
    print("=" * 60)
    if both:
        r_avg = np.mean([x[1] for x in both])
        a_avg = np.mean([x[4] for x in both])
        r_t = np.mean([x[2] for x in both])
        a_t = np.mean([x[5] for x in both])
        wins = sum(1 for x in both if x[4] < x[1])

        print(f"Grids where both reached: {len(both)}/50")
        print(f"Avg expanded (Repeated): {r_avg:.1f}")
        print(f"Avg expanded (Adaptive): {a_avg:.1f}")
        print(f"Avg time (Repeated): {r_t:.4f}s")
        print(f"Avg time (Adaptive): {a_t:.4f}s")
        print(f"Adaptive expanded fewer on: {wins} grids")


if __name__ == "__main__":
    #run_part2()

    #run_part3() 
    run_part5()
