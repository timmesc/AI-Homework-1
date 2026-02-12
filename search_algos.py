"""
search_algos.py - A* search algorithms for Assignment 1
"""

import heapq
import numpy as np


def manhattan_distance(cell, goal):
    """Manhattan distance heuristic"""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def get_neighbors(cell, size):
    """Return valid neighboring cells (N, S, W, E)"""
    row, col = cell
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # N, S, W, E
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return neighbors



def a_star_search(known_grid, start, goal, size, tie_break, counter, g, search_val):
    """
Single A* search from start to goal using agent's current knowledge.
tie_break: 'larger_g' or 'smaller_g'
Returns (path, expanded_count) or (None, expanded_count) if no path.
"""
    # Initialize start state: distance from start to itself is 0
    g[start] = 0
    search_val[start] = counter

    # Initialize goal state: distance unknown (infinity) until we find a path
    g[goal] = float('inf')
    search_val[goal] = counter

    # Tree pointers: tree[cell] = parent cell, used to reconstruct the path
    tree = {}

    # Open list: min-heap of (priority, tiebreaker, cell)
    # Closed set: cells we've already fully explored
    open_list = []
    closed_set = set()

    # Calculate f-value for start: f = g + h = 0 + manhattan_distance
    h = manhattan_distance(start, goal)
    f = h  # since g(start) = 0

    # Push start onto open list with appropriate tie-breaking
    # Python's heapq is a MIN-heap, so smallest value gets popped first
    #   larger_g:  use -g as tiebreaker → among same f, larger g gets popped first
    #   smaller_g: use +g as tiebreaker → among same f, smaller g gets popped first
    if tie_break == 'larger_g':
        heapq.heappush(open_list, (f, -0, start))
    else:
        heapq.heappush(open_list, (f, 0, start))

    expanded = 0

    while open_list:
        # Pop the cell with the smallest priority (lowest f, then tiebreaker)
        f_val, tie_val, s = heapq.heappop(open_list)

        # Skip stale entries: if this cell was already expanded, ignore it
        # (This happens because heapq doesn't support "decrease-key", so we
        #  push duplicates and skip the old ones when we pop them)
        if s in closed_set:
            continue

        # If we just popped the goal, we found the shortest path
        # (The goal does NOT count as "expanded" per assignment)
        if s == goal:
            break

        # Mark this cell as expanded (add to closed set)
        closed_set.add(s)
        expanded += 1

        # Explore all neighbors of the current cell
        for neighbor in get_neighbors(s, size):
            # Skip cells the agent knows are blocked
            if known_grid[neighbor[0], neighbor[1]] == 1:
                continue

            # Skip cells already expanded (they have optimal g-values already)
            if neighbor in closed_set:
                continue

            # Initialize g-value if this cell hasn't been seen in this search
            # (This is the counter/search optimization from the pseudocode)
            if search_val.get(neighbor, 0) < counter:
                g[neighbor] = float('inf')
                search_val[neighbor] = counter

            # If going through s is a shorter path to neighbor, update it
            new_g = g[s] + 1  # all moves cost 1
            if new_g < g[neighbor]:
                g[neighbor] = new_g
                tree[neighbor] = s  # got to neighbor through s

                h = manhattan_distance(neighbor, goal)
                f = new_g + h

                # Push onto open list with tie-breaking priority
                if tie_break == 'larger_g':
                    heapq.heappush(open_list, (f, -new_g, neighbor))
                else:
                    heapq.heappush(open_list, (f, new_g, neighbor))

    # If never found a path to the goal, return None
    if g[goal] == float('inf'):
        return None, expanded

    # Reconstruct path by following tree pointers from goal back to start
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = tree[current]
    path.append(start)
    path.reverse()  # flip it so it goes start → goal

    return path, expanded



def repeated_forward_a_star(full_grid, start, goal, tie_break='larger_g'):
    """
Repeated Forward A*: agent plans with A*, moves along the path,
re-plans when it discovers blocked cells.
Returns (reached_goal, total_expanded, trajectory, num_searches)
"""
    size = full_grid.shape[0]

    # The agent's knowledge grid:
    #   -1 = unknown (treated as unblocked under freespace assumption)
    #    0 = known unblocked
    #    1 = known blocked
    known_grid = np.full((size, size), -1, dtype=np.int8)
    known_grid[start] = 0  # agent knows its starting cell is unblocked
    known_grid[goal] = 0   # agent knows the goal cell is unblocked

    # Efficient g-value management across multiple A* searches
    counter = 0
    g = {}
    search_val = {}

    current = start
    total_expanded = 0
    trajectory = [current]
    num_searches = 0

    while current != goal:
        # Step 1: Observe the 4 adjacent cells from current position
        for n in get_neighbors(current, size):
            known_grid[n[0], n[1]] = full_grid[n[0], n[1]]

        # Step 2: Run A* from current position to goal
        counter += 1
        num_searches += 1
        path, expanded = a_star_search(
            known_grid, current, goal, size, tie_break, counter, g, search_val
        )
        total_expanded += expanded

        # If A* found no path, the target is unreachable
        if path is None:
            return False, total_expanded, trajectory, num_searches

        # Step 3: Follow the path
        for i in range(1, len(path)):
            next_cell = path[i]

            # Move to the next cell on the path
            # (This is safe because we observed from the previous position,
            #  so we know next_cell isn't blocked)
            current = next_cell
            trajectory.append(current)

            # Did we reach the goal?
            if current == goal:
                return True, total_expanded, trajectory, num_searches

            # Observe adjacent cells from new position
            for n in get_neighbors(current, size):
                known_grid[n[0], n[1]] = full_grid[n[0], n[1]]

            # Check if any cell on the REMAINING path is now known to be blocked
            path_blocked = False
            for j in range(i + 1, len(path)):
                if known_grid[path[j][0], path[j][1]] == 1:
                    path_blocked = True
                    break

            # If the path ahead is blocked, stop following it and re-plan
            if path_blocked:
                break

    return True, total_expanded, trajectory, num_searches
