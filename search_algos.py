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


def a_star_search(known_grid, start, goal, size, tie_break, counter, g, search_val,
                  h_values=None, return_closed=False):
    """
    Single A* search from start to goal using agent's current knowledge.
    tie_break: 'larger_g' or 'smaller_g'

    Returns:
        - default: (path, expanded_count) or (None, expanded_count) if no path.
        - if return_closed=True: (path, expanded_count, closed_list) or (None, expanded_count, closed_list)
          where closed_list contains the expanded states (goal excluded).
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
    closed_list = []  # keep expansion order for Adaptive A*

    def h_of(cell):
        # If adaptive heuristic table exists and contains this cell, use it.
        # Otherwise fall back to Manhattan distance.
        if h_values is not None and cell in h_values:
            return h_values[cell]
        return manhattan_distance(cell, goal)

    # Calculate f-value for start: f = g + h = 0 + heuristic
    h = h_of(start)
    f = h

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

        # Skip stale entries
        if s in closed_set:
            continue

        # If we just popped the goal, we found the shortest path
        # (The goal does NOT count as "expanded" per assignment)
        if s == goal:
            break

        # Mark this cell as expanded (add to closed set)
        closed_set.add(s)
        closed_list.append(s)
        expanded += 1

        # Explore all neighbors of the current cell
        for neighbor in get_neighbors(s, size):
            # Skip cells the agent knows are blocked
            if known_grid[neighbor[0], neighbor[1]] == 1:
                continue

            # Skip cells already expanded
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

                h = h_of(neighbor)
                f = new_g + h

                # Push onto open list with tie-breaking priority
                if tie_break == 'larger_g':
                    heapq.heappush(open_list, (f, -new_g, neighbor))
                else:
                    heapq.heappush(open_list, (f, new_g, neighbor))

    # If never found a path to the goal, return None
    if g[goal] == float('inf'):
        if return_closed:
            return None, expanded, closed_list
        return None, expanded

    # Reconstruct path by following tree pointers from goal back to start
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = tree[current]
    path.append(start)
    path.reverse()

    if return_closed:
        return path, expanded, closed_list
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
    known_grid[start] = 0
    known_grid[goal] = 0

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


def repeated_backward_a_star(full_grid, start, goal, tie_break='larger_g'):
    """
    Repeated Backward A*: each re-planning runs A* from goal (target)
    to the agent's current cell (current). Then the agent follows the
    reversed path (current -> goal).

    Returns (reached_goal, total_expanded, trajectory, num_searches)
    """
    size = full_grid.shape[0]

    # Agent knowledge:
    # -1 unknown (treated as unblocked), 0 known free, 1 known blocked
    known_grid = np.full((size, size), -1, dtype=np.int8)
    known_grid[start] = 0
    known_grid[goal] = 0

    # Efficient g-value management across searches (same as forward)
    counter = 0
    g = {}
    search_val = {}

    current = start
    total_expanded = 0
    trajectory = [current]
    num_searches = 0

    while current != goal:
        # Observe neighbors from current position
        for n in get_neighbors(current, size):
            known_grid[n[0], n[1]] = full_grid[n[0], n[1]]

        # BACKWARD A*: search from target(goal) -> agent(current)
        counter += 1
        num_searches += 1
        path_goal_to_current, expanded = a_star_search(
            known_grid, goal, current, size, tie_break, counter, g, search_val
        )
        total_expanded += expanded

        if path_goal_to_current is None:
            return False, total_expanded, trajectory, num_searches

        # reverse so agent can move current -> goal
        path = list(reversed(path_goal_to_current))

        # Follow the path until blocked info invalidates the remaining path
        for i in range(1, len(path)):
            next_cell = path[i]
            current = next_cell
            trajectory.append(current)

            if current == goal:
                return True, total_expanded, trajectory, num_searches

            # Observe neighbors from new position
            for n in get_neighbors(current, size):
                known_grid[n[0], n[1]] = full_grid[n[0], n[1]]

            # If remaining path contains a now-known blocked cell, re-plan
            path_blocked = False
            for j in range(i + 1, len(path)):
                if known_grid[path[j][0], path[j][1]] == 1:
                    path_blocked = True
                    break

            if path_blocked:
                break

    return True, total_expanded, trajectory, num_searches


def adaptive_a_star(full_grid, start, goal, tie_break='larger_g'):
    """
    Adaptive A*: repeated planning like repeated forward A*,
    but after each A* search, update h-values for expanded states:
        h(s) = g(goal) - g(s)

    Returns (reached_goal, total_expanded, trajectory, num_searches)
    """
    size = full_grid.shape[0]

    # Agent's knowledge grid:
    #   -1 = unknown (treated as unblocked)
    #    0 = known unblocked
    #    1 = known blocked
    known_grid = np.full((size, size), -1, dtype=np.int8)
    known_grid[start] = 0
    known_grid[goal] = 0

    counter = 0
    g = {}
    search_val = {}

    # Adaptive heuristic table: cell -> learned h
    h_values = {}

    current = start
    total_expanded = 0
    trajectory = [current]
    num_searches = 0

    while current != goal:
        # Step 1: Observe the 4 adjacent cells from current position
        for n in get_neighbors(current, size):
            known_grid[n[0], n[1]] = full_grid[n[0], n[1]]

        # Step 2: Run A* using adaptive heuristic table
        counter += 1
        num_searches += 1
        path, expanded, closed_list = a_star_search(
            known_grid, current, goal, size, tie_break, counter, g, search_val,
            h_values=h_values, return_closed=True
        )
        total_expanded += expanded

        if path is None:
            return False, total_expanded, trajectory, num_searches

        # Step 3: Adaptive update (only for expanded states)
        goal_g = g[goal]
        for s in closed_list:
            h_values[s] = goal_g - g[s]

        # Step 4: Follow the path (same logic as repeated forward A*)
        for i in range(1, len(path)):
            next_cell = path[i]
            current = next_cell
            trajectory.append(current)

            if current == goal:
                return True, total_expanded, trajectory, num_searches

            # Observe from new position
            for n in get_neighbors(current, size):
                known_grid[n[0], n[1]] = full_grid[n[0], n[1]]

            # If remaining path contains a now-known blocked cell, re-plan
            path_blocked = False
            for j in range(i + 1, len(path)):
                if known_grid[path[j][0], path[j][1]] == 1:
                    path_blocked = True
                    break

            if path_blocked:
                break

    return True, total_expanded, trajectory, num_searches
