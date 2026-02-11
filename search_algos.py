"""
search_algos.py - A* search implementation for Assignment 1

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
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return neighbors


def a_star_search(known_grid, start, goal, size, tie_break, counter, g, search_val):
    """
    Single A* search from start to goal using agent's current knowledge.
    tie_break: 'larger_g' or 'smaller_g'
    """
    g[start] = 0
    search_val[start] = counter
    g[goal] = float('inf')
    search_val[goal] = counter

    tree = {}
    open_list = []
    closed_set = set()

    h = manhattan_distance(start, goal)
    f = h

    if tie_break == 'larger_g':
        heapq.heappush(open_list, (f, -0, start))
    else:
        heapq.heappush(open_list, (f, 0, start))

    expanded = 0

    while open_list:
        f_val, tie_val, s = heapq.heappop(open_list)

        if s in closed_set:
            continue

        if s == goal:
            break

        closed_set.add(s)
        expanded += 1

        for neighbor in get_neighbors(s, size):
            if known_grid[neighbor[0], neighbor[1]] == 1:
                continue

            if neighbor in closed_set:
                continue

            if search_val.get(neighbor, 0) < counter:
                g[neighbor] = float('inf')
                search_val[neighbor] = counter

            new_g = g[s] + 1
            if new_g < g[neighbor]:
                g[neighbor] = new_g
                tree[neighbor] = s

                h = manhattan_distance(neighbor, goal)
                f = new_g + h

                if tie_break == 'larger_g':
                    heapq.heappush(open_list, (f, -new_g, neighbor))
                else:
                    heapq.heappush(open_list, (f, new_g, neighbor))

    if g[goal] == float('inf'):
        return None, expanded

    # reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = tree[current]
    path.append(start)
    path.reverse()

    return path, expanded


# TODO: implement repeated_forward_a_star (agent loop)
