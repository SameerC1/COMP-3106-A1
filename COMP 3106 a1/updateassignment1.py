
"""A* pathfinding on a grid with a minimum treasure-value requirement."""

import heapq

import numpy as np


def pathfinding(filepath):
    """Return the optimal path, its cost, and the number of explored states.

    Each state contains the current position and a frozenset of collected
    treasure coordinates, so a treasure is collected at most once.
    Explored states include the successful goal state.
    """
    grid, start, goals, treasures = read_env(filepath)

    start_state = (start, frozenset())
    initial_h = heuristic(start, start_state[1], treasures, goals)

    # Priority-queue entries contain (f_cost, g_cost, state).
    frontier = [(initial_h, 0, start_state)]
    bestcost = {start_state: 0}
    parents = {start_state: None}
    num_states_explored = 0

    while frontier:
        f, g, state = heapq.heappop(frontier)

        # A cheaper route to this state may have been discovered later.
        if g > bestcost[state]:
            continue

        num_states_explored += 1
        position, collected = state

        # The agent must both reach a goal and collect treasure worth >= 5.
        if position in goals and treasure_val(collected, treasures) >= 5:
            optimal_path = reconstruct_path(parents, state)
            return optimal_path, g, num_states_explored

        for new_pos in get_neighbors(grid, position):
            new_collected = collect_treasure(new_pos, collected, treasures)
            new_state = (new_pos, new_collected)
            new_g = g + 1

            # Only insert newly discovered states or cheaper paths.
            if new_g < bestcost.get(new_state, float("inf")):
                bestcost[new_state] = new_g
                parents[new_state] = state
                new_h = heuristic(new_pos, new_collected, treasures, goals)
                heapq.heappush(frontier, (new_g + new_h, new_g, new_state))

    # The assignment guarantees a valid solution exists.
    return [], float("inf"), num_states_explored


def read_env(filepath):
    """Read the CSV and locate the start, goals, and treasures."""
    grid = np.loadtxt(filepath, delimiter=",", dtype=str, ndmin=2)
    rows, columns = grid.shape
    start = None
    goals = set()
    treasures = {}

    for row in range(rows):
        for col in range(columns):
            tile = grid[row, col].strip()
            position = (row, col)
            if tile == "S":
                start = position
            elif tile == "G":
                goals.add(position)
            elif tile in ("1", "2", "3", "4", "5"):
                treasures[position] = int(tile)

    return grid, start, goals, treasures


def get_neighbors(grid, position):
    """Return in-bounds, non-wall neighbours (no diagonal moves)."""
    rows, columns = grid.shape
    row, col = position
    neighbors = []

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nrow = row + dr
        ncol = col + dc
        if not (0 <= nrow < rows and 0 <= ncol < columns):
            continue
        if grid[nrow, ncol].strip() == "X":
            continue
        neighbors.append((nrow, ncol))

    return neighbors


def collect_treasure(position, collected, treasures):
    """Return the updated immutable set of collected treasure locations."""
    if position in treasures:
        return collected | {position}
    return collected


def treasure_val(collected, treasures):
    """Sum the values of distinct collected treasures."""
    return sum(treasures[position] for position in collected)


def reconstruct_path(parents, final_state):
    """Follow parent links backward, then return coordinates in path order."""
    optimal_path = []
    current = final_state
    while current is not None:
        position, _ = current
        optimal_path.append(position)
        current = parents[current]
    optimal_path.reverse()
    return optimal_path


def heuristic(position, collected, treasures, goals):
    """Lower bound via Manhattan distances to a treasure and/or goal."""
    value = treasure_val(collected, treasures)
    best_estimate = float("inf")

    if value >= 5:
        for goal in goals:
            best_estimate = min(best_estimate, Manhattan(position, goal))
    else:
        for treasure in treasures:
            if treasure not in collected:
                for goal in goals:
                    estimate = (
                        Manhattan(position, treasure)
                        + Manhattan(treasure, goal)
                    )
                    best_estimate = min(best_estimate, estimate)

    return best_estimate


def Manhattan(a, b):
    """Distance between two grid positions using four-direction moves."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
