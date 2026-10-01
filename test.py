# Name this file assignment1.py when you submit

import numpy as np
import heapq


# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath):

    # Required outputs
    optimal_path = []
    optimal_path_cost = float("inf")
    num_states_explored = 0

    # Load grid
    grid = np.loadtxt(filepath, delimiter=",", dtype=str, ndmin=2)

    rows, columns = grid.shape

    start = None
    goals = set()
    treasures = {}

    # Search the grid and map out:
    # start position, goals, and treasures
    for row in range(rows):
        for col in range(columns):

            tile = grid[row, col]
            position = (row, col)

            if tile == "S":
                start = position

            elif tile == "G":
                goals.add(position)

            elif tile in ("1", "2", "3", "4", "5"):
                treasures[position] = int(tile)


    # ---------------------------------------------------------
    # A* SETUP
    # ---------------------------------------------------------

    # State is:
    # (current position, treasures already collected)
    start_state = (start, frozenset())

    # Priority queue
    frontier = []

    g = 0
    h = heuristic(start, frozenset(), treasures, goals)
    f = g + h

    # Each frontier item:
    # (f cost, g cost, state)
    heapq.heappush(
        frontier,
        (f, g, start_state)
    )


    # Best known g-cost for each state
    best_cost = {
        start_state: 0
    }


    # Used to reconstruct the final path
    came_from = {
        start_state: None
    }


    # Four legal movement directions:
    # up, down, left, right
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]


    # ---------------------------------------------------------
    # A* SEARCH
    # ---------------------------------------------------------

    while frontier:

        # Pop state with smallest f value
        f, g, state = heapq.heappop(frontier)

        position, collected = state


        # If this heap entry is outdated because we have already
        # found a cheaper path to the same state, skip it
        if g > best_cost[state]:
            continue


        num_states_explored += 1


        # -----------------------------------------------------
        # GOAL TEST
        # -----------------------------------------------------

        treasure_value = sum(
            treasures[t] for t in collected
        )

        # Valid solution:
        # - standing on a goal
        # - collected treasure worth at least 5
        if position in goals and treasure_value >= 5:

            optimal_path_cost = g


            # Reconstruct path backwards using came_from
            current = state

            while current is not None:

                current_position, current_collected = current

                optimal_path.append(current_position)

                current = came_from[current]


            # Path was reconstructed backwards
            optimal_path.reverse()

            break


        # -----------------------------------------------------
        # GENERATE SUCCESSORS
        # -----------------------------------------------------

        row, col = position


        for dr, dc in directions:

            new_row = row + dr
            new_col = col + dc


            # ---------------------------------------------
            # CHECK GRID BOUNDARIES
            # ---------------------------------------------

            if not (
                0 <= new_row < rows
                and
                0 <= new_col < columns
            ):
                continue


            new_position = (new_row, new_col)


            # ---------------------------------------------
            # CHECK WALL
            # ---------------------------------------------

            if grid[new_row, new_col] == "X":
                continue


            # ---------------------------------------------
            # TREASURE COLLECTION
            # ---------------------------------------------

            new_collected = collected

            if new_position in treasures:

                new_collected = (
                    collected | {new_position}
                )


            # ---------------------------------------------
            # CREATE NEW STATE
            # ---------------------------------------------

            new_state = (
                new_position,
                new_collected
            )


            # Every move costs 1
            new_g = g + 1


            # ---------------------------------------------
            # CHECK WHETHER THIS PATH IS BETTER
            # ---------------------------------------------

            if (
                new_state not in best_cost
                or
                new_g < best_cost[new_state]
            ):

                best_cost[new_state] = new_g


                # Remember where we came from
                came_from[new_state] = state


                # Calculate heuristic
                new_h = heuristic(
                    new_position,
                    new_collected,
                    treasures,
                    goals
                )


                # A* priority
                new_f = new_g + new_h


                # Add successor to frontier
                heapq.heappush(
                    frontier,
                    (
                        new_f,
                        new_g,
                        new_state
                    )
                )


    return (
        optimal_path,
        optimal_path_cost,
        num_states_explored
    )





# ---------------------------------------------------------
# HEURISTIC
# ---------------------------------------------------------

def heuristic(position, collected, treasures, goals):

    # Current total value of collected treasure
    value = sum(
        treasures[t] for t in collected
    )

    best_estimate = float("inf")


    # If we already have enough treasure,
    # estimate distance to closest goal
    if value >= 5:

        for goal in goals:

            estimate = Manhattan(
                position,
                goal
            )

            best_estimate = min(
                best_estimate,
                estimate
            )


    # Otherwise estimate:
    # current -> treasure -> goal
    else:

        for treasure in treasures:

            if treasure not in collected:

                for goal in goals:

                    estimate = (
                        Manhattan(
                            position,
                            treasure
                        )
                        +
                        Manhattan(
                            treasure,
                            goal
                        )
                    )

                    best_estimate = min(
                        best_estimate,
                        estimate
                    )


    return best_estimate





# ---------------------------------------------------------
# MANHATTAN DISTANCE
# ---------------------------------------------------------

def Manhattan(a, b):

    return (
        abs(a[0] - b[0])
        +
        abs(a[1] - b[1])
    )