# Name this file to assignment1.py when you submit
import numpy as np
import heapq
# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath):
    """Return an optimal path, its cost, and the number of explored states.

    A state is (position, collected), where collected is a frozenset of
    treasure coordinates. Different treasure sets at the same position
    are different states.

    Count every non-outdated state removed from the frontier, including
    the successful goal state. This function does not write to any file.
    """
    grid,start,goals,treasures=read_env(filepath)

    optimal_path=[]
    optimal_path_cost=float("inf")
    num_states_explored=0

    start_state=(start,frozenset())
    h=heuristic(start,start_state[1],treasures,goals)

    #each heap item is a tuple (f cost, g cost, state)
    frontier=[]
    heapq.heappush(frontier,(hash,0,start_state))

    bestcost={start_state:0}
    path={start_state:None}

    while frontier:
        #g is cost so far, h is estimated cost to goal, f=g+h
        f,g,state =heapq.heappop(frontier)

        #cheaper route to state might have been found since insertion
        #if cost so far is greater than the best cost, skip this state
        if g>bestcost[state]:
            continue

        #count state as explored, even if not a goal state
        num_states_explored+=1
        position,collected=state
        #check if state is a goal state with enough treasure, if so, record the path and cost
        if position in goals and treasure_val(collected, treasures)>=5:
            optimal_path=reconstruct_path(path, state)
            optimal_path_cost=g
            return optimal_path, optimal_path_cost, num_states_explored
            
        #explore neighbors of current state, add to the frontier if they are new or cheaper than previous ones
        for new_pos in get_neighbors(grid,position):
            new_collected = collect_treasure(new_pos,collected,treasures)
            new_state = (new_pos,new_collected)
            new_g=g+1
            #if new state is not in bestcost or new_g is less than the best cost for that state,
            #update best cost, path, and add to the frontier with new f cost
            if new_state not in bestcost or new_g < bestcost[new_state]:
                bestcost[new_state]=new_g
                path[new_state]=state

                new_f=heuristic(new_pos, new_collected, treasures, goals)
                newf=new_g+new_f
                heapq.heappush(frontier, (newf, new_g, new_state))

    #if no qualifying solution exists, return [], inf, and count.
    return [], float("inf"), num_states_explored




def read_env(filepath):
    #read CSV and identify start, goals, and treasure values

    grid=np.loadtxt(filepath, delimiter=",", dtype=str, ndmin=2)
    rows,columns = grid.shape
    start=None
    goals=set()
    treasures={}

    for row in range(rows):
        for col in range(columns):
            tile=grid[row, col]
            position=(row, col)
            if tile=="S":
                start=position
            elif tile=="G":
                goals.add(position)
            elif tile in ("1", "2", "3", "4", "5"):
                treasures[position]=int(tile)

    return grid, start, goals, treasures



def get_neighbors(grid, position):
    #return legal positions one step in any direction(not diagonal)
    rows, columns=grid.shape
    row, col=position
    neighbors = []
    #check four directions, if the neighbor is within bounds and not a wall, add to neighbors
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nrow = row+dr
        ncol = col+dc
        if not (0 <= nrow < rows and 0 <= ncol < columns):
            continue
        if grid[nrow, ncol] == "X":
            continue

        neighbors.append((nrow, ncol))
    
    
    return neighbors



def collect_treasure(position, collected, treasures):
    #return a frozenset that includes any treasure at this position
    #union does not duplicate an already collected treasure, also it doesnt
    #mutate the parent's frozenset or shared grid
  
    if position in treasures:
        return collected | {position}
    return collected


def treasure_val(collected, treasures):
    #get total value of distinct collected treasures.
    return sum(treasures[position] for position in collected)



def reconstruct_path(path, final_state):
    #Follow parent states backward and return coordinates in path order
    optimal_path = []
    curr = final_state

    while curr is not None:
        position, collected = curr
        optimal_path.append(position)
        curr = path[curr]

    optimal_path.reverse()
    return optimal_path




#heuristic, uses manhattan as the agent is in a grid with 4 direction movement
def heuristic(position, collected, treasures, goals):
    value = sum(treasures[t] for t in collected)
    best_estimate = float("inf")

    if value >= 5:
        for goal in goals:
            estimate = Manhattan(position, goal)
            best_estimate = min(best_estimate, estimate)
    else:
        for treasure in treasures:
            if treasure not in collected:
                for goal in goals:
                    estimate = Manhattan(position, treasure) + Manhattan(treasure, goal)
                    best_estimate = min(best_estimate, estimate)

    return best_estimate

#helper function for manhattan heuristic
def Manhattan(a,b):
  return abs(a[0] - b[0]) + abs(a[1] - b[1])


#NOTE:REMOVE FUNCTION BEFORE SUBMISSION. ONLY FOR TESTING PURPOSES.
def test_examples():
    from pathlib import Path
    import ast

    examples_folder = (
        Path(__file__).resolve().parent / "Examples" / "Examples"
    )
    passed = 0

    for number in range(1, 5):
        folder = examples_folder / f"Example{number}"
        print(f"\n--- Example{number} ---")

        try:
            expected_cost = int(
                (folder / "optimal_path_cost.txt").read_text().strip()
            )
            expected_path = ast.literal_eval(
                (folder / "optimal_path.txt").read_text()
            )

            # Example4 does not include this optional reference file.
            explored_file = folder / "num_states_explored.txt"
            expected_explored = (
                int(explored_file.read_text().strip())
                if explored_file.exists()
                else "not provided"
            )

            grid_file = folder / "grid.txt"
            grid = np.loadtxt(
                grid_file, delimiter=",", dtype=str, ndmin=2
            )
            rows, columns = grid.shape

            optimal_path, optimal_path_cost, num_states_explored = (
                pathfinding(str(grid_file))
            )

            print(f"Cost: {optimal_path_cost} | Expected: {expected_cost}")
            print(
                f"Explored: {num_states_explored} "
                f"| Example count: {expected_explored}"
            )
            print(f"Path: {optimal_path}")

            # Check optimal cost and the number of moves.
            assert optimal_path, "The returned path is empty."
            assert optimal_path_cost == expected_cost, (
                "Incorrect optimal cost."
            )
            assert len(optimal_path) - 1 == optimal_path_cost, (
                "Path length does not match the returned cost."
            )

            # Check coordinates and walls.
            for position in optimal_path:
                assert isinstance(position, tuple) and len(position) == 2, (
                    f"Invalid coordinate: {position}"
                )

                row, col = position
                assert isinstance(row, int) and isinstance(col, int), (
                    f"Coordinates must be integers: {position}"
                )
                assert 0 <= row < rows and 0 <= col < columns, (
                    f"Position {position} is outside the grid."
                )
                assert grid[row, col] != "X", (
                    f"Path enters a wall at {position}."
                )

            # Check the start and end tiles.
            assert grid[optimal_path[0]] == "S", (
                "Path must begin at S."
            )
            assert grid[optimal_path[-1]] == "G", (
                "Path must end at G."
            )

            # Each move must be one horizontal or vertical step.
            for current, following in zip(
                optimal_path, optimal_path[1:]
            ):
                distance = (
                    abs(current[0] - following[0])
                    + abs(current[1] - following[1])
                )
                assert distance == 1, (
                    f"Illegal move: {current} -> {following}"
                )

            # Count treasure at each location only once.
            collected_value = sum(
                int(grid[position])
                for position in set(optimal_path)
                if grid[position] in ("1", "2", "3", "4", "5")
            )
            assert collected_value >= 5, (
                "Collected treasure value is below 5."
            )

            assert isinstance(num_states_explored, int), (
                "The explored-state count must be an integer."
            )
            assert num_states_explored > 0, (
                "The explored-state count must be positive."
            )

            # Different optimal paths and explored counts are allowed.
            if optimal_path != expected_path:
                print("A different valid optimal path was found.")

            print("PASS")
            passed += 1

        except Exception as error:
            print(f"FAIL: {type(error).__name__}: {error}")

    print(f"\nPassed {passed}/4 examples.")
    return passed == 4

if __name__ == "__main__":
    test_examples()