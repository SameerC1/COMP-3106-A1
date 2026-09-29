# Name this file to assignment1.py when you submit
import numpy as np
# The pathfinding function must implement A* search to find the goal state
def pathfinding(filepath):
  # filepath is the path to a CSV file containing a grid 
  # optimal_path is a list of coordinate of squares visited (in order)
  # optimal_path_cost is the cost of the optimal path
  # num_states_explored is the number of states explored during A* search


  optimal_path = []
  optimal_path_cost = float("inf")
  num_states_explored = 0

  grid = np.loadtxt(filepath, delimiter=",", dtype=str, ndmin=2)
  rows,columns=grid.shape
  start=None
  goals=set()
  treasures={}
  #the environment is fully observable
  #search the grid and map out goals, treasures, and the startpoint
  for row in range(rows):
    for col in range(columns):
      tile=grid[row,col]
      position=(row,col)
      if tile=="S":
        start=position
      elif tile=="G":
        goals.add(position)
      elif tile in ("1", "2", "3", "4", "5"):
        treasures[position]=int(tile)
  #set starting state
  start_state = (start, frozenset())
  
    #TODO: Algorithm

  return optimal_path, optimal_path_cost, num_states_explored

#Heuristic, uses manhattan as the agent is in a grid with 4 direction movement
def heuristic(position, collected, treasures, goals):
  value=sum(treasures[t] for t in collected)
  best_estimate=float("inf")

  if value>=5:
    for goal in goals:
      estimate=Manhattan(position,goal)
      best_estimate=min(best_estimate,estimate)

  else:
    for treasure in treasures:
      if treasure not in collected:
        for goal in goals:
          estimate=Manhattan(position,treasure)+Manhattan(treasure,goal)
          best_estimate=min(best_estimate,estimate)

  return best_estimate
#helper function for manhattan heuristic
def Manhattan(a,b):
  return abs(a[0] - b[0]) + abs(a[1] - b[1])