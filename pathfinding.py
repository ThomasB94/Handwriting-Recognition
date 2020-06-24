from math import sqrt
from collections import defaultdict
import heapq
import time

WHITE = 255
BLACK = 0
# Added to make sure it doesn't explore loads of unnecessary nodes
BANDWIDTH = 200

def compute_heuristic(current, goal):
    dist = [(a - b)**2 for a, b in zip(current, goal)]
    dist = sqrt(sum(dist))
    return dist

def get_neighbours(im, current, boundaries):
  upper = boundaries[0]
  lower = boundaries[1]
  r = current[0]
  c = current[1]

  if r - 1 >= 0 and r - 1 > upper:
    if im[r-1][c] == WHITE:
      # Up
      yield (r-1,c)
    if c + 1 < im.shape[1]:
      if im[r-1][c+1] == WHITE:
        # Diagonal up
        yield (r-1, c+1)

  if c + 1 < im.shape[1]:
    if im[r][c+1] == WHITE:
      # Right
      yield (r, c+1)

  if r + 1 < im.shape[0] and r + 1 < lower:
    if im[r+1][c] == WHITE:
      # Down
      yield (r+1, c)
    if c + 1 < im.shape[1]:
      if im[r+1][c+1] == WHITE:
        # Diagonal down
        yield (r+1, c+1)

def compute_cost(current, neighbour):
  r1 = current[0]
  r2 = neighbour[0]
  c1 = current[1]
  c2 = neighbour[1]
  if r1 != r2 and c1 != c2:
    return 1.4
  else:
    return 1

def generate_path(path, start, goal):
  path_list = []
  current = goal
  while current != start:
    current = path[current]
    path_list.append(current)

  path_list.reverse()
  return path_list

def a_star(im, start, goal):
  boundaries = (start[0]-BANDWIDTH, start[0]+BANDWIDTH)

  hq = [(0, start)]
  costs = {}
  costs = defaultdict(lambda:0, costs)
  path = {}
  t = time.time()

  while hq:
    current_cost, current = heapq.heappop(hq)
    # print("Current node", current)
    if current == goal:
      elapsed = time.time() - t
      print("Dijkstra took:", elapsed)
      path = generate_path(path, start, goal)
      return path

    for neighbour in get_neighbours(im, current, boundaries):
      cost = current_cost + compute_cost(current, neighbour)

      if neighbour not in costs or cost < costs[neighbour]:
        f = cost + compute_heuristic(current, neighbour)
        costs[neighbour] = cost
        path[neighbour] = current
        heapq.heappush(hq, (f, neighbour))

  print("Couldn't find path")
  return path