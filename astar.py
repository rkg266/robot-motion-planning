
# priority queue for OPEN list
from pqdict import pqdict
import numpy as np
import math

class AStarNode(object):
  def __init__(self, pqkey, coord, hval):
    self.pqkey = pqkey
    self.coord = coord
    self.g = math.inf
    self.h = hval
    self.parent_node = None
    self.parent_action = None
    self.closed = False
  def __lt__(self, other):
    return self.g < other.g     


class AStar(object):
  @staticmethod
  def plan(environment, epsilon = 1):
    # Initialize the graph and open list
    Graph = {}
    OPEN = pqdict()

    # All possible directions to move in grid
    num_dir = 26
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1) # remove [0,0,0]

    start_coord = environment.grid_start
    goal_coord = environment.grid_goal
    
    start_node = AStarNode(tuple(start_coord), np.array(start_coord), AStar.getHeuristicEuclidean(start_coord, goal_coord))
    start_node.g = 0
    Graph[start_node.pqkey] = start_node # start node added to graph
    goal_node = AStarNode(tuple(goal_coord), np.array(goal_coord), 0)
    goal_node.g = float('inf')
    Graph[goal_node.pqkey] = goal_node # goal node added to graph

    # Add start node to OPEN
    OPEN.additem(start_node.pqkey, start_node.g + start_node.h)

    while not goal_node.closed:
      curr_key = OPEN.popitem()
      curr = Graph[curr_key[0]]
      curr.closed = True
      for r in range(num_dir):
        next_coord = (curr.coord + dR[:, r]).astype('int16') 
        if not AStar.isAllowed(environment, curr.coord, next_coord):
          continue
        if tuple(next_coord) not in Graph.keys(): # if child not already present in graph
          child = AStarNode(tuple(next_coord), np.array(next_coord), AStar.getHeuristicEuclidean(next_coord, goal_coord))
          child.g = float('inf')
          Graph[child.pqkey] = child
        else:
          child = Graph[tuple(next_coord)]
        if child.closed: 
          continue
        cost = np.linalg.norm(dR[:, r])
        if child.g > curr.g + cost:
          child.g = curr.g + cost
          child.parent_node = curr
          child.parent_action = dR[:, r]
          # if child in OPEN, update priority else add to OPEN
          if child.pqkey in OPEN:
            OPEN.updateitem(child.pqkey, child.g + epsilon*child.h)
          else:
            OPEN.additem(child.pqkey, child.g + epsilon*child.h)
          # if child.pqkey == goal_node.pqkey:
          #   bu=4

    # Trace the path
    curr = goal_node
    path = []
    path.append(curr.coord)
    while curr.parent_node is not None:
      curr = curr.parent_node
      path.append(curr.coord)
    return path
      


  @staticmethod
  def isAllowed(environment, parent_coord, child_coord):
    grid_size = environment.grid_size
    #blocks = environment.grid_aabbs
    blocks = environment.blocks
    boundary = environment.boundary.flatten()

    for i in range(3):
      if child_coord[i] < 0 or child_coord[i] > grid_size[i]-1:
        return False
    
    res = environment.res
    min_bound = boundary[0:3]
    for aabb in blocks:
      aabb_min = aabb[0:3]
      aabb_max = aabb[3:6]
      if AStar.check_intersection_segment_block(min_bound+res*parent_coord, min_bound+res*child_coord, aabb_min, aabb_max):
        return False
    return True
  
  @staticmethod
  def getHeuristicEuclidean(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))
  
  @staticmethod
  def check_intersection_segment_block(P1, P2, min_extent, max_extent):
    # Check for early exit conditions
    if (P1[0] < min_extent[0] and P2[0] < min_extent[0]) or (P1[0] > max_extent[0] and P2[0] > max_extent[0]):
      return False
    if (P1[1] < min_extent[1] and P2[1] < min_extent[1]) or (P1[1] > max_extent[1] and P2[1] > max_extent[1]):
      return False
    if (P1[2] < min_extent[2] and P2[2] < min_extent[2]) or (P1[2] > max_extent[2] and P2[2] > max_extent[2]):
      return False

    # Compute direction vector of the line segment
    dir = (P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2])

    # Compute minimum and maximum extents of the AABB
    AABB_min = min_extent
    AABB_max = max_extent

    # Compute t values for each axis
    t_min = float('-inf')
    t_max = float('inf')

    for i in range(3):
      if abs(dir[i]) < 1e-6:  # Line segment is parallel to the AABB face
        if P1[i] < AABB_min[i] or P1[i] > AABB_max[i]:
          return False  # No intersection
      else:
        t1 = (AABB_min[i] - P1[i]) / dir[i]
        t2 = (AABB_max[i] - P1[i]) / dir[i]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))

    # Check for intersection
    if t_max < 0 or t_min > 1 or t_min > t_max:
      return False
    return True
  



