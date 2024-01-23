import numpy as np
import astar
import rrtAlgo

class MyPlanner:
  __slots__ = ['boundary', 'blocks', 'start', 'goal', 'grid_start',
               'grid_goal', 'grid_aabbs', 'grid_size', 'res']
  
  def __init__(self, boundary, blocks, start, goal):
    self.boundary = boundary
    self.blocks = blocks
    self.start = start
    self.goal = goal
    self.grid_start=[]
    self.grid_goal=[]
    self.grid_aabbs=[]
    self.grid_size=[]
    self.res = None

  def discretize(self, res, start, goal):
    self.res = res
    boundaries = self.boundary.flatten()
    blocks = self.blocks
    bnd_min = boundaries[0:3]
    bnd_max = boundaries[3:6]
    numXgrid = int(np.floor((bnd_max[0] - bnd_min[0])/res + 1))
    numYgrid = int(np.floor((bnd_max[1] - bnd_min[1])/res + 1))
    numZgrid = int(np.floor((bnd_max[2] - bnd_min[2])/res + 1))
    self.grid_size = [numXgrid, numYgrid, numZgrid]
    self.grid_start = self.find_grid_coord(res, start)
    self.grid_goal = self.find_grid_coord(res, goal)
    for aabb in blocks:
      i_min, j_min, k_min = self.find_grid_coord(res, aabb[0:3])
      i_max, j_max, k_max = self.find_grid_coord(res, aabb[3:6])
      self.grid_aabbs.append([i_min, j_min, k_min, i_max, j_max, k_max]) # expanding the boxes for more safety


  def find_grid_coord(self, res, point):
    x, y, z = point
    boundaries = self.boundary.flatten()
    blocks = self.blocks
    num_blocks = len(blocks)
    bnd_min = boundaries[0:3]
    bnd_max = boundaries[3:6]
    x_i = int(np.floor((x - bnd_min[0])/res))
    y_j = int(np.floor((y - bnd_min[1])/res))
    z_k = int(np.floor((z - bnd_min[2])/res))
    return [x_i, y_j, z_k]

  def AstarPlan(self, epsilon):
    boundaries = self.boundary.flatten()
    bnd_min = boundaries[0:3]
    bnd_max = boundaries[3:6]
    path_grid = astar.AStar.plan(self, epsilon)
    path_coord = []
    for cur_grid_coord in reversed(path_grid):
      cur_env_coord = bnd_min + self.res*cur_grid_coord
      path_coord.append(cur_env_coord)
    return np.array(path_coord)
  
  def RRTPlan(self):
    boundaries = self.boundary.flatten()
    bnd_min = boundaries[0:3]
    bnd_max = boundaries[3:6]
    path = rrtAlgo.RRTClass.plan(self)
    # path_coord = []
    # for cur_grid_coord_tuple in path_grid:
    #   cur_grid_coord = np.array([cur_grid_coord_tuple[0], cur_grid_coord_tuple[1], cur_grid_coord_tuple[2]])
    #   cur_env_coord = bnd_min + self.res*cur_grid_coord
    #   path_coord.append(cur_env_coord)
    return np.array(path)

  def plan(self,start,goal):
    path = [start]
    numofdirs = 26 # 26 neighbours for a cube
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    [tpX, tpY] = np.meshgrid([-1,0],[-1,0])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1) # remove [0,0,0]
    dR = dR / np.sqrt(np.sum(dR**2,axis=0)) / 2.0
    
    for _ in range(2000):
      mindisttogoal = 1000000
      node = None
      for k in range(numofdirs):
        next = path[-1] + dR[:,k]
        
        # Check if this direction is valid
        if( next[0] < self.boundary[0,0] or next[0] > self.boundary[0,3] or \
            next[1] < self.boundary[0,1] or next[1] > self.boundary[0,4] or \
            next[2] < self.boundary[0,2] or next[2] > self.boundary[0,5] ):
          continue
        
        valid = True
        for k in range(self.blocks.shape[0]):
          if( next[0] >= self.blocks[k,0] and next[0] <= self.blocks[k,3] and\
              next[1] >= self.blocks[k,1] and next[1] <= self.blocks[k,4] and\
              next[2] >= self.blocks[k,2] and next[2] <= self.blocks[k,5] ):
            valid = False
            break
        if not valid:
          continue
        
        # Update next node
        disttogoal = sum((next - goal)**2)
        if( disttogoal < mindisttogoal):
          mindisttogoal = disttogoal
          node = next
      
      if node is None:
        break
      
      path.append(node)
      
      # Check if done
      if sum((path[-1]-goal)**2) <= 0.1:
        break
      
    return np.array(path)
  
  def check_collision(self, path):
    # path: n x 3, n is number of stop points in the path 
    boundaries = self.boundary
    blocks = self.blocks
    num_blocks = len(blocks)
    seg_points = list(zip(path[0:-1], path[1:]))
    for segment in seg_points:
      for aabb in blocks:
        if check_collision_segment_aabb(segment[0], segment[1], aabb[0:3], aabb[3:6]):
          return True
    return False

def check_collision_segment_aabb(P1, P2, min_extent, max_extent):
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
