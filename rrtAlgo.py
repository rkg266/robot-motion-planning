import numpy as np

from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

class RRTClass(object):
    @staticmethod
    def plan(environment):
        #grid_size = environment.grid_size
        bdry = environment.boundary.flatten()
        X_dimensions = np.array([(bdry[0], bdry[3]), (bdry[1], bdry[4]), (bdry[2], bdry[5])])  # dimensions of Search Space
        x_init = tuple(environment.start)  # starting location
        x_goal = tuple(environment.goal)  # goal location

        Q = np.array([(2, 4), (1, 4)])  # length of tree edges (dist, number of samples)
        r = 0.1  # length of smallest edge to check for intersection with obstacles. NOT USED HERE
        max_samples = 100*1024  # max number of samples to take before timing out
        prc = 0.1  # probability of checking for a connection to goal

        # create Search Space
        Obstacles_lists = environment.blocks
        Obstacles = [tuple(x[0:6]) for x in Obstacles_lists]
        Obstacles = np.array(Obstacles)
        X = SearchSpace(X_dimensions, Obstacles)
        
        # create rrt_search
        rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
        path = rrt.rrt_search()
        return path
    