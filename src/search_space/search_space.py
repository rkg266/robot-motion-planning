# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

from src.utilities.geometry import es_points_along_line
from src.utilities.obstacle_generation import obstacle_generator


class SearchSpace(object):
    def __init__(self, dimension_lengths, O=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = self.dimensions
        if O is None:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != len(dimension_lengths) for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            tp4 = obstacle_generator(O)
            self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)
            self.obs1_renu = O # My edit

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        return self.obs.count(x) == 0

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        points = es_points_along_line(start, end, r)
        coll_free = all(map(self.obstacle_free, points))
        return coll_free
    
    def collision_free_RENU(self, start, end, r): # MY EDIT
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        blocks = self.obs1_renu
        for aabb in blocks:
            aabb_min = aabb[0:3]
            aabb_max = aabb[3:6]
            if SearchSpace.check_intersection_(start, end, aabb_min, aabb_max):
                return False
        return True

    @staticmethod
    def check_intersection_(P1, P2, min_extent, max_extent):
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

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)
