"""
Extrema detection functions

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import numpy as np
from itertools import product


# @brief polyhedron: find points in smallest polyhedron enclosing point x_i.
# @brief Assumes uniform distribution
# @param x_i: tuple
# @param min_value: int. Min value allowed in returned tuples
# @param max_value: int. Max value allowed in returned tuples
# @returns: list of tuples
def polyhedron(x_i, min_value, max_value):
    points = []
    for idx in range(len(x_i)):
        point = list(x_i)  # create mutable list
        point[idx] -= 1
        if point[idx] >= min_value:
            points.append(tuple(point))
        point = list(x_i)  # create mutable list
        point[idx] += + 1
        if point[idx] <= max_value:
            points.append(tuple(point))
    return points


# @brief extremum_locator: finds extremum in a dataset based on formula in
# https://web.njit.edu/~ansari/papers/04neurocomputing.pdf. It is that f is
# is sorted according to it's independent variables. ie: f_(i-1), f_i, f_(i+1)
# @param f: array like. Shape (# samples dim 0, ..., # samples dim n)
# @param eta: float. A qualifier for what is considered an extremum.
# The smaller, the more likely an extremum will qualify
# @returns: list of tuples
def extremum_locator(f, eta):
    valid_points = []
    no_points_in_polyhedron = 2*len(f.shape)
    samples_per_dimension = f.shape[0]
    for idx, val in np.ndenumerate(f):
        neighbours = polyhedron(idx, 0, samples_per_dimension-1)
        if len(neighbours) == no_points_in_polyhedron:
            f_points = [f[points] for points in neighbours]
            if val < min(f_points)-eta or val > max(f_points)+eta:
                    valid_points.append(idx)
    return valid_points
