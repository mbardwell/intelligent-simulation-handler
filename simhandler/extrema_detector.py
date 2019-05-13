"""
Extrema detection functions

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import numpy as np
from itertools import product
from math import sqrt


def euclidean_distance(point, reference):
    '''
    @brief euclidean_distance: distance between point and reference. Both
    point and reference are in R^n
    @param point: list. All coordinates in point
    @param reference: list. All coordinates in reference
    @returns float
    '''
    distance = 0
    for dim in range(len(reference)):
        distance += (reference[dim]-point[dim])**2
    return sqrt(distance)


def which_cell(coordinate, reference):
    '''
    @brief which_cell: returns "cell index". There are 2^n cells in a R^n
    coordinate. The cell index corresponds to a binary mapping with x_n
    variables
    @param coordinate: list
    @param reference: list. Reference coordinate
    @return cell_index: int
    '''
    no_cells = len(reference)
    n = 0
    loop = 0
    while n < len(coordinate):
        value = bigger_than(coordinate[n], reference[n])
        loop += 1
    return cell_index


def mgrid_shape(x):
    '''
    @brief determine_if_uniformly_distrubuted. Based on mgrid nested list
    structure. If x is uniformly distributed all shape(x)[1:] would be equal.
    N: number of samples. n: number of dimensions
    @param x: nested list
    @returns bool
    '''
    x_shape = np.array(x).shape
    try:
        if np.all([x_shape[n] == x_shape[n+1] for n in range(1, x_shape[0])]):
            return True  # input has mgrid shape (n, N dim_0, ..., N dim_n)
    except IndexError:
        pass
    return False


def mgrid_polytope(point, min_value, max_value):
    '''
    @brief fast_polytope: returns coords that form smallest polytope
    around point. Assumes grid x is uniformly distributed mgrid.
    @param index: int. Index of point to be enclosed
    @returns: list of ints. Indices of polytope points in x
    '''
    points = []
    for idx in range(len(point)):
        point_copy = list(point)  # create mutable list
        point_copy[idx] -= 1
        if point_copy[idx] >= min_value:
            points.append(tuple(point_copy))
        point_copy = list(point)  # create mutable list
        point_copy[idx] += 1
        if point_copy[idx] <= max_value:
            points.append(tuple(point_copy))
    return points


def polytope(x, point):
    '''
    @brief polytope: find points in x that form smallest polytope around
    point. For x in R^n, the smallest polytope will require 2^n points
    @param x: nested list. All coordinates in n-dimensional grid
    @param point: tuple. Coordinate(s) of point to find polytope around
    @returns: list of ints. Indices of polytope points in x
    '''
    x = np.array(x)
    if mgrid_shape(x):
        # if mgrid shape we assume it is uniformly distributed. Then we can
        # use faster, mgrid_polytope function
        # TODO: either add uniform dist. check here or stop using mgrids
        samples_per_dimension = len(x[0])
        return mgrid_polytope(point, 0, samples_per_dimension-1)

    for dim in range(len(x)-1):
        if len(x[dim]) != len(x[dim+1]):
            raise UserWarning("Each dimension must have same length")
        else:
            no_samples = len(x[dim])

    polytope_coords = np.zeros(2**np.array(x).shape[0])

    distances = [euclidean_distance(x.T[i], x.T[point]) for i in range(no_samples)]
    sorted_eucl_distance_indices = np.argsort(distances)[1:]  # remove index
    polytope_coords = sorted_eucl_distance_indices[0:2**np.array(x).shape[0]]

    return polytope_coords


def extremum_locator(x, f, eta):
    '''
    @brief extremum_locator: finds extremum in a dataset based on formula in
    https://web.njit.edu/~ansari/papers/04neurocomputing.pdf
    @param x: nested list. Shape (# dim, # samples) or mgrid-like
    @param f: nested list. Shape (# dim, # samples) or mgrid-like
    @param eta: float. A qualifier for what is considered an extremum.
    The smaller, the more likely an extremum will qualify
    @returns: list of tuples
    '''
    valid_points = []
    no_points_in_polytope = 2*len(f.shape)
    samples_per_dimension = f.shape[0]
    for idx, val in np.ndenumerate(f):
        neighbours = polytope(x, idx)
        if len(neighbours) == no_points_in_polytope:
            f_points = [f[points] for points in neighbours]
            if val < min(f_points)-eta or val > max(f_points)+eta:
                    valid_points.append(idx)
    return valid_points
