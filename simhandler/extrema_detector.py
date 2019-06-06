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
    point and reference can be in R^. Must be the same dimension
    @param point: list-like. All coordinates in point
    @param reference: list-like. All coordinates in reference
    @returns float
    '''

    if len(point) != len(reference):
        raise ValueError("length of point not equal to reference")

    distance = 0
    for dim in range(len(reference)):
        distance += (reference[dim]-point[dim])**2
    return sqrt(distance)


def which_cell(point, reference):
    '''
    @brief which_cell: returns "cell index". There are 2^n cells in a R^n
    point. The cell index corresponds to a binary mapping with x_n
    variables
    @param point: list-like. All coordinates in point
    @param reference: list-like. All coordinates in reference
    @return int
    '''

    cell_string = ""
    for dim in range(len(reference)):
        if point[dim] > reference[dim]:
            cell_string += '0'
        elif point[dim] < reference[dim]:
            cell_string += '1'
        else:
            raise NotImplementedError("TODO: handle equal case")
    return int(cell_string, 2)


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
    @brief fast_polytope: find coordinates of that form smallest polytope
    around point. Assumes grid x is uniformly distributed mgrid.
    @param point: point to find polytope around
    @param min_value: min value of coordinates
    @param max_value: max value of coordinates
    @returns: list of ints. Indices of polytope coordinates in x
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


def mgrid_extrema_locator(x, f, eta):
    '''
    @brief mgrid_extrema_locator: finds extrema in a dataset based on \
    formula in https://web.njit.edu/~ansari/papers/04neurocomputing.pdf
    @param x: nested list. Shape (# dim, # samples) or mgrid-like
    @param f: nested list. Shape (# dim, # samples) or mgrid-like
    @param eta: float. A qualifier for what is considered an extrema.
    The smaller, the more likely an extrema will qualify
    @returns: list of tuples
    '''
    if not mgrid_shape(x):
        raise UserWarning("Input is not mgrid shape")

    valid_points = []
    no_points_in_polytope = 2*x.shape[0]
    samples_per_dimension = f.shape[0]
    for idx, val in np.ndenumerate(f):
        neighbours = mgrid_polytope(idx, 0, samples_per_dimension-1)
        if len(neighbours) == no_points_in_polytope:
            f_points = [f[points] for points in neighbours]
            if val < min(f_points)-eta or val > max(f_points)+eta:
                    valid_points.append(idx)
    return valid_points


def organize_neighbours_by_cell(x, reference, one_per_cell=False):
    '''
    @brief TODO
    @param x: list. Shape (N sample, n dim). Function faster if sorted
    @param reference: list. Shape (n dim)
    @param early_stop=False: bool. If True, assumes x is sorted by distance
    and stop when all of the cells have a value
    @return dict with keys [0, 2^len(x)]
    '''

    if isinstance(x, list):
        x = np.array(x)
    if x.shape[1] != len(reference):
        raise ValueError("n dim in x: {} not the same as in reference: {}".
                         format(x.shape[1], len(reference)))

    neighbours_by_cell = {}
    count = 0
    for idx, neighbour in enumerate(x):

        cell = which_cell(neighbour, reference)
        if cell in neighbours_by_cell:
            if not one_per_cell:
                neighbours_by_cell[cell].append(idx)
        else:
            neighbours_by_cell[cell] = [idx]
        count += 1
        if one_per_cell and count == 2**len(x):
            break

    return neighbours_by_cell


def polytope(x, reference):
    '''
    @brief polytope: find n+1 points in (x in R^n) that form smallest \
    polytope around reference. N is the symbol for number of samples. \
    TODO: want to phase everything into mgrid style OR (n dim, N samples) \
    but until then they are handled as separate cases
    @param x: nested list or np.ndarray. Potential polytope points in shape (n, N)
    @param reference: list. Point you want polytope around. Shape (n)
    @returns: list of ints. Indices of polytope points in x
    '''

    try:
        dummy = x[0][0]
    except IndexError:
        raise UserWarning("x is not a nested list")

    if isinstance(x, list):
        x = np.array(x)

    if not (isinstance(x, np.ndarray) and
            isinstance(reference, list) or isinstance(reference, np.ndarray)):
        raise TypeError("x: {}, type: {}, reference: {}, type: {} are not \
proper types".format(x, type(x), reference, type(reference)))

    n = len(x)
    if n > 1:
        for dim in range(n-1):
            if len(x[dim]) != len(x[dim+1]):
                raise UserWarning("Each dimension must have same length")
    N = len(x[0])

    x_T = x.T
    idx_distances = []
    for i in range(N):
        idx_distances.append(euclidean_distance(x_T[i], reference))
    idx_sorted_by_distance = np.argsort(idx_distances)

    idx_neighbours_by_cell = organize_neighbours_by_cell(
        [x.T[idx] for idx in idx_sorted_by_distance], reference, True)
    neighbours = []
    for cell in idx_neighbours_by_cell:
        try:
            idx_sorted_neighbour = idx_neighbours_by_cell[cell][0]  # only 1 element
            neighbours.append(idx_sorted_by_distance[idx_sorted_neighbour])
        except Exception as e:
            print("TODO exception: ", e)
            # "Polytopes only have one neighbour per cell"

    return neighbours


def extrema_locator(x, f, eta):
    '''
    @brief extrema_locator: finds extrema in a dataset based on formula in
    https://web.njit.edu/~ansari/papers/04neurocomputing.pdf
    @param x: nested list. Shape (# dim, # samples) or mgrid-like
    @param f: nested list. Shape (# dim, # samples) or mgrid-like
    @param eta: float. A qualifier for what is considered an extrema.
    The smaller, the more likely an extrema will qualify
    @returns: list of tuples
    '''
    if isinstance(x, list):
        x = np.array(x)
        n = x.shape[0]

    idx_extrema = []
    samples_per_dimension = len(f)
    x_T = x.T
    for ref, val in enumerate(f):
        if ref < len(f):
            x_without_ref = np.delete(x_T, ref, 0).T
            idx_neighbours = polytope(x_without_ref, x_T[ref])
            for idx, value in enumerate(idx_neighbours):
                if value >= ref:
                    idx_neighbours[idx] = value + 1
        if len(idx_neighbours) == 2**x.shape[0]:
            f_neighbours = [f[idx] for idx in idx_neighbours]
            if val < min(f_neighbours)-eta or val > max(f_neighbours)+eta:
                    idx_extrema.append(ref)

    return idx_extrema
