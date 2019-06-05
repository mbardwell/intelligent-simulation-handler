"""
Produces R^n -> R datasets for n in Z

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import numpy as np
import random


def uniform_mesh(n, start, stop, steps):
    '''
    Mgrid style n-dimensional mesh

    Parameters
    ----------
    n: int

    start: number-like

    stop: number-like

    steps: number-like

    Returns
    -------
    Nested list
    '''
    if n < 1 or not isinstance(n, int):
        raise ValueError("dimension passed to mesh is invalid")
    mgrid = np.mgrid[tuple(slice(start, stop+steps, steps) for _ in range(n))]
    return mgrid


def stochastic_mesh(n, start, stop, N, seed=None, precision=None):
    '''
    @brief mesh: sorted, random n-dimensional layout with guaranteed domain
    @param n: int. Number of dimensions
    @param start: float
    @param stop: float
    @param N: int. Number of samples
    @returns: nested list. Always includes stop & start
    '''
    if n < 1 or not isinstance(n, int):
        raise ValueError("dimension passed to mesh is invalid")
    if not (isinstance(N, int) or N > 0):
        raise ValueError("number of samples must be an integer and >0")
    if seed is not None and isinstance(seed, int):
        random.seed(seed)

    def random_vector(start, stop, N):
        K = stop-start
        if precision is not None:
            return [start]+[K*round(random.random(), precision)+start for _ in range(N-2)]+[stop]
        else:
            return [start]+[K*random.random()+start for _ in range(N-2)]+[stop]

    grid = np.array([random_vector(start, stop, N) for _ in range(n)])
    return grid


# @brief flattened_mesh: arranges mesh() return in shape (1, no samples)
# @param x: array like. Shape (n dim, # samples dim 0, ..., # samples dim n)
# @output: list
def flattened_mesh(x):
    X = []
    X_sub = []
    for idx, _ in np.ndenumerate(x[0]):
        for dim in range(x.shape[0]):
            X_sub.append(x[dim][idx])
        X.append(X_sub)
        X_sub = []
    return X


# @brief decaying_sinewave_nd: produces n-dimensional decaying sinewave dataset
# @param x: nested list
# @param frequency=2: float
# @param noise=0: float. Noise argument fills the scale argument in
# @param noise=0: numpy.random.normal
# @returns: nested list. Same as mgrid
def decaying_sinewave_nd(x, frequency=2, noise=0):
    x = np.array(x)
    n = x.shape[0]
    f = np.ones(x[0].shape)
    for point, _ in np.ndenumerate(f):
        for dim in range(n):
            f[point] *= np.cos(frequency*np.pi*x[dim][point])*np.exp(-x[dim][point]**2)
        if noise > 0.1:
            noise = 0.1
        elif noise < 0:
            noise = 0
    f += np.random.normal(0, noise, f.shape)
    return f


# @brief logistic: produces n-dimensional logistic dataset
# @param x: nested list
# @returns: nested list. Same as mgrid
def logistic_nd(x):
    n = x.shape[0]
    f = np.ones(x[0].shape)
    for point, _ in np.ndenumerate(f):
        for dim in range(n):
            f[point] *= np.exp(-x[dim][point])
        f[point] = f[point] + 1
        f[point] = 1/f[point]
    return f


# @brief logistic: produces n-dimensional logistic dataset
# @param w: nested list. The ith element in the list represents the
# @param w: weight matrix corresponding to layer i
# @param x: nested list
# @returns: nested list. Same as mgrid
def weighted_logistic_nd(w, x):
    n = x.shape[0]
    if (w[0].shape[0] != n):
        raise ValueError("Shape of weights: {} and number of first hidden\
layer neurons: {} not compatible".format(
            w[0].shape[0], n))
    f = np.ones(x[0].shape)
    for point, _ in np.ndenumerate(f):
        for dim in range(n):
            f[point] *= np.exp(-w[0][dim]*x[dim][point])
        f[point] = f[point] + 1
        f[point] = 1/f[point]
    return f


# @brief constant_nd: produces n-dimensional constant valued dataset
# @param x: nested list
# @param offset: float. Offset from zero. For example in 3D each point on the
# @param offset: plane will have value offset
# @returns: nested list. Same as mgrid
def constant_nd(x, offset):
    n = x.shape[0]
    f = np.ones(x[0].shape)*offset
    return f


def random_nd(x, scale, seed=None):
    '''
    Produces n-dimensional random valued dataset

    Parameters
    ----------
    x: nested list
    bias: float
        Noise amplitude

    Returns
    -------
    np.ndarray
    '''
    n = x.shape[0]
    np.random.seed(seed)
    f = np.random.normal(0, scale, x[0].shape)
    return f


def sloped_nd(x, bias):
    '''
    Produces n-dimensional sloped dataset

    Parameters
    ----------
    x: list-like
    bias: float/int

    Returns
    -------
    np.ndarray
    '''

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    dim = x.shape[0]
    f = np.zeros(x[0].shape)
    for i in range(len(f)):
        for j in range(dim):
            f[i] += x[j][i]
        f[i] += bias
    return f


# @brief perturb_nd: perturbs n-dimensional dataset with a smooth bump
# @param f: nested list
# @param bump_x: nested list
# @param scale: float. The amplitude of the bump
# @param sharpness: float. The frequency arguement of decaying_sinewave_nd
# @returns: nested list. Same as mgrid
def perturb_nd(f, bump_x, scale=0.1, sharpness=6):
    if np.array(bump_x).shape[0] != len(bump_idx) != len(f):
        raise ValueError("Shape of index: {}, input x: {}, and output f: {} do\
not match".format(
            len(bump_idx), np.array(f).shape[0]), len(f))
    bump = decaying_sinewave_nd(bump_x, sharpness)
    for point, value in np.ndenumerate(bump):
        if value < 0:
            value = 0
        f[point] += value*scale
    return f
