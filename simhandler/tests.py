"""
Unit tests

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

import unittest
from n_dimensional_datasets import *
from extrema_detector import *


class TestNDimensionalDatasets(unittest.TestCase):

    def test_mesh(self):
        no_dimensions_to_test = 5
        # keep these small for test speed
        start = 0
        stop = 1
        steps = 0.25
        for n in range(1, no_dimensions_to_test):
            self.assertEqual(len(uniform_mesh(n, start, stop, steps).shape), n+1)

        testee = uniform_mesh(1, start, stop, steps)
        answer = [[0, 0.25, 0.5, 0.75, 1]]
        for idx, value in np.ndenumerate(answer):
            self.assertEqual(testee[idx], value)

    def test_decaying_sinewave_nd(self):
        # keep these small for test speed
        start = 0
        stop = 1
        steps = 0.25
        n = 2
        x = uniform_mesh(n, start, stop, steps)
        testee = decaying_sinewave_nd(x)
        answer = [[1.00000000e+00,  5.75224600e-17, -7.78800783e-01,
                  -1.04667407e-16,  3.67879441e-01],
                  [5.75224600e-17,  3.30883341e-33, -4.47985369e-17,
                   -6.02072673e-33,  2.11613304e-17],
                  [-7.78800783e-01, -4.47985369e-17,  6.06530660e-01,
                   8.15150584e-17, -2.86504797e-01],
                  [-1.04667407e-16, -6.02072673e-33,  8.15150584e-17,
                   1.09552661e-32, -3.85049872e-17],
                  [3.67879441e-01,  2.11613304e-17, -2.86504797e-01,
                   -3.85049872e-17,  1.35335283e-01]]
        for idx, value in np.ndenumerate(answer):
            # 9 is the highest precision test should pass with
            self.assertEqual(round(testee[idx], 9), round(value, 9))


class TestExtremaDetector(unittest.TestCase):

    def test_mgrid_polytope(self):
        idx = (1, 3, 3)
        testee = mgrid_polytope(idx, 0, 4)
        answer = [(0, 3, 3), (2, 3, 3),
                  (1, 2, 3), (1, 4, 3),
                  (1, 3, 2), (1, 3, 4)]
        for i, value in enumerate(answer):
            self.assertEqual(testee[i], value)

    def test_mgrid_extrema_locator(self):
        n = 3
        start = -1
        stop = 1
        steps = 0.1
        x = uniform_mesh(n, start, stop, steps)
        f = decaying_sinewave_nd(x)
        eta = 1e-5

        testee = len(mgrid_extrema_locator(x, f, eta))
        answer = 27
        self.assertEqual(testee, answer)

    def test_euclidean_distance(self):
        point = [1, 2, 3]
        ref = [2, 3, 4]
        testee = euclidean_distance(point, ref)
        answer = 1.7320508075688772
        self.assertEqual(testee, answer)
        point = [1]
        ref = [3]
        self.assertEqual(euclidean_distance(point, ref), 2.0)

    def test_which_cell(self):
        ref = [0, 0, 0]
        answer = 7
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    point = [x, y, z]
                    testee = which_cell(point, ref)
                    self.assertEqual(testee, answer)
                    answer -= 1

    def test_organize_neighbours_by_cell(self):
        x = np.array([[0, 2, 200, 1]])
        ref = [1.5]
        testee = organize_neighbours_by_cell(x.T, ref)
        answer = {0: [1, 2], 1: [0, 3]}
        self.assertEqual(testee, answer)
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        ref = [2.5, 2.5]
        testee = organize_neighbours_by_cell(x.T, ref)
        answer = {2: [0, 1], 0: [2, 3]}
        self.assertEqual(testee, answer)
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        ref = [2.5, 2.5]
        testee = organize_neighbours_by_cell(x.T, ref, True)
        answer = {2: [0], 0: [2]}
        self.assertEqual(testee, answer)

    def test_polytope(self):
        x = [[0, 1, 200, 3]]
        ref = [0.5]
        self.assertEqual(polytope(x, ref), [0, 1])
        x = [[0, 1, 1.1, 0.1, 2], [0, 0.1, 1.1, 1, 2]]
        ref = [0.5, 0.5]
        self.assertEqual(polytope(x, ref), [1, 3, 0, 2])
        x = [[1, 1], [123, 412], [423, 1212]]
        ref = [1, 10, 100]
        self.assertRaises(NotImplementedError, polytope, x, ref)

    def test_extrema_locator(self):
        x = [[0, 0.5, 1, 1.5]]
        f = decaying_sinewave_nd(x)
        eta = 1e-5
        self.assertEqual(extrema_locator(x, f, eta), [1, 2])
        x = [[-1, 1, 1.1, -0.9, 0], [-1, -1.1, 1, 0.9, 0]]
        f = decaying_sinewave_nd(x)
        eta = 1e-5
        self.assertEqual(extrema_locator(x, f, eta), [4])


if __name__ == "__main__":
    unittest.main()
