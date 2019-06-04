import unittest
import numpy as np
import utilities


class TestUtilities(unittest.TestCase):
    def test_elementwise_diff_checks_input_size(self):
        """
        Elementwise diff doesn't work if there is only 1 element
        """

        # try passing only one element in list
        A = [1]
        with self.assertRaises(ValueError):
            utilities.elementwise_diff(A)

    def test_elementwise_diff_returns_correct_value(self):
        """
        Checks the return values from the function
        """

        # input array
        A = [1, 1, 2, 3, 5, 8, 13, 21]
        act_diff = utilities.elementwise_diff(A)

        # expected diff
        exp_diff = [0, 1, 1, 2, 3, 5, 8]

        self.assertTrue(np.allclose(act_diff, exp_diff))

    def test_reshape_to_structured_for_nice_xy_input(self):
        """
        Since we just have a list of locations and values, we need to work out
        the dimensions of the data such that it can be interpolated in a
        structured manner

        This test uses 'nice' input, i.e. the data in is still sorted, just
        flattened
        """

        # create structured set of points using meshgrid
        strt, fin, step = 1, 101, 10
        x = np.arange(strt, fin, step)
        y = np.arange(strt, fin + 5, step / 2)
        X, Y = np.meshgrid(x, y)
        X1d, Y1d = X.flatten(), Y.flatten()

        # for now the function is just returning the spacing it has calculated
        # so that we can test this is correct
        x_2d, y_2d = utilities.reshape_to_structured_equivalent(X1d, Y1d)

        # check that the spacing is as was used to create the array
        self.assertTrue(np.allclose(x_2d, X))
        self.assertTrue(np.allclose(y_2d, Y))

    def test_reshape_to_structured_for_unsorted_xy_raises_error(self):
        """
        In the event that the input data is not sorted, raise a ValueError
        """

        # create structured set of points using meshgrid
        strt, fin, step = 1, 101, 10
        x = np.arange(strt, fin, step)
        y = np.arange(strt, fin + 5, step / 2)
        X, Y = np.meshgrid(x, y)
        # only shuffles along the first axis, i.e. re-orders row locations
        np.random.shuffle(X)
        np.random.shuffle(Y)
        X1d, Y1d = X.flatten(), Y.flatten()

        # now check that the method raises a valueError since this would be
        # very hard to resolve
        with self.assertRaises(ValueError):
            utilities.reshape_to_structured_equivalent(X1d, Y1d)

    def test_reshape_to_structured_with_single_function_values(self):
        """
        Checks that the method also reshapes the optional first 'function'
        values and returns 3, 2d arrays all with the same size and correct
        as expected
        """

        # create structured set of points using meshgrid
        strt, fin, step = 1, 101, 10
        x = np.arange(strt, fin, step)
        y = np.arange(strt, fin + 5, step / 2)
        X, Y = np.meshgrid(x, y)
        U = np.exp(-(2 * X)**2 - (Y / 2)**2)
        X1d, Y1d, U1d = X.flatten(), Y.flatten(), U.flatten()

        x_2d, y_2d, u_2d = utilities.reshape_to_structured_equivalent(
            X1d, Y1d, U1d)

        # check that the spacing is as was used to create the array
        self.assertTrue(np.allclose(x_2d, X))
        self.assertTrue(np.allclose(y_2d, Y))
        self.assertTrue(np.allclose(u_2d, U))

    def test_reshape_to_structured_with_both_function_values(self):
        """
        Checks that the method also reshapes the optional first and second
        'function' alues and returns 4, 2d arrays all with the same size and
        correct as expected
        """

        # create structured set of points using meshgrid
        strt, fin, step = 1, 101, 10
        x = np.arange(strt, fin, step)
        y = np.arange(strt, fin + 5, step / 2)
        X, Y = np.meshgrid(x, y)
        U = np.exp(-(2 * X)**2 - (Y / 2)**2)
        V = 2 * np.exp(-(2 * X)**2 - (Y / 2)**2)
        X1d, Y1d, U1d, V1d = X.flatten(), Y.flatten(), U.flatten(), V.flatten()

        x_2d, y_2d, u_2d, v_2d = utilities.reshape_to_structured_equivalent(
            X1d, Y1d, U1d, V1d)

        # check that the spacing is as was used to create the array
        self.assertTrue(np.allclose(x_2d, X))
        self.assertTrue(np.allclose(y_2d, Y))
        self.assertTrue(np.allclose(u_2d, U))
        self.assertTrue(np.allclose(v_2d, V))
