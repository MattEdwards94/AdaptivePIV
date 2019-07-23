import pytest
import numpy as np
import utilities


def test_elementwise_diff_checks_input_size():
    """
    Elementwise diff doesn't work if there is only 1 element
    """

    # try passing only one element in list
    A = [1]
    with pytest.raises(ValueError):
        utilities.elementwise_diff(A)


def test_elementwise_diff_returns_correct_value():
    """
    Checks the return values from the function
    """

    # input array
    A = [1, 1, 2, 3, 5, 8, 13, 21]
    act_diff = utilities.elementwise_diff(A)

    # expected diff
    exp_diff = [0, 1, 1, 2, 3, 5, 8]

    assert np.allclose(act_diff, exp_diff)


def test_auto_reshape_for_nice_xy_input():
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
    x_2d, y_2d = utilities.auto_reshape(X1d, Y1d)

    # check that the spacing is as was used to create the array
    assert np.allclose(x_2d, X)
    assert np.allclose(y_2d, Y)


def test_auto_reshape_for_unsorted_xy_raises_error():
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
    with pytest.raises(ValueError):
        utilities.auto_reshape(X1d, Y1d)


def test_auto_reshape_with_single_function_values():
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

    x_2d, y_2d, u_2d = utilities.auto_reshape(
        X1d, Y1d, U1d)

    # check that the spacing is as was used to create the array
    assert np.allclose(x_2d, X)
    assert np.allclose(y_2d, Y)
    assert np.allclose(u_2d, U)


def test_auto_reshape_with_both_function_values():
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

    x_2d, y_2d, u_2d, v_2d = utilities.auto_reshape(
        X1d, Y1d, U1d, V1d)

    # check that the spacing is as was used to create the array
    assert np.allclose(x_2d, X)
    assert np.allclose(y_2d, Y)
    assert np.allclose(u_2d, U)
    assert np.allclose(v_2d, V)


def test_lin_extrap_edges_with_1d_default_npad():
    """
    Check for a one dimensional input that the output is linearly
    extrapolated at each end
    """

    # input values. The 3000 shouldn't affect the extrapolation
    in_values = [1, 1.5, 3000, 2.5, 3]

    # extrapolate
    out = utilities.lin_extrap_edges(in_values)
    print(out)

    # expected
    out_expected = [0.5] + in_values + [3.5]
    assert np.allclose(out, out_expected)


def test_lin_extrap_edges_with_1d_modified_npad():
    """
    Check for a one dimensional input that the output is linearly
    extrapolated at each end if the n_pad is greater than 1
    """

    # input values. The 3000 shouldn't affect the extrapolation
    in_values = [1, 1.5, 3000, 2.5, 3]

    # extrapolate
    out = utilities.lin_extrap_edges(in_values, n_pad=3)
    print(out)

    # expected
    out_expected = [-0.5, 0, 0.5] + in_values + [3.5, 4, 4.5]
    assert np.allclose(out, out_expected)


def test_lin_extrap_edges_with_2d_default_npad():
    """
    Tests the output is linearly extrapolated for a 2D input
    """

    in_values = np.arange(16).reshape((4, 4))

    # extrapolate
    out = utilities.lin_extrap_edges(in_values)
    print(out)

    out_expected = [[-5, -4, -3, -2, -1, 0],
                    [-1, 0, 1, 2, 3, 4],
                    [3, 4, 5, 6, 7, 8],
                    [7, 8, 9, 10, 11, 12],
                    [11, 12, 13, 14, 15, 16],
                    [15, 16, 17, 18, 19, 20], ]

    assert np.allclose(out, out_expected)


def test_lin_extrap_edges_with_2d_modified_npad():
    """
    Tests the output is linearly extrapolated for a 2D input
    """

    in_values = np.arange(16).reshape((4, 4))

    # extrapolate
    out = utilities.lin_extrap_edges(in_values, n_pad=2)
    print(out)

    out_expected = [[-10, -9, -8, -7, -6, -5, -4, -3],
                    [-6, -5, -4, -3, -2, -1, 0, 1],
                    [-2, -1, 0, 1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    [6, 7, 8, 9, 10, 11, 12, 13],
                    [10, 11, 12, 13, 14, 15, 16, 17],
                    [14, 15, 16, 17, 18, 19, 20, 21],
                    [18, 19, 20, 21, 22, 23, 24, 25], ]

    assert np.allclose(out, out_expected)


def test_round_to_odd_doesnt_change_odd_int():
    """
    If an odd integer is passed in, then the output should be the same
    """

    vals = [15, 45, 101]
    output = list(map(utilities.round_to_odd, vals))

    assert output == vals


def test_round_to_odd_for_even_value_rounds_up():
    """
    If an even number is passed in, the value should round up
    """

    vals = [14, 44, 100]
    output = list(map(utilities.round_to_odd, vals))

    exp = [15, 45, 101]
    assert output == exp


def test_round_to_odd_round_down():
    """
    values up to 0.9999 greater than an odd should round down
    """

    vals = [15.55, 45.495, 101.9999]
    output = list(map(utilities.round_to_odd, vals))

    exp = [15, 45, 101]
    assert output == exp


def test_round_to_odd_round_up():
    """
    Values -1 and above should round up
    """

    vals = [14, 44.4681, 100.22516]
    output = list(map(utilities.round_to_odd, vals))

    exp = [15, 45, 101]
    assert output == exp
