import numpy as np


def elementwise_diff(A):
    """Returns the list of differences i.e. A[i+1] - A[i]

    Args:
        A (list, numeric): the input list to calculate the diff of

    Returns:
        (list, numeric): list of differences of next element to current element

    Raises:
        ValueError: If the input has only 0 or 1 elements
    """
    if len(A) < 2:
        raise ValueError("More than 2 elements needed to perform diff")

    return [nxt - curr for curr, nxt in zip(A, A[1:])]


def auto_reshape(x, y, f1=None, f2=None):
    """
    Returns 2D structured data of unknown dimensions, from the flattened data

    If the dimensions are known then reshape should be used.

    Args:
        x (list): flattened structured data locations
        y (list): flattened structured data locations
        f1 (list, optional): flattened structured data values
        f2 (list, optional): flattened structured data values. Optional to allow
                            for same functionality for either a single set of
                            data values or for two sets of values, e.g. u,v
    """

    # get spacing in x by finding the first value which is different from the
    # preceeding value
    # this should be the first element if we are flattening row wise
    x_diff = elementwise_diff(x)
    for ii, item in enumerate(x_diff):
        if not item == 0:
            x_spacing = item
            break

    # get spacing in y as above
    # also we need to grab the location where the value is different, so that
    # we can work out where the correct dimensions to return
    y_diff = elementwise_diff(y)
    for ii, item in enumerate(y_diff):
        if not item == 0:
            y_spacing = item
            # get the length of the row, i.e. the x_dim
            x_dim = ii + 1
            break

    # get y dim by seeing how many rows fit into the total number of elements
    y_dim = int(len(x) / x_dim)

    # now reshape the input arrays and data
    x_2d = np.reshape(x, (y_dim, x_dim))
    y_2d = np.reshape(y, (y_dim, x_dim))

    # check that the array is valid - i.e. equivalent to a meshgrid
    # get the first and last values of x and y
    x_strt, y_strt = x[0], y[0]
    x_end, y_end = x_2d[0, -1], y_2d[-1, 0]

    # now create equivalent meshgrid
    X_check, Y_check = np.meshgrid(np.arange(x_strt, x_end + 1, x_spacing),
                                   np.arange(y_strt, y_end + 1, y_spacing), )

    if not (np.allclose(X_check, x_2d) and np.allclose(Y_check, y_2d)):
        raise ValueError(
            "Input must be sorted, i.e. just flattened along rows")

    # determine outputs
    if (f1 is not None) and (f2 is None):  # f1 defined, f2 is not
        f1_2d = np.reshape(f1, (y_dim, x_dim))
        return x_2d, y_2d, f1_2d
    elif (f1 is None) and (f2 is not None):  # f2 is defined, f1 is not
        f2_2d = np.reshape(f2, (y_dim, x_dim))
        return x_2d, y_2d, f2_2d
    elif (f1 is not None) and (f2 is not None):  # both f1 and f2 are defined
        f1_2d = np.reshape(f1, (y_dim, x_dim))
        f2_2d = np.reshape(f2, (y_dim, x_dim))
        return x_2d, y_2d, f1_2d, f2_2d
    else:  # neither f1 or f2 are defined
        return x_2d, y_2d


if __name__ == '__main__':
    strt, fin, step = 1, 41, 10
    x = np.arange(strt, fin, step)
    y = np.arange(strt, fin + 6, step / 2)
    X, Y = np.meshgrid(x, y)
    print(X)
    print(Y)
    X_1d = X.flatten()
    Y_1d = Y.flatten()
    print(X_1d)
    print(Y_1d)

    # for now the function is just returning the spacing it has calculated
    # so that we can test this is correct
    x_2d, y_2d = auto_reshape(X_1d, Y_1d, X_1d)
    print(x_2d, y_2d)
