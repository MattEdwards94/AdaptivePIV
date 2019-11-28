import pytest
import numpy as np
import PIV.distribution as distribution
import PIV.corr_window as corr_window
import PIV.piv_image as piv_image
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors
import PIV.dense_predictor as dense_predictor


def nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


@pytest.fixture
def mock_cw():
    """Creates a dummy correlation window
    """
    return corr_window.CorrWindow(x=10, y=15, WS=21)


@pytest.fixture
def mock_cwList():
    """Creates a dummy list of correlation windows
    """
    return [corr_window.CorrWindow(x=20, y=30, WS=41),
            corr_window.CorrWindow(x=30, y=45, WS=51),
            corr_window.CorrWindow(x=40, y=60, WS=31), ]


@pytest.fixture
def mock_dist(mock_cwList):
    """Creates a dummy distribution object
    """
    return distribution.Distribution(mock_cwList)


def test_initialisation_with_no_inputs():
    """
    Test that creating a Distribution with no inputs creates a valid object
    with self.windows initialised
    """

    dist = distribution.Distribution()

    # check that there is a list initialised and empty
    assert dist.windows == []


def test_initialisation_with_a_single_corr_window(mock_cw):
    """
    I don't think this should cause any issues, but perhaps (as in matlab)
    having 1 item vs a list of items might cause issues
    check that the windows property is a list
    """

    dist = distribution.Distribution(mock_cw)
    assert type(dist.windows) == list


def test_initialisation_with_a_list_of_corr_windows(mock_cwList):
    """
    check that it initialises properly - this should just be storing
    the list internally in the object,
    """

    dist = distribution.Distribution(mock_cwList)
    assert type(dist.windows) == list


def test_initialisation_with_list_is_shallow_only(mock_cwList):
    """
    using something such as list.copy() creates a shallow copy
    That is, the 'list' itself is a 'copy' and will be unique, but the
    contents will be the same.

    i.e.
    >>> list1 = [1, 2, 3,]
    >>> list2 = list1.copy()
    >>> list1.append(4)
    >>> list1 == list2
    ... False
    >>> list1[0] == list2[0]
    ... True

    """

    dist = distribution.Distribution(mock_cwList)
    assert type(dist.windows) == list
    assert dist.windows is not mock_cwList
    print(dist.windows[0], mock_cwList[0])
    assert dist.windows[0] == mock_cwList[0]


def test_n_windows_returns_number_of_windows(mock_cw, mock_cwList):
    """
    Checks it returns the correct number of windows for empty, 1 and many
    """

    dist = distribution.Distribution()
    assert dist.n_windows() == 0

    dist = distribution.Distribution(mock_cw)
    assert dist.n_windows() == 1

    dist = distribution.Distribution(mock_cwList)
    assert dist.n_windows() == 3


def test_get_all_x(mock_cwList):
    """Check that calling obj.x returns a numpy array of the correlation window
    horizontal locations
    """

    # create the distribution with known locations
    dist = distribution.Distribution(mock_cwList)
    x_exp = np.array([20, 30, 40])
    assert np.allclose(x_exp, dist.x)


def test_get_all_y(mock_cwList):
    """Check that calling obj.y returns a numpy array of the correlation window
    vertical locations
    """

    # create the distribution with known locations
    dist = distribution.Distribution(mock_cwList)
    y_exp = np.array([30, 45, 60])
    assert np.allclose(y_exp, dist.y)


def test_get_all_u(mock_cwList):
    """Check that calling obj.u returns a numpy array of the correlation window
    horizontal displacements
    """

    # create the distribution with known locations
    dist = distribution.Distribution(mock_cwList)

    # check that to begin with the u values are all set to NaN
    u_exp = np.array([np.NaN, np.NaN, np.NaN])
    assert np.allclose(u_exp, dist.u, equal_nan=True)

    # now set the displacements to something and check just for good measure
    for ii, cw in enumerate(dist.windows):
        cw.u = 10 * ii

    u_exp = np.array([0, 10, 20])
    assert np.allclose(u_exp, dist.u)


def test_get_all_v(mock_cwList):
    """Check that calling obj.v returns a numpy array of the correlation window
    vertical displacements
    """

    # create the distribution with known locations
    dist = distribution.Distribution(mock_cwList)

    # check that to begin with the u values are all set to NaN
    v_exp = np.array([np.NaN, np.NaN, np.NaN])
    assert np.allclose(v_exp, dist.v, equal_nan=True)

    # now set the displacements to something and check just for good measure
    for ii, cw in enumerate(dist.windows):
        cw.v = 15 * ii

    v_exp = np.array([0, 15, 30])
    assert np.allclose(v_exp, dist.v)


def test_get_all_xy_returns_array_of_xy_locations(mock_cwList):
    """
    Test that an array of all xy locations are returned
    """

    dist = distribution.Distribution(mock_cwList)
    assert np.allclose(dist.get_all_xy(),
                       np.array([[20, 30], [30, 45], [40, 60]]))


def test_get_unmasked_xy_raise_error(mock_cwList):
    """
    When we want to perform vector validation, we don't want to be using the
    masked vectors for the comparison, as this will significantly bias the
    results

    If, however, we have not correlated then the information will not be
    stored in the corr_window and should therefore raise and error
    """

    # create distribution
    dist = distribution.Distribution(mock_cwList)

    # attempt to call method without mask data stored, raise error
    with pytest.raises(ValueError):
        dist.get_unmasked_xy()


def test_get_unmasked_xy_returns_only_unmasked(mock_cwList):
    """
    Check that the returned corr_windows don't include the ones that we
    define as being masked
    """

    # create distribution
    dist = distribution.Distribution(mock_cwList)

    # go through the distribution and set one to be masked, set the rest to
    # be unmasked - set the second one to be masked
    for ii, window in enumerate(dist.windows):
        if ii == 1:
            window.is_masked = True
        else:
            window.is_masked = False

    exp = np.array([[20, 30], [40, 60]])
    assert np.allclose(dist.get_unmasked_xy(), exp)


def test_get_all_uv_returns_array_of_uv_locations(mock_cwList):
    """
    test that an array of all uv values are returned
    """

    dist = distribution.Distribution(mock_cwList)
    act = dist.get_all_uv()
    print(act)
    exp = np.array([[np.NaN, np.NaN],
                    [np.NaN, np.NaN],
                    [np.NaN, np.NaN]])
    print(exp)
    assert nan_equal(act, exp)

    for cw in dist.windows:
        cw.u = 10

    assert nan_equal(dist.get_all_uv(),
                     np.array([[10, np.NaN],
                               [10, np.NaN],
                               [10, np.NaN]]))

    for cw in dist.windows:
        cw.v = 20

    assert nan_equal(dist.get_all_uv(),
                     np.array([[10, 20],
                               [10, 20],
                               [10, 20]]))


def test_get_unmasked_uv_raise_error(mock_cwList):
    """
    When we want to perform vector validation, we don't want to be using the
    masked vectors for the comparison, as this will significantly bias the
    results

    If, however, we have not correlated then the information will not be
    stored in the corr_window and should therefore raise and error
    """

    dist = distribution.Distribution(mock_cwList)

    # attempt to call method without mask data stored, raise error
    with pytest.raises(ValueError):
        dist.get_unmasked_uv()


def test_get_unmasked_uv_returns_only_unmasked(mock_cwList):
    """
    Check that the returned corr_windows don't include the ones that we
    define as being masked
    """

    # create distribution
    dist = distribution.Distribution(mock_cwList)

    # go through the distribution and set one to be masked, set the rest to
    # be unmasked - set the second one to be masked
    for ii, window in enumerate(dist.windows):
        if ii == 1:
            window.is_masked = True
            window.u, window.v = 15, 25
        else:
            window.is_masked = False
            window.u, window.v = 10, 20

    exp = np.array([[10, 20], [10, 20]])
    assert np.allclose(dist.get_unmasked_uv(), exp)


def test_get_all_WS_returns_array_of_WS(mock_cwList):
    """
    test that an array of all WS values are returned
    """

    dist = distribution.Distribution(mock_cwList)
    act = dist.get_all_WS()
    exp = np.array([41, 51, 31])

    assert np.allclose(exp, dist.get_all_WS())


def test_get_unmasked_WS_raise_error(mock_cwList):
    """
    When we want to perform vector validation, we don't want to be using the
    masked vectors for the comparison, as this will significantly bias the
    results

    If, however, we have not correlated then the information will not be
    stored in the corr_window and should therefore raise and error
    """

    dist = distribution.Distribution(mock_cwList)

    # attempt to call method without mask data stored, raise error
    with pytest.raises(ValueError):
        dist.get_unmasked_WS()


def test_get_unmasked_WS_returns_only_unmasked(mock_cwList):
    """
    Check that the returned corr_windows don't include the ones that we
    define as being masked
    """

    # create distribution
    dist = distribution.Distribution(mock_cwList)

    # go through the distribution and set one to be masked, set the rest to
    # be unmasked - set the second one to be masked
    for ii, window in enumerate(dist.windows):
        if ii == 1:
            window.is_masked = True
        else:
            window.is_masked = False

    exp = np.array([41, 31])
    assert np.allclose(dist.get_unmasked_WS(), exp)


def test_NMT_detection_selects_correct_neighbour_values():
    """
    Checks that the numpy indexing is performed correctly
    """

    # creates a diagonal line
    x, y, u, v = (np.arange(50) * 1, np.arange(50) * 2,
                  np.arange(50) * 3, np.arange(50) * 4, )
    xy = np.transpose(np.array([x, y]))
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(xy)
    nb_dist, nb_ind = nbrs.kneighbors(xy)

    # we know that u, v, should be 3 and 4 times the nb_ind, respectively
    norm_exp = []
    for row in nb_ind:
        u_nb, v_nb = 3 * row, 4 * row
        u_med, v_med = np.median(u_nb[1:]), np.median(v_nb[1:])

        u_ctr_fluct, v_ctr_fluct = u_nb[0] - u_med, v_nb[0] - v_med
        u_fluct, v_fluct = u_nb[1:] - u_med, v_nb[1:] - v_med

        u_norm = np.abs(u_ctr_fluct / (np.median(np.abs(u_fluct)) + 0.1))
        v_norm = np.abs(v_ctr_fluct / (np.median(np.abs(v_fluct)) + 0.1))

        norm_exp.append(np.sqrt(u_norm**2 + v_norm**2))

    norm_act = distribution.NMT_detection(u, v, nb_ind)
    assert np.allclose(norm_exp, norm_act)


def test_NMT_detection_all_uniform_returns_zeros():
    """
    If all the values are uniform then the norm should be 0's
    """

    x, y = np.arange(100) * 1, np.arange(100) * 2
    u, v, = np.ones((100, )), np.ones((100, ))
    xy = np.transpose(np.array([x, y]))
    print(np.shape(xy))
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(xy)
    nb_dist, nb_ind = nbrs.kneighbors(xy)

    norm = distribution.NMT_detection(u, v, nb_ind)
    # print(norm)

    assert np.allclose(norm, np.zeros((100, )))


def test_validation_NMT_8NN_stores_old_value():
    """
    We want to keep a track of the old displacement value for future
    reference if needed.

    We don't particularly care in this test whether the validation is
    correct or not, just that if a vector is replaced, that we store the
    old value.
    So we will create a grid of random displacement values, run the vec
    validation, and check that all the invalid vectors have
    u_pre_validation and v_pre_validation set.
    """

    # creates a random displacement field
    x, y = np.arange(49) * 1, np.arange(49) * 1
    u, v = np.random.rand(49), np.random.rand(49)

    # create distribution object, by creating all the corrWindow objects
    dist = distribution.Distribution()
    for xi, yi, ui, vi in zip(x, y, u, v):
        cw = corr_window.CorrWindow(xi, yi, WS=31)
        cw.u, cw.v = ui, vi
        cw.is_masked = False
        dist.windows.append(cw)

    # now run the vector validation
    dist.validation_NMT_8NN()

    # get the flagged vectors
    flag = dist.get_flag_values()

    # now check that they all have the value they originally were given
    for cw, ui, vi, flagi in zip(dist.windows, u, v, flag):
        if flagi is True:
            # it's an outlier
            assert ui == cw.u_pre_validation
            assert vi == cw.v_pre_validation


def test_outlier_replacement_replaces_0_if_all_neighbours_outliers():
    """
    If all neighbours are outliers, then the replaced value should be 0
    """

    # creates a diagonal line
    x, y, u, v = (np.arange(50) * 1, np.arange(50) * 2,
                  np.arange(50) * 3, np.arange(50) * 4, )
    xy = np.transpose(np.array([x, y]))
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(xy)
    nb_dist, nb_ind = nbrs.kneighbors(xy)

    # dummy flag value
    flag = np.zeros((50, ))
    # set all neighbours of 1 location to outlier
    flag[nb_ind[10, :]] = 1
    flag = flag > 0

    # replacement
    u, v = distribution.outlier_replacement(flag, u, v, nb_ind)

    assert u[10] == 0
    assert v[10] == 0


def test_outlier_replacement_is_median_of_valid_neighbours():
    """
    The replacement should be the median value of the neighbouring valid
    vectors
    """

    # creates a diagonal line
    x, y, u, v = (np.arange(50) * 1, np.arange(50) * 2,
                  np.arange(50) * 3, np.arange(50) * 4, )
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    xy = np.transpose(np.array([x, y]))
    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(xy)
    nb_dist, nb_ind = nbrs.kneighbors(xy)

    # dummy flag value
    flag = np.zeros((50, ))
    # set some neighbours of a location to outlier
    flag[nb_ind[10, 0::2]] = 1
    flag = flag > 0

    # expected value
    u_neigh, v_neigh = u[nb_ind[10, 1::2]], v[nb_ind[10, 1::2]]
    u_exp, v_exp = np.median(u_neigh), np.median(v_neigh)

    # replacement
    u, v = distribution.outlier_replacement(flag, u, v, nb_ind)
    assert u[10] == u_exp
    assert v[10] == v_exp


def test_validation_NMT_8NN_ignores_masked_values():

    # creates a random displacement field
    x, y = np.arange(49) * 1, np.arange(49) * 1
    u, v = np.random.rand(49), np.random.rand(49)

    # create distribution object, by creating all the corrWindow objects
    dist = distribution.Distribution()
    for ii, (xi, yi, ui, vi) in enumerate(zip(x, y, u, v)):
        cw = corr_window.CorrWindow(xi, yi, WS=31)
        cw.u, cw.v = ui, vi
        if not (ii % 7):
            cw.is_masked = True
        else:
            cw.is_masked = False
        dist.windows.append(cw)

    # now run the vector validation
    flag = dist.validation_NMT_8NN()

    # check the flag compared to the expected validation
    # a very simple test is just to make sure the flag length is the right size
    assert len(flag) == 42


def test_interpolate_checks_method(mock_dist):
    """
    Checks the method passed is checked for validity
    """

    unacceptable_options = ["not this", "or this", "cub_str"]

    for item in unacceptable_options:
        # check that ValueError is raised
        with pytest.raises(ValueError):
            mock_dist.interp_to_densepred(item, (10, 10))


def test_interpolate_checks_out_dimensions(mock_dist):
    """
    Checks that the output dimensions are checked to be positive integers
    """

    # should run fine
    # self.dist.interp_to_densepred('str_lin', (100, 100))

    # check decimals
    with pytest.raises(ValueError):
        mock_dist.interp_to_densepred('struc_lin', (4.5, 4.5))
    # check negatives
    with pytest.raises(ValueError):
        mock_dist.interp_to_densepred('struc_lin', (-4, 4))


def test_linear_interpolate_onto_pixelgrid():
    """
    check that the interpolation performed is correct
    """

    # create sample meshgrid 0-12 in steps of 2
    xs, ys = np.meshgrid(np.arange(13, step=2), np.arange(13, step=2))
    # sample values
    us, vs = (np.arange(98, step=2).reshape((7, 7)),
              np.arange(98, step=2).reshape((7, 7)) * 4, )

    # expected interpolation
    # this will interpolate 0-13, i.e. will extrap at the far edges
    eval_dim = (14, 14)
    u_exp = (np.tile(np.arange(eval_dim[1]), (eval_dim[0], 1)) +
             (np.arange(eval_dim[0]).reshape((eval_dim[0], 1)) * 7))
    v_exp = u_exp * 4

    # create corr windows
    cwList = []
    for x, y, u, v in zip(xs.ravel(), ys.ravel(), us.ravel(), vs.ravel()):
        cw = corr_window.CorrWindow(x, y, WS=31)
        cw.u, cw.v = u, v
        cwList.append(cw)

    # create distribution
    dist = distribution.Distribution(cwList)

    # now interpolate using method
    u_int, v_int = dist.interp_to_densepred('struc_lin', eval_dim)

    assert np.allclose(u_int, u_exp)
    assert np.allclose(v_int, v_exp)


def test_cubic_interpolate_onto_pixelgrid():
    """
    This one is a bit harder to test without getting into complicated
    formula to work out the expected values.
    For now, we will test that the 'distribution' specific method is the
    same as when being called in the intended way
    """

    # conventional approach
    h = 5
    x, y = np.arange(101, step=h), np.arange(101, step=h)
    xs, ys = np.meshgrid(x, y)
    us, vs = (np.sin(np.pi * xs / 2) * np.exp(ys / 2),
              np.cos(np.pi * xs / 2) * np.exp(ys / 2))

    # we must not extend the evaluation beyond the data points, as we are'nt
    # checking the extrap here
    eval_dim = (100, 100)
    xe, ye = np.arange(eval_dim[1]), np.arange(eval_dim[0])

    # perform interpolation
    f_u = interpolate.interp2d(x, y, us, kind='linear')
    f_v = interpolate.interp2d(x, y, vs, kind='linear')
    u_exp, v_exp = f_u(xe, ye), f_v(xe, ye)

    # now interpolate by creating a distribution and calling the method
    cwList = []
    for x, y, u, v in zip(xs.ravel(), ys.ravel(), us.ravel(), vs.ravel()):
        cw = corr_window.CorrWindow(x, y, WS=31)
        cw.u, cw.v = u, v
        cwList.append(cw)
    dist = distribution.Distribution(cwList)
    u_int, v_int = dist.interp_to_densepred('struc_lin', eval_dim)

    assert np.allclose(u_int, u_exp)
    assert np.allclose(v_int, v_exp)


def test_all_windows_have_displacements_after_correlating():
    """
    Assuming that there is no mask in place, then following
    dist.correlate_all_windows(img, dp) all corr windows should have
    some displacement value
    """

    # create random image
    IA, IB, mask = (np.random.rand(100, 100),
                    np.random.rand(100, 100), np.ones((100, 100)))
    img = piv_image.PIVImage(IA, IB, mask)
    # create empty displacement field
    dp = dense_predictor.DensePredictor(
        u=np.zeros_like(IA), v=np.zeros_like(IA))

    # create random distribution
    x, y = (np.random.randint(0, 100, (100, )),
            np.random.randint(0, 100, (100, )))
    dist = distribution.Distribution()
    for xi, yi in zip(x, y):
        dist.windows.append(corr_window.CorrWindow(xi, yi, WS=31))

    # correlate all locations
    dist.correlate_all_windows(img, dp)

    for cw in dist.windows:
        assert not np.isnan(cw.u)
        assert not np.isnan(cw.v)
