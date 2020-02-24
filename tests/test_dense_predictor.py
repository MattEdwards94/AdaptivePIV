import pytest
import numpy as np
import PIV.dense_predictor as dense_predictor
import PIV.utilities as utils


def test_initialisation_with_mask():
    """
    checks the initialisation using u, v, and mask
    checks that the stored object values are correct
    """

    u = np.random.rand(100, 100)
    v = np.random.rand(100, 100)
    mask = np.random.randint(0, 2, (100, 100))
    dp = dense_predictor.DensePredictor(u, v, mask)

    u[mask == 0] = 0
    v[mask == 0] = 0

    # check the saved u, v, mask
    assert np.alltrue(dp.u == u)
    assert np.alltrue(dp.v == v)
    assert np.alltrue(dp.mask == mask)


def test_initialisation_without_mask():
    """
    Checks that u and v are stored correctly, and that a mask of
    ones is created with the same shape as u and v
    """
    u = np.random.rand(100, 100)
    v = np.random.rand(100, 100)
    dp = dense_predictor.DensePredictor(u, v)

    # check the saved u, v, mask. mask should be ones
    assert np.alltrue(dp.u == u)
    assert np.alltrue(dp.v == v)
    assert np.alltrue(dp.mask == np.ones((100, 100)))


def test_initialisation_with_mask_checks_size():
    """
    u, v, mask must have the same shape
    """
    u = np.random.rand(100, 100)
    v = np.random.rand(110, 110)
    mask = np.random.randint(0, 2, (100, 100))

    # check u vs v
    with pytest.raises(ValueError):
        dense_predictor.DensePredictor(u, v, mask)

    # check mask
    with pytest.raises(ValueError):
        dense_predictor.DensePredictor(v, v, mask)


def test_initialisation_without_mask_checks_size():
    """
    checks that u and v sizes are still compared even if mask isn't passed
    """
    u = np.random.rand(100, 100)
    v = np.random.rand(110, 110)

    # check u vs v
    with pytest.raises(ValueError):
        dense_predictor.DensePredictor(u, v)


def test_initialisation_saves_mask_status():
    """
    If there has been a mask passed we want to save this information as
    an easily checkable bool
    """
    u = np.random.rand(100, 100)
    v = np.random.rand(100, 100)
    mask = np.random.randint(0, 2, (100, 100))
    dp = dense_predictor.DensePredictor(u, v)
    assert not dp.has_mask
    dp2 = dense_predictor.DensePredictor(u, v, mask)
    assert dp2.has_mask


def test_image_dimensions_are_captured():
    """check that the size of the image is captured into the variables
    n_rows
    n_cols
    img_dim
    """

    # use non-square images so we are sure that we are capturing the
    # correct dimensions
    u = np.random.rand(50, 100)
    v = np.random.rand(50, 100)
    mask = np.random.randint(0, 2, (50, 100))
    dp = dense_predictor.DensePredictor(u, v, mask)
    assert dp.n_rows == 50
    assert dp.n_cols == 100
    assert dp.dim == (50, 100)


def test_get_region_returns_correct_region():
    """
    The region returned should be ctr-rad:ctr+rad in both x and y
    Can test this by creating an image with known displacements

    [[ 1,  2,  3,  4,  5,  6],
     [ 7,  8,  9, 10, 11, 12],
     [13, 14, 15, 16, 17, 18],
     [19, 20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29, 30],
     [31, 32, 33, 34, 35, 36]]

    """

    size_of_img = (6, 6)
    u = np.arange(1, size_of_img[0] * size_of_img[1] + 1)
    u = np.reshape(u, size_of_img)
    v = np.array(u)
    mask = np.array(u)
    dp = dense_predictor.DensePredictor(u, v, mask)
    u, v, mask = dp.get_region(3, 3, 2)

    # manually determine the expected array
    exp_arr = np.array([[8, 9, 10, 11, 12],
                        [14, 15, 16, 17, 18],
                        [20, 21, 22, 23, 24],
                        [26, 27, 28, 29, 30],
                        [32, 33, 34, 35, 36]])
    print(u)
    assert np.allclose(u, exp_arr)
    assert np.allclose(v, exp_arr)
    assert np.allclose(mask, exp_arr)

    # what happens if we truncate to the top left:
    u, v, mask = dp.get_region(1, 1, 2, truncate=True)
    exp_arr = np.array([[1, 2, 3, 4],
                        [7, 8, 9, 10],
                        [13, 14, 15, 16],
                        [19, 20, 21, 22]])
    assert np.allclose(u, exp_arr)
    assert np.allclose(v, exp_arr)
    assert np.allclose(mask, exp_arr)

    # if we pad with 0's instead
    u, v, mask = dp.get_region(1, 1, 2, truncate=False)
    exp_arr = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4],
                        [0, 7, 8, 9, 10],
                        [0, 13, 14, 15, 16],
                        [0, 19, 20, 21, 22]])
    print(u)
    assert np.allclose(u, exp_arr)
    assert np.allclose(v, exp_arr)
    assert np.allclose(mask, exp_arr)

    # what happens if we truncate to the bottom right:
    u, v, mask = dp.get_region(4, 4, 2, truncate=True)
    exp_arr = np.array([[15, 16, 17, 18],
                        [21, 22, 23, 24],
                        [27, 28, 29, 30],
                        [33, 34, 35, 36]])
    print(u)
    assert np.allclose(u, exp_arr)
    assert np.allclose(v, exp_arr)
    assert np.allclose(mask, exp_arr)

    # if we pad with 0's
    u, v, mask = dp.get_region(4, 4, 2, truncate=False)
    exp_arr = np.array([[15, 16, 17, 18, 0],
                        [21, 22, 23, 24, 0],
                        [27, 28, 29, 30, 0],
                        [33, 34, 35, 36, 0],
                        [0, 0, 0, 0, 0]])
    print(u)
    assert np.allclose(u, exp_arr)
    assert np.allclose(v, exp_arr)
    assert np.allclose(mask, exp_arr)


def test_eq_method_evaluates_correctly():
    """
    the __eq__ method should compare if two DensePredictor objects are
    the same or not

    raises NotImplemented if the other class is not a DensePredictor

    """

    # create sets of images
    u1 = np.random.rand(50, 50)
    v1 = np.random.rand(50, 50)
    u2 = np.random.rand(50, 50)
    v2 = np.random.rand(50, 50)
    u3 = np.random.rand(10, 10)
    v3 = np.random.rand(10, 10)
    dp1 = dense_predictor.DensePredictor(u1, v1)
    dp1_copy = dense_predictor.DensePredictor(u1, v1)
    dp2 = dense_predictor.DensePredictor(u2, v2)
    dp3 = dense_predictor.DensePredictor(u3, v3)

    # check dp1 and dp1_copy return equal
    assert dp1 == dp1_copy
    assert not dp1 == dp2
    assert not dp2 == dp3

    # check that NotImplemented is raised if compared to another object
    assert dp1.__eq__(4) == NotImplemented


def test_overload_add_operator_sums_correctly():
    """
    This test function is the check that the contents of
    dp3 = dp1 + dp2
    is mathematically correct
    i.e. checking that it has added correctly
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2)
    dp2 = dense_predictor.DensePredictor(u2, u3)
    dp3 = dp1 + dp2
    assert np.allclose(dp3.u, u1 + u2)
    assert np.allclose(dp3.v, u2 + u3)


def test_overload_add_throws_error_if_masks_different():
    """
    This test function checks that if two different masks are passed that
    a ValueErorr is raised
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u1, u2)
    with pytest.raises(ValueError):
        dp1 + dp2


def test_overload_add_applies_mask_correctly():
    """
    This test method is to check that the mask is applied correctly
    implying that the mask is stored in the new object and that all
    locations where the mask is defined, that the output is 0
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u2, u3, mask1)
    dp3 = dp1 + dp2

    exp_dp = dense_predictor.DensePredictor(u1 + u2, u2 + u3, mask1)
    assert dp3 == exp_dp


def test_overload_sub_operator_sums_correctly():
    """
    This test function is the check that the contents of
    dp3 = dp1 - dp2
    is mathematically correct
    i.e. checking that it has subtracted correctly
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2)
    dp2 = dense_predictor.DensePredictor(u2, u3)
    dp3 = dp1 - dp2
    assert np.allclose(dp3.u, u1 - u2)
    assert np.allclose(dp3.v, u2 - u3)


def test_overload_sub_throws_error_if_masks_different():
    """
    This test function checks that if two different masks are passed that
    a ValueErorr is raised
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u1, u2)
    with pytest.raises(ValueError):
        dp1 - dp2


def test_overload_sub_applies_mask_correctly():
    """
    This test method is to check that the mask is applied correctly
    implying that the mask is stored in the new object and that all
    locations where the mask is defined, that the output is 0
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u2, u3, mask1)
    dp3 = dp1 - dp2

    exp_dp = dense_predictor.DensePredictor(u1 - u2, u2 - u3, mask1)
    assert dp3 == exp_dp


def test_overload_mul_operator_sums_correctly():
    """
    This test function is the check that the contents of
    dp3 = dp1 * dp2
    is mathematically correct
    i.e. checking that it has multiplied correctly
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2)
    dp2 = dense_predictor.DensePredictor(u2, u3)
    dp3 = dp1 * dp2
    assert np.allclose(dp3.u, u1 * u2)
    assert np.allclose(dp3.v, u2 * u3)


def test_overload_mul_throws_error_if_masks_different():
    """
    This test function checks that if two different masks are passed that
    a ValueErorr is raised
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u1, u2)
    with pytest.raises(ValueError):
        dp1 * dp2


def test_overload_mul_applies_mask_correctly():
    """
    This test method is to check that the mask is applied correctly
    implying that the mask is stored in the new object and that all
    locations where the mask is defined, that the output is 0
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u2, u3, mask1)
    dp3 = dp1 * dp2

    exp_dp = dense_predictor.DensePredictor(u1 * u2, u2 * u3, mask1)
    assert dp3 == exp_dp


def test_overload_div_operator_sums_correctly():
    """
    This test function is the check that the contents of
    dp3 = dp1 / dp2
    is mathematically correct
    i.e. checking that it has multiplied correctly
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2)
    dp2 = dense_predictor.DensePredictor(u2, u3)
    dp3 = dp1 / dp2
    assert np.allclose(dp3.u, u1 / u2)
    assert np.allclose(dp3.v, u2 / u3)


def test_overload_div_throws_error_if_masks_different():
    """
    This test function checks that if two different masks are passed that
    a ValueErorr is raised
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u1, u2)
    with pytest.raises(ValueError):
        dp1 / dp2


def test_overload_div_applies_mask_correctly():
    """
    This test method is to check that the mask is applied correctly
    implying that the mask is stored in the new object and that all
    locations where the mask is defined, that the output is 0
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))
    u3 = np.arange(201, 282).reshape((9, 9))
    mask1 = np.random.randint(0, 2, (9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
    dp2 = dense_predictor.DensePredictor(u2, u3, mask1)
    dp3 = dp1 / dp2

    exp_dp = dense_predictor.DensePredictor(u1 / u2, u2 / u3, mask1)
    assert dp3 == exp_dp


def test_apply_mask_sets_mask_regions_to_zero():
    """
    We want the 'apply mask' to set the appropriate region's to NaN
    """

    arr = np.random.rand(10, 10)
    mask = np.ones((10, 10))
    mask[(0, 1, 2, 3, 4), (0, 1, 2, 3, 4)] = 0
    dp = dense_predictor.DensePredictor(arr, arr * 2, mask)

    # set regions of the mask with 0 to nan
    dp.apply_mask()
    uExp = arr
    uExp[(0, 1, 2, 3, 4), (0, 1, 2, 3, 4)] = 0
    vExp = arr * 2
    vExp[(0, 1, 2, 3, 4), (0, 1, 2, 3, 4)] = 0
    assert np.allclose(dp.u, uExp)
    assert np.allclose(dp.v, vExp)
    assert np.allclose(dp.mask, mask)


def test_magnitude():
    """
    Check that the magnitude returns the euclidean norm of the velocity
    """

    u1 = np.arange(1, 82).reshape((9, 9))
    u2 = np.arange(101, 182).reshape((9, 9))

    dp1 = dense_predictor.DensePredictor(u1, u2)
    exp = np.sqrt(u1*u1 + u2*u2)

    assert np.allclose(dp1.magnitude(), exp)


def test_get_local_average_disp():
    """Checks that the displacement obtained from the local average using summed
    area tables is the same as calculating traditionally

    make sure that the mask is considered
    """

    u = np.random.rand(50, 50)
    v = np.random.rand(50, 50)
    mask = np.random.randint(0, 2, (50, 50))
    dp = dense_predictor.DensePredictor(u, v, mask)
    WS, rad = 33, 16

    u_exp, v_exp = np.zeros_like(u), np.zeros_like(v)
    u_act, v_act = np.zeros_like(u), np.zeros_like(v)

    for y in range(50):
        for x in range(50):
            dpx, dpy, mask_reg = dp.get_region(x, y, rad)
            n_elem = np.sum(mask_reg)
            u_exp[y, x] += (np.sum(dpx[mask_reg == 1]) / n_elem)
            v_exp[y, x] += (np.sum(dpy[mask_reg == 1]) / n_elem)

            u_act[y, x], v_act[y, x] = dp.get_local_avg_disp(x, y, rad)

    assert np.allclose(u_act, u_exp)
    assert np.allclose(v_act, v_exp)


def test_create_densepredictor_from_dimensions():
    """Check the default behaviour which is to create a densepredictor with all 
    0 values
    """

    dim = (100, 100)
    exp = dense_predictor.DensePredictor(np.zeros(dim), np.zeros(dim))

    act = dense_predictor.DensePredictor.from_dimensions(dim)

    assert act == exp


def test_create_densepredictor_from_dimensions_with_value():
    """Check that we can pass a value or tuple of values to the constructor and
    this is the values it will adopt
    """

    dim = (100, 100)
    uv_vals = (4, 7)
    exp = dense_predictor.DensePredictor(np.ones(dim)*uv_vals[0],
                                         np.ones(dim)*uv_vals[1])

    act = dense_predictor.DensePredictor.from_dimensions(dim, uv_vals)

    assert act == exp
