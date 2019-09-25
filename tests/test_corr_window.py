import pytest
import corr_window
import numpy as np
import piv_image
import math
import scipy.signal
import dense_predictor
import cyth_corr_window


def test_initialisation_correct_inputs():
    """
    This initialisation should just store the x,y,WS information
    """
    x, y, WS = 10, 15, 21
    cw = corr_window.CorrWindow(x, y, WS)
    assert cw.x == x
    assert cw.y == y
    assert cw.WS == WS


def test_initialisation_with_decimal_x():
    """
    Here we want to check that a value error is raised if the input location
    is a decimal
    """
    x, y, WS = 10.5, 15, 21
    with pytest.raises(ValueError):
        _ = corr_window.CorrWindow(x, y, WS)


def test_initialisation_with_decimal_y():
    """
    Here we want to check that a decimal input for y is dropped to int
    """
    x, y, WS = 10, 15.5, 21
    with pytest.raises(ValueError):
        _ = corr_window.CorrWindow(x, y, WS)


def test_initialisation_with_negative_x():
    """
    Although we can't test the upper limit of the x,y input coordinates
    (since we don't know the size of the image), we can at least check at
    this point that the x value is positive
    """
    x, y, WS = -5, 15, 33

    # check with negative x
    with pytest.raises(ValueError):
        corr_window.CorrWindow(x, y, WS)


def test_initialisation_with_negative_y():
    """
    Although we can't test the upper limit of the x,y input coordinates
    (since we don't know the size of the image), we can at least check at
    this point that the y value is positive
    """
    x, y, WS = 5, -15, 33
    # check with negative y
    with pytest.raises(ValueError):
        corr_window.CorrWindow(x, y, WS)


def test_initialisation_with_negative_WS():
    """
    The WS must also be positive or otherwise this doesn't make
    pyhsical sense
    """
    x, y, WS = 5, 15, -33

    # check with negative WS
    with pytest.raises(ValueError):
        corr_window.CorrWindow(x, y, WS)


def test_initialisation_with_non_odd_WS():
    """
    Here we want to check that non-odd WS's are caught
    """
    x, y = 10, 15

    # check with even WS
    with pytest.raises(ValueError):
        corr_window.CorrWindow(x, y, 22)

    # check with decimal WS
    with pytest.raises(ValueError):
        corr_window.CorrWindow(x, y, 21.4)


def test_rad():
    """
    Check that the radius is calculated correctly as the distance either side
    of the central pixel
    """
    x, y, WS = 10, 15, 21
    cw = corr_window.CorrWindow(x, y, WS)
    assert cw.rad == 10


def test_eq_method():
    """
    Checks that the eq method returns true if objects are equivalent
    also returns false if they are different
    """

    x, y, WS = 10, 15, 21
    cw = corr_window.CorrWindow(x, y, WS)
    same = corr_window.CorrWindow(x, y, WS)
    diff = corr_window.CorrWindow(x, y, WS=WS + 2)
    also_diff = corr_window.CorrWindow(x, y, WS)
    also_diff.u, also_diff.v = 1, 2
    assert cw == same
    assert cw != diff
    assert cw != also_diff


def test_prepare_correlation_window_subtracts_mean():
    """
    Checks that the mean of the correlation windows is subtracted
    """

    # create PIV image
    IA = np.arange(100).reshape((10, 10))
    IB = np.arange(100).reshape((10, 10)) * 2
    img = piv_image.PIVImage(IA, IB)

    # create correlation window
    x = 4
    y = 4
    WS = 5
    rad = int((WS - 1) * 0.5)
    cw = corr_window.CorrWindow(x, y, WS)

    # expected solution
    bfa = IA[y - rad:y + rad + 1, x - rad:x + rad + 1]
    bfb = IB[y - rad:y + rad + 1, x - rad:x + rad + 1]
    wsa_expected = bfa - np.mean(bfa)
    wsb_expected = bfb - np.mean(bfb)

    # compare to actual solution
    wsa, wsb, mask = cw.prepare_correlation_windows(img)
    assert np.allclose(wsa, wsa_expected)
    assert np.allclose(wsb, wsb_expected)
    assert np.allclose(mask, np.ones_like(wsa))


def test_prepare_correlation_window_subtracts_applies_mask():
    """
    Checks that if a mask is present, then the subtracted mean only
    considers the non-masked intensity values

    furthermore, we want to make sure that the masked intensity values are
    set to 0
    """

    # create PIV image
    IA = np.arange(100).reshape((10, 10))
    IB = np.arange(100).reshape((10, 10)) * 2
    imgMask = np.random.randint(0, 2, (10, 10))
    img = piv_image.PIVImage(IA, IB, imgMask)

    # create correlation window
    x = 4
    y = 4
    WS = 5
    rad = int((WS - 1) * 0.5)
    cw = corr_window.CorrWindow(x, y, WS)

    # expected solution
    bfa = IA[y - rad:y + rad + 1, x - rad:x + rad + 1]
    bfb = IB[y - rad:y + rad + 1, x - rad:x + rad + 1]
    bfmask = imgMask[y - rad:y + rad + 1, x - rad:x + rad + 1]
    wsa_expected = bfa - np.mean(bfa[bfmask == 1])
    wsb_expected = bfb - np.mean(bfb[bfmask == 1])
    wsa_expected[bfmask == 0] = 0
    wsb_expected[bfmask == 0] = 0

    # compare to actual solution
    wsa, wsb, mask = cw.prepare_correlation_windows(img)
    assert np.allclose(wsa, wsa_expected)
    assert np.allclose(wsb, wsb_expected)
    assert np.allclose(mask, bfmask)


def test_calculate_correlation_map_is_padded_by_10_zeros():
    """
    We have found from previous testing that we need some minimum amount
    of zero padding to be present, let's check that is actually done

    here we're going to correlate a window of size 55
    This should cause the calculate_correlation_map() code to use a
    WS of 128
    We can then test the resulting corrmap to make sure this happened

    """

    # perform the correlation using the method being tested
    wsa = np.random.rand(33, 33)
    wsb = np.random.rand(33, 33)
    WS = 33
    rad = 16
    corrmap = corr_window.calculate_correlation_map(wsa, wsb, WS, rad)

    # now we want to manually perform the correlation with the padding
    # wsa needs flipping
    wsa = wsa[::-1, ::-1]

    # find the nearest power of 2 (assuming square windows)
    nPow2 = 2**(math.ceil(np.log2(WS + 10)))

    # perform the correlation
    corrmap_test = np.real(
        np.fft.ifftn(
            np.fft.fftn(wsa, [nPow2, nPow2])
            * np.fft.fftn(wsb, [nPow2, nPow2])
        )
    )

    # return the correct region
    idx = (np.arange(WS) + rad) % nPow2
    corrmap_test = corrmap_test[np.ix_(idx, idx)]

    assert np.allclose(corrmap, corrmap_test)


def test_calculate_correlation_map_all_real():
    """
    If the correlation map isn't real then it causes knock on effects to
    e.g. sub pixel fitting
    """

    # perform the correlation using the method being tested
    wsa = np.random.rand(33, 33)
    wsb = np.random.rand(33, 33)
    WS = 33
    rad = 16
    corrmap = corr_window.calculate_correlation_map(wsa, wsb, WS, rad)

    assert np.all(np.isreal(corrmap))


def test_calculate_correlation_is_same_as_dcc():
    """
    By definition it should be the same as direct cross correlation,
    we simply use fft's to make things much quicker
    """

    # perform the correlation using the method being tested
    wsa = np.random.rand(33, 33)
    wsb = np.random.rand(33, 33)
    WS = 33
    rad = 16
    corrmap = corr_window.calculate_correlation_map(wsa, wsb, WS, rad)

    # perform the dcc
    dcc_corrmap = scipy.signal.correlate2d(wsb, wsa, 'same')
    print(corrmap)
    print(dcc_corrmap)
    assert np.allclose(corrmap, dcc_corrmap)


def test_correlate_checks_if_location_is_masked():
    """
    We don't want to correlate if the location is masked, so if it is masked
    return NaN
    """

    # create image object with a masked region
    IA, IB = np.random.rand(100, 100), np.random.rand(100, 100)
    # mask on the right
    left, right = np.ones((100, 50)), np.zeros((100, 50))
    mask = np.hstack((left, right))
    img = piv_image.PIVImage(IA, IB, mask)
    u, v = np.zeros((100, 100)), np.zeros((100, 100))
    dp = dense_predictor.DensePredictor(u, v, mask)

    # create corr window in masked region and check for NaN response
    x, y, WS = 75, 25, 31
    cw = corr_window.CorrWindow(x, y, WS)
    u, v, snr = cw.correlate(img, dp)

    assert u == 0
    assert v == 0
    assert snr == 0


def test_correlate_combines_with_densepredictor():
    """
    Need to make sure that the average of the local densepredictor is
    combined with the correlation displacement
    """

    # create PIV image
    IA = np.random.rand(100, 100)
    IB = np.random.rand(100, 100)
    img = piv_image.PIVImage(IA, IB)

    # create correlation window
    x = 50
    y = 50
    WS = 33
    cw = corr_window.CorrWindow(x, y, WS)

    # correlate with no densepredictor
    u_zeros = np.zeros((100, 100))
    v_zeros = np.zeros((100, 100))
    dp_zeros = dense_predictor.DensePredictor(u_zeros, v_zeros)
    u1, v1, snr1 = cw.correlate(img, dp_zeros)

    # correlate with known non-zero densepredictor
    u_five = 5 * np.ones((100, 100))
    v_three = 3 * np.ones((100, 100))
    dp_non_zero = dense_predictor.DensePredictor(u_five, v_three)
    u2, v2, snr2 = cw.correlate(img, dp_non_zero)

    assert (u1 + 5) == u2
    assert (v1 + 3) == v2
    assert snr1 == snr2


def test_displacement_is_stored_in_object():
    """
    Although we may want to access the returned variables, we also need to
    store them in the object for future reference
    """

    # create PIV image
    IA = np.random.rand(100, 100)
    IB = np.random.rand(100, 100)
    img = piv_image.PIVImage(IA, IB)

    # create correlation window
    x = 50
    y = 50
    WS = 33
    cw = corr_window.CorrWindow(x, y, WS)

    # correlate
    u_zeros = np.zeros((100, 100))
    v_zeros = np.zeros((100, 100))
    dp_zeros = dense_predictor.DensePredictor(u_zeros, v_zeros)
    u1, v1, snr1 = cw.correlate(img, dp_zeros)

    # check they are stored
    assert u1 == cw.u
    assert v1 == cw.v
    assert snr1 == cw.SNR


def test_get_displacement_from_corrmap_catches_if_max_is_on_edge():
    """
    Tests that if the max is on the edge of the domain, then it is
    identified as being wrong and will return 0 displacement and SNR = 1
    """

    x, y, WS = 10, 10, 5
    cw = corr_window.CorrWindow(x, y, WS)

    # 4 test cases, i, j = 0 | WS
    a = np.zeros((5, 5))

    # case 1, i = 0 (top row)
    a[0, 2] = 1
    u, v, SNR = cw.get_displacement_from_corrmap(a)
    assert u == 0
    assert v == 0
    assert SNR == 1
    a[0, 2] = 0

    # case 2, i = WS (bottom row)
    a[-1, 2] = 1
    u, v, SNR = cw.get_displacement_from_corrmap(a)
    assert u == 0
    assert v == 0
    assert SNR == 1
    a[-1, 2] = 0

    # case 3, j = 0
    a[2, 0] = 1
    u, v, SNR = cw.get_displacement_from_corrmap(a)
    assert u == 0
    assert v == 0
    assert SNR == 1
    a[2, 0] = 0

    # case 4, j = WS
    a[2, -1] = 1
    u, v, SNR = cw.get_displacement_from_corrmap(a)
    assert u == 0
    assert v == 0
    assert SNR == 1
    a[2, -1] = 0


def test_get_displacement_from_corrmap_central_peak():
    # create a dummy correlation
    corrmap = np.zeros((7, 7))
    corrmap[3, 3] = 1

    # run through the script to get the displacement
    u, v, SNR = cyth_corr_window.get_displacement_from_corrmap(corrmap, 7, 3)

    assert u == 0
    assert v == 0


def test_changing_values_in_get_displacement_doesnt_change_outer():
    """
    we need to set the values around the corrmap peak to NaN such that we
    can find the location of the second peak

    Just a check to make sure that this doesn't affect the outer variable
    """

    x, y, WS = 10, 10, 5
    cw = corr_window.CorrWindow(x, y, WS)

    a = np.zeros((5, 5))
    b = np.zeros((5, 5))

    a[2, 2] = 1
    b[2, 2] = 1
    u, v, SNR = cw.get_displacement_from_corrmap(a)
    print(a)
    print(b)
    assert np.allclose(a, b)


def test_get_corrwindow_scaling_is_equal_to_convolution():
    """
    Looking at Raffel pg. 162, we know that we need to normalise the
    correlation map to avoid the bias towards the origin.
    The 'correct' procedure is to convolute the image sampling function
    which typically has unity weight and therefore ends up as a triangle
    function.

    However, we only need the values around the correlation peak and so
    this is an extremely inefficient way to go about calculating the
    scaling terms

    The function get_corrwindow_scaling will aim to get these scalings in
    an efficient manner, but we need to make sure it is equivalent to a
    convolution
    """

    # arbitrary size window
    WS = 33
    rad = int((WS - 1) * 0.5)
    a = np.ones((WS, WS))
    b = np.ones((WS, WS))
    convolved = scipy.signal.convolve2d(a, b, 'same')
    convolved /= (WS * WS)

    # test at the centre that the value is 1
    assert convolved[rad, rad] == 1

    # now use function the get value at centre and check it is 1
    scaling = corr_window.get_corrwindow_scaling(rad, rad, WS, rad)
    assert scaling[1, 1] == 1

    # test the values at a particular region
    # this is the top left in terms of the matrix, bottom left in terms
    # of the image
    scaling = corr_window.get_corrwindow_scaling(rad - 5, rad - 7, WS, rad)
    assert np.allclose(1 / scaling,
                       convolved[rad - 6:rad - 3, rad - 8:rad - 5])

    # bottom right (as above)
    scaling = corr_window.get_corrwindow_scaling(rad + 5, rad + 7, WS, rad)
    assert np.allclose(1 / scaling,
                       convolved[rad + 4:rad + 7, rad + 6:rad + 9])


def test_u_and_v_are_init_to_none():
    """
    The values for u and v need to be set in the object at initialisation
    """

    x, y, WS = 10, 15, 33
    cw = corr_window.CorrWindow(x, y, WS)

    assert np.isnan(cw.u)
    assert np.isnan(cw.v)


def test_u_and_v_pre_validation_are_initialised_to_none():
    """
    These values need initialising to None so that we can store the values
    of u and v before validation - i.e. the pure output of the correlation
    """

    x, y, WS = 10, 15, 33
    cw = corr_window.CorrWindow(x, y, WS)

    assert np.isnan(cw.u_pre_validation)
    assert np.isnan(cw.v_pre_validation)


def test_flag_initialised_to_none():
    """
    Check that the flag is initialised to none
    """

    x, y, WS = 10, 15, 33
    cw = corr_window.CorrWindow(x, y, WS)

    assert np.isnan(cw.flag)


def test_corrwindow_list_from_locations():
    """
    Test the list is created according to the input values
    """

    xv, yv, WS = np.arange(100), np.arange(100), np.ones((100,)) * 33
    expList = []
    for x, y, WS_ in zip(xv, yv, WS):
        cw = corr_window.CorrWindow(x, y, WS_)
        expList.append(cw)

    actlist = corr_window.corrWindow_list(xv, yv, WS)

    assert actlist == expList


def test_corrwindow_list_with_scalar_WS():
    """
    Test that the list is properly initialised if only a single WS value is
    passed in
    """

    xv, yv, WS = np.arange(100), np.arange(100), 33
    expList = []
    for x, y in zip(xv, yv):
        cw = corr_window.CorrWindow(x, y, WS)
        expList.append(cw)

    actlist = corr_window.corrWindow_list(xv, yv, WS)

    assert actlist == expList
