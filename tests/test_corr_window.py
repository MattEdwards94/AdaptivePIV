import unittest
import corr_window
import numpy as np
import piv_image
import math
import scipy.signal
import dense_predictor


class TestCorrWindow(unittest.TestCase):
    def test_initialisation_correct_inputs(self):
        """
        This initialisation should just store the x,y,WS information
        """

        x = 10
        y = 15
        WS = 21
        cw = corr_window.CorrWindow(x, y, WS)
        self.assertEqual(cw.x, x)
        self.assertEqual(cw.y, y)
        self.assertEqual(cw.WS, WS)

    def test_initialisation_with_decimal_x(self):
        """
        Here we want to check that a decimal input for x is dropped to int
        """

        x = 10.5
        y = 15
        WS = 21
        cw = corr_window.CorrWindow(x, y, WS)
        self.assertEqual(cw.x, 10)
        self.assertEqual(cw.y, y)
        self.assertEqual(cw.WS, WS)

    def test_initialisation_with_decimal_y(self):
        """
        Here we want to check that a decimal input for y is dropped to int
        """

        x = 10
        y = 15.5
        WS = 21
        cw = corr_window.CorrWindow(x, y, WS)
        self.assertEqual(cw.x, x)
        self.assertEqual(cw.y, 15)
        self.assertEqual(cw.WS, WS)

    def test_initialisation_with_negative_x(self):
        """
        Although we can't test the upper limit of the x,y input coordinates
        (since we don't know the size of the image), we can at least check at
        this point that the x value is positive
        """

        x = -5
        y = 15
        WS = 33

        # check with negative x
        with self.assertRaises(ValueError):
            corr_window.CorrWindow(x, y, WS)

    def test_initialisation_with_negative_y(self):
        """
        Although we can't test the upper limit of the x,y input coordinates
        (since we don't know the size of the image), we can at least check at
        this point that the y value is positive
        """

        x = 5
        y = -15
        WS = 33

        # check with negative y
        with self.assertRaises(ValueError):
            corr_window.CorrWindow(x, y, WS)

    def test_initialisation_with_negative_WS(self):
        """
        The WS must also be positive or otherwise this doesn't make
        pyhsical sense
        """

        x = 5
        y = 15
        WS = -33

        # check with negative WS
        with self.assertRaises(ValueError):
            corr_window.CorrWindow(x, y, WS)

    def test_initialisation_with_non_odd_WS(self):
        """
        Here we want to check that non-odd WS's are caught
        """

        x = 10
        y = 15

        # check with even WS
        with self.assertRaises(ValueError):
            corr_window.CorrWindow(x, y, 22)

        # check with decimal WS
        with self.assertRaises(ValueError):
            corr_window.CorrWindow(x, y, 21.4)

    def test_rad_is_stored(self):
        """
        A useful variable is the rad so it makes sense to save this instead
        of constantly having to calculate it
        """
        x = 10
        y = 15
        WS = 21
        cw = corr_window.CorrWindow(x, y, WS)
        self.assertEqual(cw.rad, int((WS - 1) * 0.5))

    def test_prepare_correlation_window_subtracts_mean(self):
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
        self.assertTrue(np.allclose(wsa, wsa_expected))
        self.assertTrue(np.allclose(wsb, wsb_expected))
        self.assertTrue(np.allclose(mask, np.ones_like(wsa)))

    def test_prepare_correlation_window_subtracts_applies_mask(self):
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
        self.assertTrue(np.allclose(wsa, wsa_expected))
        self.assertTrue(np.allclose(wsb, wsb_expected))
        self.assertTrue(np.allclose(mask, bfmask))

    def test_calculate_correlation_map_is_padded_by_10_zeros(self):
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

        self.assertTrue(np.allclose(corrmap, corrmap_test))

    def test_calculate_correlation_map_all_real(self):
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

        self.assertTrue(np.all(np.isreal(corrmap)))

    def test_calculate_correlation_is_same_as_dcc(self):
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
        self.assertTrue(np.allclose(corrmap, dcc_corrmap))

    def test_correlate_combines_with_densepredictor(self):
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

        self.assertEqual(u1 + 5, u2)
        self.assertEqual(v1 + 3, v2)
        self.assertEqual(snr1, snr2)

    def test_displacement_is_stored_in_object(self):
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
        self.assertEqual(u1, cw.u)
        self.assertEqual(v1, cw.v)
        self.assertEqual(snr1, cw.SNR)

    def test_get_displacement_from_corrmap_catches_if_max_is_on_edge(self):
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
        self.assertEqual(u, 0)
        self.assertEqual(v, 0)
        self.assertEqual(SNR, 1)
        a[0, 2] = 0

        # case 2, i = WS (bottom row)
        a[-1, 2] = 1
        u, v, SNR = cw.get_displacement_from_corrmap(a)
        self.assertEqual(u, 0)
        self.assertEqual(v, 0)
        self.assertEqual(SNR, 1)
        a[-1, 2] = 0

        # case 3, j = 0
        a[2, 0] = 1
        u, v, SNR = cw.get_displacement_from_corrmap(a)
        self.assertEqual(u, 0)
        self.assertEqual(v, 0)
        self.assertEqual(SNR, 1)
        a[2, 0] = 0

        # case 4, j = WS
        a[2, -1] = 1
        u, v, SNR = cw.get_displacement_from_corrmap(a)
        self.assertEqual(u, 0)
        self.assertEqual(v, 0)
        self.assertEqual(SNR, 1)
        a[2, -1] = 0

    def test_changing_values_in_get_displacement_doesnt_change_outer(self):
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
        self.assertTrue(np.allclose(a, b))

    def test_get_corrwindow_scaling_is_equal_to_convolution(self):
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
        self.assertEqual(convolved[rad, rad], 1)

        # now use function the get value at centre and check it is 1
        scaling = corr_window.get_corrwindow_scaling(rad, rad, WS, rad)
        self.assertEqual(scaling[1, 1], 1)

        # test the values at a particular region
        # this is the top left in terms of the matrix, bottom left in terms
        # of the image
        scaling = corr_window.get_corrwindow_scaling(rad - 5, rad - 7, WS, rad)
        self.assertTrue(np.allclose(
            1 / scaling, convolved[rad - 6:rad - 3, rad - 8:rad - 5]))

        # bottom right (as above)
        scaling = corr_window.get_corrwindow_scaling(rad + 5, rad + 7, WS, rad)
        self.assertTrue(np.allclose(
            1 / scaling, convolved[rad + 4:rad + 7, rad + 6:rad + 9]))

    def test_u_and_v_are_init_to_nan(self):
        """
        The values for u and v need to be set in the object at initialisation
        """

        x, y, WS = 10, 15, 33
        cw = corr_window.CorrWindow(x, y, WS)

        self.assertTrue(np.isnan(cw.u))
        self.assertTrue(np.isnan(cw.v))

    def test_u_and_v_pre_validation_are_initialised_to_nan(self):
        """
        These values need initialising to NaN so that we can store the values
        of u and v before validation - i.e. the pure output of the correlation
        """

        x, y, WS = 10, 15, 33
        cw = corr_window.CorrWindow(x, y, WS)

        self.assertTrue(np.isnan(cw.u_pre_validation))
        self.assertTrue(np.isnan(cw.v_pre_validation))

    def test_flag_initialised_to_nan(self):
        """
        Check that the flag is initialised
        """

        x, y, WS = 10, 15, 33
        cw = corr_window.CorrWindow(x, y, WS)

        self.assertTrue(np.isnan(cw.flag))


if __name__ == "__main__":
    unittest.main(buffer=True)
