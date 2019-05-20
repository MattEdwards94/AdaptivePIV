import unittest
import corr_window
import numpy as np
import piv_image


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

    def test_initialisation_with_negative_x(self):
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


if __name__ == "__main__":
    unittest.main(buffer=True)
