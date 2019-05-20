import unittest
import corr_window


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


if __name__ == "__main__":
    unittest.main(buffer=True)
