import unittest
import numpy as np
import distribution
import corr_window


class TestDistributions(unittest.TestCase):
    def setUp(self):
        """
        create a dummy correlation window and a dummy list
        """

        self.cw = corr_window.CorrWindow(x=10, y=15, WS=21)
        self.cwList = [corr_window.CorrWindow(x=20, y=30, WS=41),
                       corr_window.CorrWindow(x=30, y=45, WS=51),
                       corr_window.CorrWindow(x=40, y=60, WS=31), ]

    def test_initialisation_with_no_inputs(self):
        """
        Test that creating a Distribution with no inputs creates a valid object
        with self.windows initialised
        """

        dist = distribution.Distribution()

        # check that there is a list initialised and empty
        self.assertEqual(dist.windows, [])

    def test_initialisation_with_a_single_corr_window(self):
        """
        I don't think this should cause any issues, but perhaps (as in matlab)
        having 1 item vs a list of items might cause issues
        check that the windows property is a list
        """

        dist = distribution.Distribution(self.cw)
        self.assertEqual(type(dist.windows), list)

    def test_initialisation_with_a_list_of_corr_windows(self):
        """
        check that it initialises properly - this should just be storing
        the list internally in the object,
        """

        dist = distribution.Distribution(self.cwList)
        self.assertEqual(type(dist.windows), list)

    def test_initialisation_with_list_is_shallow_only(self):
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

        dist = distribution.Distribution(self.cwList)
        self.assertEqual(type(dist.windows), list)
        self.assertFalse(dist.windows is self.cwList)
        self.assertTrue(dist.windows[0] == self.cwList[0])

    def test_n_windows_returns_number_of_windows(self):
        """
        Checks it returns the correct number of windows for empty, 1 and many
        """

        dist = distribution.Distribution()
        self.assertEqual(dist.n_windows(), 0)

        dist = distribution.Distribution(self.cw)
        self.assertEqual(dist.n_windows(), 1)

        dist = distribution.Distribution(self.cwList)
        self.assertEqual(dist.n_windows(), 3)

    def test_get_values_x_returns_list_of_x_locations(self):
        """
        calling e.g. distribution.values("x") returns a list of the x
        coordinates.
        Although perhaps a tuple would make sense here since we aren't
        going to modify the locations this way, a list seems quicker
        """

        dist = distribution.Distribution(self.cwList)
        self.assertEqual(dist.values("x"), [20, 30, 40])

    def test_get_values_y_returns_list_of_y_locations(self):
        """
        calling e.g. distribution.values("y") returns a list of the y
        coordinates.
        Although perhaps a tuple would make sense here since we aren't
        going to modify the locations this way, a list seems quicker
        """

        dist = distribution.Distribution(self.cwList)
        self.assertEqual(dist.values("y"), [30, 45, 60])

    def test_get_values_u_returns_list_of_u_locations(self):
        """
        calling e.g. distribution.values("u") returns a list of the u
        displacements.
        Although perhaps a tuple would make sense here since we aren't
        going to modify the locations this way, a list seems quicker
        """

        dist = distribution.Distribution(self.cwList)
        self.assertEqual(dist.values("u"), [np.nan, np.nan, np.nan])

        for cw in dist.windows:
            cw.u = 10

        self.assertEqual(dist.values("u"), [10, 10, 10])

    def test_get_values_v_returns_list_of_v_locations(self):
        """
        calling e.g. distribution.values("v") returns a list of the v
        displacements.
        Although perhaps a tuple would make sense here since we aren't
        going to modify the locations this way, a list seems quicker
        """

        dist = distribution.Distribution(self.cwList)
        self.assertEqual(dist.values("v"), [np.nan, np.nan, np.nan])

        for cw in dist.windows:
            cw.v = 20

        self.assertEqual(dist.values("v"), [20, 20, 20])


if __name__ == "__main__":
    unittest.main(buffer=True)
