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


if __name__ == "__main__":
    unittest.main(buffer=True)
