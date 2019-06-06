import unittest
import numpy as np
import distribution
import corr_window
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors


class TestDistributions(unittest.TestCase):
    def setUp(self):
        """
        create a dummy correlation window and a dummy list
        """

        self.cw = corr_window.CorrWindow(x=10, y=15, WS=21)
        self.cwList = [corr_window.CorrWindow(x=20, y=30, WS=41),
                       corr_window.CorrWindow(x=30, y=45, WS=51),
                       corr_window.CorrWindow(x=40, y=60, WS=31), ]
        self.dist = distribution.Distribution(self.cwList)

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
        self.assertTrue(np.allclose(dist.get_values("x"), [20, 30, 40]))

    def test_get_values_y_returns_list_of_y_locations(self):
        """
        calling e.g. distribution.get_values("y") returns a list of the y
        coordinates.
        Although perhaps a tuple would make sense here since we aren't
        going to modify the locations this way, a list seems quicker
        """

        dist = distribution.Distribution(self.cwList)
        self.assertTrue(np.allclose(dist.get_values("y"), [30, 45, 60]))

    def test_get_values_u_returns_list_of_u_locations(self):
        """
        calling e.g. distribution.get_values("u") returns a list of the u
        displacements.
        Although perhaps a tuple would make sense here since we aren't
        going to modify the locations this way, a list seems quicker
        """

        dist = distribution.Distribution(self.cwList)
        self.assertTrue(np.all(np.isnan(dist.get_values("u"))))

        for cw in dist.windows:
            cw.u = 10

        self.assertTrue(np.allclose(dist.get_values("u"), [10, 10, 10]))

    def test_get_values_v_returns_list_of_v_locations(self):
        """
        calling e.g. distribution.get_values("v") returns a list of the v
        displacements.
        Although perhaps a tuple would make sense here since we aren't
        going to modify the locations this way, a list seems quicker
        """

        dist = distribution.Distribution(self.cwList)
        self.assertTrue(np.all(np.isnan(dist.get_values("v"))))

        for cw in dist.windows:
            cw.v = 20

        self.assertTrue(np.allclose(dist.get_values("v"), [20, 20, 20]))

    def test_get_values_wrong_key_raises_error(self):
        """
        The expected behaviour is to raise a KeyError
        """

        dist = distribution.Distribution(self.cw)

        with self.assertRaises(KeyError):
            dist.get_values('WrongKey')

    def test_NMT_detection_selects_correct_neighbour_values(self):
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
        self.assertTrue(np.allclose(norm_exp, norm_act))

    def test_NMT_detection_all_uniform_returns_zeros(self):
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

        self.assertTrue(np.allclose(norm, np.zeros((100, ))))

    def test_validation_NMT_8NN_stores_old_value(self):
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
            dist.windows.append(cw)

        # now run the vector validation
        dist.validation_NMT_8NN()

        # get the flagged vectors
        flag = dist.get_values('flag')

        # now check that they all have the value they originally were given
        for cw, ui, vi, flagi in zip(dist.windows, u, v, flag):
            if flagi is True:
                # it's an outlier
                self.assertEqual(ui, cw.u_pre_validation)
                self.assertEqual(vi, cw.v_pre_validation)

    def test_outlier_replacement_replaces_0_if_all_neighbours_outliers(self):
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

        self.assertEqual(u[10], 0)
        self.assertEqual(v[10], 0)

    def test_outlier_replacement_is_median_of_valid_neighbours(self):
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
        self.assertEqual(u[10], u_exp)
        self.assertEqual(v[10], v_exp)

    def test_set_values_changes_CorrWindow_objects(self):
        """
        Checks that the method Distribution.set_values('prop', values)
        correctly updates the vales of the relevant property in the individual
        corrWindow objects stored by the distribution
        """

        dist = distribution.Distribution(self.cwList)

        dist.set_values('x', [2, 3, 4])
        print(dist.get_values('x'))

        self.assertTrue(np.allclose(dist.get_values('x'), [2, 3, 4]))

    def test_set_values_check_dimensions_of_values(self):
        """
        The number of values must be equal to the total number of corrwindows
        """

        dist = distribution.Distribution(self.cwList)

        with self.assertRaises(ValueError):
            dist.set_values('x', [2, 3])

    def test_set_values_wrong_key_raises_error(self):
        """
        If the wrong key is specified then a KeyError should be raised
        """

        dist = distribution.Distribution(self.cwList)

        with self.assertRaises(KeyError):
            dist.set_values('WrongKey', [1, 2, 3])
            print(dist.windows[0].__dict__)

    def test_interpolate_checks_method(self):
        """
        Checks the method passed is checked for validity
        """

        acceptable_options = ["struc_lin", "struc_cub"]

        unacceptable_options = ["not this", "or this", "cub_str"]

        # for item in acceptable_options:
        # check that no error is thrown
        # self.dist.interp_to_densepred(item, (10, 10))

        for item in unacceptable_options:
            # check that ValueError is raised
            with self.assertRaises(ValueError):
                self.dist.interp_to_densepred(item, (10, 10))

    def test_interpolate_checks_out_dimensions(self):
        """
        Checks that the output dimensions are checked to be positive integers
        """

        # should run fine
        # self.dist.interp_to_densepred('str_lin', (100, 100))

        # check decimals
        with self.assertRaises(ValueError):
            self.dist.interp_to_densepred('struc_lin', (4.5, 4.5))
        # check negatives
        with self.assertRaises(ValueError):
            self.dist.interp_to_densepred('struc_lin', (-4, 4))

    def test_linear_interpolate_onto_pixelgrid(self):
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

        self.assertTrue(np.allclose(u_int, u_exp))
        self.assertTrue(np.allclose(v_int, v_exp))

    def test_cubic_interpolate_onto_pixelgrid(self):
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

        self.assertTrue(np.allclose(u_int, u_exp))
        self.assertTrue(np.allclose(v_int, v_exp))


if __name__ == "__main__":
    unittest.main(buffer=True)
