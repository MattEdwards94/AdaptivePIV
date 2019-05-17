import unittest
import numpy as np
import dense_predictor


class testDensePredictor(unittest.TestCase):
    def test_initialisation_with_mask(self):
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
        self.assertTrue(np.alltrue(dp.u == u))
        self.assertTrue(np.alltrue(dp.v == v))
        self.assertTrue(np.alltrue(dp.mask == mask))

    def test_initialisation_without_mask(self):
        """
        Checks that u and v are stored correctly, and that a mask of
        ones is created with the same shape as u and v
        """
        u = np.random.rand(100, 100)
        v = np.random.rand(100, 100)
        dp = dense_predictor.DensePredictor(u, v)

        # check the saved u, v, mask. mask should be ones
        self.assertTrue(np.alltrue(dp.u == u))
        self.assertTrue(np.alltrue(dp.v == v))
        self.assertTrue(np.alltrue(dp.mask == np.ones((100, 100))))

    def test_initialisation_with_mask_checks_size(self):
        """
        u, v, mask must have the same shape
        """
        u = np.random.rand(100, 100)
        v = np.random.rand(110, 110)
        mask = np.random.randint(0, 2, (100, 100))

        # check u vs v
        with self.assertRaises(ValueError):
            dp = dense_predictor.DensePredictor(u, v, mask)

        # check mask
        with self.assertRaises(ValueError):
            dp = dense_predictor.DensePredictor(v, v, mask)

    def test_initialisation_without_mask_checks_size(self):
        """
        checks that u and v sizes are still compared even if mask isn't passed
        """
        u = np.random.rand(100, 100)
        v = np.random.rand(110, 110)

        # check u vs v
        with self.assertRaises(ValueError):
            dp = dense_predictor.DensePredictor(u, v)

    def test_initialisation_saves_mask_status(self):
        """
        If there has been a mask passed we want to save this information as
        an easily checkable bool
        """
        u = np.random.rand(100, 100)
        v = np.random.rand(100, 100)
        mask = np.random.randint(0, 2, (100, 100))
        dp = dense_predictor.DensePredictor(u, v)
        self.assertFalse(dp.has_mask)
        dp2 = dense_predictor.DensePredictor(u, v, mask)
        self.assertTrue(dp2.has_mask)

    def test_image_dimensions_are_captured(self):
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
        self.assertEqual(dp.n_rows, 50)
        self.assertEqual(dp.n_cols, 100)
        self.assertEqual(dp.img_dim, [50, 100])

    def test_get_region_returns_correct_region(self):
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
        self.assertTrue(np.allclose(u, exp_arr))
        self.assertTrue(np.allclose(v, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

        # what happens if we truncate to the top left:
        u, v, mask = dp.get_region(1, 1, 2, truncate=True)
        exp_arr = np.array([[1, 2, 3, 4],
                            [7, 8, 9, 10],
                            [13, 14, 15, 16],
                            [19, 20, 21, 22]])
        self.assertTrue(np.allclose(u, exp_arr))
        self.assertTrue(np.allclose(v, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

        # if we pad with 0's instead
        u, v, mask = dp.get_region(1, 1, 2, truncate=False)
        exp_arr = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 2, 3, 4],
                            [0, 7, 8, 9, 10],
                            [0, 13, 14, 15, 16],
                            [0, 19, 20, 21, 22]])
        print(u)
        self.assertTrue(np.allclose(u, exp_arr))
        self.assertTrue(np.allclose(v, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

        # what happens if we truncate to the bottom right:
        u, v, mask = dp.get_region(4, 4, 2, truncate=True)
        exp_arr = np.array([[15, 16, 17, 18],
                            [21, 22, 23, 24],
                            [27, 28, 29, 30],
                            [33, 34, 35, 36]])
        print(u)
        self.assertTrue(np.allclose(u, exp_arr))
        self.assertTrue(np.allclose(v, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

        # if we pad with 0's
        u, v, mask = dp.get_region(4, 4, 2, truncate=False)
        exp_arr = np.array([[15, 16, 17, 18, 0],
                            [21, 22, 23, 24, 0],
                            [27, 28, 29, 30, 0],
                            [33, 34, 35, 36, 0],
                            [0, 0, 0, 0, 0]])
        print(u)
        self.assertTrue(np.allclose(u, exp_arr))
        self.assertTrue(np.allclose(v, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

    def test_eq_method_evaluates_correctly(self):
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
        self.assertEqual(dp1, dp1_copy)
        self.assertNotEqual(dp1, dp2)
        self.assertNotEqual(dp2, dp3)

        # check that NotImplemented is raised if compared to another object
        self.assertEqual(dp1.__eq__(4), NotImplemented)

    def test_overload_add_operator_sums_correctly(self):
        """
        This test function is the check that the contents of
        dp3 = dp1 + dp2
        is mathematically correct
        i.e. checking that it has added correctly
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        dp3 = dp1 + dp2
        exp_dp = dense_predictor.DensePredictor(u1 + u2, u2 + u3)
        self.assertTrue(dp3 == exp_dp)

    def test_overload_add_takes_mask_from_lhs(self):
        """
        This test function checks that, in the presence of two distinct masks
        that the mask from the lhs is taken.
        Also checks that a warning is raised in this case
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        mask1 = np.random.randint(0, 2, (9, 9))
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        with self.assertWarns(UserWarning):
            dp3 = dp1 + dp2

        exp_dp = dense_predictor.DensePredictor(u1 + u2, u2 + u3, mask1)
        # areas in dp1 with a mask will still possess a mask, and so will be 0
        exp_dp.apply_mask()
        self.assertTrue(dp3 == exp_dp)

        with self.assertWarns(UserWarning):
            dp3 = dp2 + dp1

        # areas in dp1 with a mask will add 0 to the relavent location in dp2
        u1_temp = np.array(u1)
        u1_temp[mask1 == 0] = 0
        u2_temp = np.array(u2)
        u2_temp[mask1 == 0] = 0
        exp_dp = dense_predictor.DensePredictor(
            u2 + u1_temp, u3 + u2_temp, np.ones((9, 9)))
        self.assertTrue(dp3 == exp_dp)

    def test_overload_minus_operator_sums_correctly(self):
        """
        This test function is the check that the contents of
        dp3 = dp1 - dp2
        is mathematically correct
        i.e. checking that it has subtracted correctly
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        dp3 = dp1 - dp2
        exp_dp = dense_predictor.DensePredictor(u1 - u2, u2 - u3)
        self.assertTrue(dp3 == exp_dp)

    def test_overload_minus_takes_mask_from_lhs(self):
        """
        This test function checks that, in the presence of two distinct masks
        that the mask from the lhs is taken.
        Also checks that a warning is raised in this case
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        mask1 = np.random.randint(0, 2, (9, 9))
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        with self.assertWarns(UserWarning):
            dp3 = dp1 - dp2

        exp_dp = dense_predictor.DensePredictor(u1 - u2, u2 - u3, mask1)
        exp_dp.apply_mask()
        self.assertTrue(dp3 == exp_dp)

        with self.assertWarns(UserWarning):
            dp3 = dp2 - dp1

        # areas in dp1 with a mask will add 0 to the relavent location in dp2
        u1_temp = np.array(u1)
        u1_temp[mask1 == 0] = 0
        u2_temp = np.array(u2)
        u2_temp[mask1 == 0] = 0

        exp_dp = dense_predictor.DensePredictor(
            u2 - u1_temp, u3 - u2_temp, np.ones((9, 9)))
        self.assertTrue(dp3 == exp_dp)

    def test_overload_multiply_operator_sums_correctly(self):
        """
        This test function is the check that the contents of
        dp3 = dp1 * dp2
        is mathematically correct
        i.e. checking that it has multiplied correctly
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        dp3 = dp1 * dp2

        exp_dp = dense_predictor.DensePredictor(u1 * u2, u2 * u3)
        self.assertTrue(dp3 == exp_dp)

    def test_overload_multiply_takes_mask_from_lhs(self):
        """
        This test function checks that, in the presence of two distinct masks
        that the mask from the lhs is taken.
        Also checks that a warning is raised in this case
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        mask1 = np.random.randint(0, 2, (9, 9))
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        with self.assertWarns(UserWarning):
            dp3 = dp1 * dp2

        exp_dp = dense_predictor.DensePredictor(u1 * u2, u2 * u3, mask1)
        exp_dp.apply_mask()
        self.assertTrue(dp3 == exp_dp)

        with self.assertWarns(UserWarning):
            dp3 = dp2 * dp1

        u1_temp = np.array(u1)
        u1_temp[mask1 == 0] = 0
        u2_temp = np.array(u2)
        u2_temp[mask1 == 0] = 0

        exp_dp = dense_predictor.DensePredictor(
            u2 * u1_temp, u3 * u2_temp, np.ones((9, 9)))
        exp_dp.apply_mask()
        self.assertTrue(dp3 == exp_dp)

    def test_overload_divide_operator_sums_correctly(self):
        """
        This test function is the check that the contents of
        dp3 = dp1 + dp2
        is mathematically correct
        i.e. checking that it has divided correctly
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        dp3 = dp1 / dp2

        exp_dp = dense_predictor.DensePredictor(u1 / u2, u2 / u3)
        exp_dp.apply_mask()
        self.assertTrue(dp3 == exp_dp)

    def test_overload_divide_takes_mask_from_lhs(self):
        """
        This test function checks that, in the presence of two distinct masks
        that the mask from the lhs is taken.
        Also checks that a warning is raised in this case
        """

        u1 = np.arange(1, 82)
        u2 = np.arange(101, 182)
        u3 = np.arange(201, 282)
        mask1 = np.random.randint(0, 2, (9, 9))
        u1 = np.reshape(u1, (9, 9))
        u2 = np.reshape(u2, (9, 9))
        u3 = np.reshape(u3, (9, 9))

        dp1 = dense_predictor.DensePredictor(u1, u2, mask1)
        dp2 = dense_predictor.DensePredictor(u2, u3)
        with self.assertWarns(UserWarning):
            dp3 = dp1 / dp2

        exp_dp = dense_predictor.DensePredictor(u1 / u2, u2 / u3, mask1)
        exp_dp.apply_mask()
        self.assertTrue(dp3 == exp_dp)

        with self.assertWarns(UserWarning):
            dp3 = dp2 / dp1

        u1_temp = np.array(u1)
        u1_temp[mask1 == 0] = 0
        u2_temp = np.array(u2)
        u2_temp[mask1 == 0] = 0

        with np.errstate(divide='ignore'):
            exp_dp = dense_predictor.DensePredictor(
                u2 / u1_temp, u3 / u2_temp, np.ones((9, 9)))
        exp_dp.apply_mask()
        self.assertTrue(dp3 == exp_dp)

    def test_apply_mask_sets_mask_regions_to_zero(self):
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
        self.assertTrue(np.allclose(dp.u, uExp))
        self.assertTrue(np.allclose(dp.v, vExp))
        self.assertTrue(np.allclose(dp.mask, mask))


if __name__ == '__main__':
    unittest.main()
