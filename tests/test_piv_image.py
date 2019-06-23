import unittest
import piv_image
import numpy as np
import dense_predictor
import sym_filt
import image_info


class TestPIVImage(unittest.TestCase):
    def setUp(self):
        """Initialises an IA, IB and mask
        """
        img_size = (55, 55)
        self.IA = np.random.rand(img_size[1], img_size[0])
        self.IB = np.random.rand(img_size[1], img_size[0])
        self.mask = np.random.randint(0, 2, size=img_size)

    def test_initialisation_checks_image_size(self):
        """Checks that the inputs IA, IB must be of the same size otherwise
        a value error is thrown
        If mask is passed then this should also be the same size
        """

        # self.IA, IB, mask are all the same size so this should work fine
        piv_image.PIVImage(self.IA, self.IB, self.mask)

        # now change the size of IB and check an error is thrown
        size = np.shape(self.IA)
        bad_size_IB = np.random.rand(size[1] + 10, size[0] - 10)
        with self.assertRaises(ValueError):
            piv_image.PIVImage(self.IA, bad_size_IB, self.mask)

        # check that the size of mask if given is checked
        bad_mask = np.random.randint(0, 2, size=(size[1] + 10, size[0] - 10))
        with self.assertRaises(ValueError):
            piv_image.PIVImage(self.IA, self.IB, bad_mask)

    def test_mask_set_to_ones_if_not_passed(self):
        """The default value for mask is None. In this case, an empty mask of
        ones will be created with the same size as IA and IB
        """

        # if no mask is set, we should obtain an array of ones of size img_size
        img = piv_image.PIVImage(self.IA, self.IB)
        expected = np.ones(np.shape(self.IA))
        self.assertTrue(np.alltrue(img.mask == expected))

    def test_initialisation_assigns_intensities_correctly(self):
        """
        Check that IA is assigned to self.IA and that IB is assigned to self.IB
        """
        img = piv_image.PIVImage(self.IA, self.IB)
        self.assertTrue(np.allclose(self.IA, img.IA))

    def test_initialisation_saves_mask_status(self):
        """
        If there has been a mask passed we want to save this information as
        an easily checkable bool
        """

        img = piv_image.PIVImage(self.IA, self.IB)
        self.assertFalse(img.has_mask)
        img2 = piv_image.PIVImage(self.IA, self.IB, self.mask)
        self.assertTrue(img2.has_mask)

    def test_image_dimensions_are_captured(self):
        """check that the size of the image is captured into the variables
        n_rows
        n_cols
        img_dim
        """

        # use non-square images so we are sure that we are capturing the
        # correct dimensions
        IA = np.random.rand(50, 100)
        IB = np.random.rand(50, 100)
        mask = np.random.randint(0, 2, (50, 100))
        img = piv_image.PIVImage(IA, IB, mask)
        self.assertEqual(img.n_rows, 50)
        self.assertEqual(img.n_cols, 100)
        self.assertEqual(img.dim, (50, 100))

    def test_eq_method_evaluates_correctly(self):
        """
        the __eq__ method should compare if two image objects are the same.
        this should also be able to distinguish if two image objects are not
        equal
        """

        # create sets of images
        IA1 = np.random.rand(50, 50)
        IB1 = np.random.rand(50, 50)
        IA2 = np.random.rand(50, 50)
        IB2 = np.random.rand(50, 50)
        IA3 = np.random.rand(10, 10)
        IB3 = np.random.rand(10, 10)
        img1 = piv_image.PIVImage(IA1, IB1)
        img1_copy = piv_image.PIVImage(IA1, IB1)
        img2 = piv_image.PIVImage(IA2, IB2)
        img3 = piv_image.PIVImage(IA3, IB3)

        # check img1 and img1_copy return equal
        self.assertEqual(img1, img1_copy)
        self.assertNotEqual(img1, img2)
        self.assertNotEqual(img2, img3)

        # check that NotImplemented is raised if compared to another object
        self.assertEqual(img1.__eq__(4), NotImplemented)

    def test_get_region_with_negative_x_raises_error(self):
        """
        negative x doesn't make sense.
        This needs to be captured here instead of letting e.g. numpy catch it
        because we squash/truncate the x-rad and y-rad locations
        """

        IA = np.random.rand(50, 50)
        IB = np.random.rand(50, 50)
        img = piv_image.PIVImage(IA, IB)
        x, y, rad = -5, 10, 4

        with self.assertRaises(ValueError):
            img.get_region(x, y, rad)

    def test_get_region_with_negative_y_raises_error(self):
        """
        negative x doesn't make sense.
        This needs to be captured here instead of letting e.g. numpy catch it
        because we squash/truncate the x-rad and y-rad locations
        """

        IA = np.random.rand(50, 50)
        IB = np.random.rand(50, 50)
        img = piv_image.PIVImage(IA, IB)
        x, y, rad = 5, -10, 4

        with self.assertRaises(ValueError):
            img.get_region(x, y, rad)

    def test_get_region_with_x_out_of_bounds_raises_error(self):
        """
        negative x doesn't make sense.
        This needs to be captured here instead of letting e.g. numpy catch it
        because we squash/truncate the x-rad and y-rad locations
        """

        IA = np.random.rand(50, 50)
        IB = np.random.rand(50, 50)
        img = piv_image.PIVImage(IA, IB)
        x, y, rad = 55, 10, 4

        with self.assertRaises(ValueError):
            img.get_region(x, y, rad)

    def test_get_region_with_y_out_of_bounds_raises_error(self):
        """
        negative x doesn't make sense.
        This needs to be captured here instead of letting e.g. numpy catch it
        because we squash/truncate the x-rad and y-rad locations
        """

        IA = np.random.rand(50, 50)
        IB = np.random.rand(50, 50)
        img = piv_image.PIVImage(IA, IB)
        x, y, rad = 5, 100, 4

        with self.assertRaises(ValueError):
            img.get_region(x, y, rad)

    def test_get_region_returns_correct_region(self):
        """
        The region returned should be ctr-rad:ctr+rad in both x and y
        Can test this by creating an image with known pixel 'intensities'

        [[ 1,  2,  3,  4,  5,  6],
         [ 7,  8,  9, 10, 11, 12],
         [13, 14, 15, 16, 17, 18],
         [19, 20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29, 30],
         [31, 32, 33, 34, 35, 36]]

        """

        size_of_img = (6, 6)
        IA = np.arange(1, size_of_img[0] * size_of_img[1] + 1)
        IA = np.reshape(IA, size_of_img)
        IB = np.array(IA)
        mask = np.array(IA)
        img = piv_image.PIVImage(IA, IB, mask)
        ia, ib, mask = img.get_region(3, 3, 2)

        # manually determine the expected array
        exp_arr = np.array([[8, 9, 10, 11, 12],
                            [14, 15, 16, 17, 18],
                            [20, 21, 22, 23, 24],
                            [26, 27, 28, 29, 30],
                            [32, 33, 34, 35, 36]])
        print(ia)
        self.assertTrue(np.allclose(ia, exp_arr))
        self.assertTrue(np.allclose(ib, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

        # what happens if we truncate to the top left:
        ia, ib, mask = img.get_region(1, 1, 2)
        exp_arr = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 2, 3, 4],
                            [0, 7, 8, 9, 10],
                            [0, 13, 14, 15, 16],
                            [0, 19, 20, 21, 22]])
        print(ia)
        self.assertTrue(np.allclose(ia, exp_arr))
        self.assertTrue(np.allclose(ib, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

        # what happens if we truncate to the bottom right:
        ia, ib, mask = img.get_region(4, 4, 2)
        exp_arr = np.array([[15, 16, 17, 18, 0],
                            [21, 22, 23, 24, 0],
                            [27, 28, 29, 30, 0],
                            [33, 34, 35, 36, 0],
                            [0, 0, 0, 0, 0]])
        print(ia)
        self.assertTrue(np.allclose(ia, exp_arr))
        self.assertTrue(np.allclose(ib, exp_arr))
        self.assertTrue(np.allclose(mask, exp_arr))

    def test_get_region_returns_mask_if_not_defined(self):
        """
        The region returned should be ctr-rad:ctr+rad in both x and y
        Can test this by creating an image with known pixel 'intensities'

        [[ 1,  2,  3,  4,  5,  6],
         [ 7,  8,  9, 10, 11, 12],
         [13, 14, 15, 16, 17, 18],
         [19, 20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29, 30],
         [31, 32, 33, 34, 35, 36]]

        """

        size_of_img = (6, 6)
        IA = np.arange(1, size_of_img[0] * size_of_img[1] + 1)
        IA = np.reshape(IA, size_of_img)
        IB = np.array(IA)
        img = piv_image.PIVImage(IA, IB)
        ia, ib, mask = img.get_region(3, 3, 2)

        # manually determine the expected array
        exp_arr = np.array([[8, 9, 10, 11, 12],
                            [14, 15, 16, 17, 18],
                            [20, 21, 22, 23, 24],
                            [26, 27, 28, 29, 30],
                            [32, 33, 34, 35, 36]])
        print(ia)
        self.assertTrue(np.allclose(ia, exp_arr))
        self.assertTrue(np.allclose(ib, exp_arr))
        self.assertTrue(np.allclose(mask, np.ones((5, 5))))

    def test_load_mat_image_from_flowtype(self):
        """
        This test method tests that we can load .mat files in which are before
        version 7.3 as well as use the correct h5py library to load v7.3

        The vortex array and lamb oseen are both in format v7
        The Gaussian images are in v7.3
        """

        # we just want to check that it loads without issue
        flowtype = 22  # vortex array
        IA, IB, mask = piv_image.load_image_from_flow_type(flowtype, 1)

        flowtype = 24  # gaussian smoothed
        IA, IB, mask = piv_image.load_image_from_flow_type(flowtype, 1)

    def test_load_image_file(self):
        """
        Want to test that the method loads images if the requested file is a
        regular image file
        """

        # just checking it loads without issue
        flowtype = 1  # bfs
        IA, IB, mask = piv_image.load_image_from_flow_type(flowtype, 1)

    def test_load_image_loads_mask_file_if_no_file(self):
        """
        If the mask stored in the database is 'None' then set array with
        all ones
        vortex array (22) doesn't load a mask
        """

        flowtype = 22  # vortex array
        IA, IB, mask = piv_image.load_image_from_flow_type(flowtype, 1)
        self.assertTrue(np.allclose(mask, np.ones(np.shape(IA))))

    def test_load_image_builds_object(self):
        """
        Created as a result of Issue #4

        When loading images, we are getting a multi layered array for the mask
        This seems to be because of trying to load a colour image.

        Test loading a single image from every flow type and ensuring that
        it can create a PIVImage object without raising an error
        """

        # get flowtypes
        flowtypes = image_info.all_flow_types()
        for item in flowtypes:
            im_number = 1
            IA, IB, mask = piv_image.load_image_from_flow_type(item, im_number)
            # test that the object creates just fine
            piv_image.PIVImage(IA, IB, mask)

    def test_deformation_is_done_on_filtered_images(self):
        """
        just checks that the correct process is taken
        """

        IA = np.random.rand(100, 100)
        IB = np.random.rand(100, 100)
        img = piv_image.PIVImage(IA, IB)

        # deform image by 8 in x and 4 in y
        dp = dense_predictor.DensePredictor(
            np.ones((100, 100)) * 8, np.ones((100, 100)) * 4)
        img_def = img.deform_image(dp)

        # check process
        # filter images
        IA_filt = piv_image.quintic_spline_image_filter(IA)
        IB_filt = piv_image.quintic_spline_image_filter(IB)

        # get new pixel locations
        npx, npy = np.meshgrid(np.r_[1:101.], np.r_[1:101.])
        npx_a = npx - 4
        npy_a = npy - 2
        npx_b = npx + 4
        npy_b = npy + 2

        # deform images
        IA_def = sym_filt.bs5_int(IA_filt, 100, 100, npx_a, npy_a)
        IB_def = sym_filt.bs5_int(IB_filt, 100, 100, npx_b, npy_b)

        self.assertTrue(np.allclose(IA_def, img_def.IA))
        self.assertTrue(np.allclose(IB_def, img_def.IB))


if __name__ == "__main__":
    unittest.main(buffer=True)
