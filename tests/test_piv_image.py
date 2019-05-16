import unittest
import piv_image
import numpy as np


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

    def test_mask_set_to_zeros_if_not_passed(self):
        """The default value for mask is None. In this case, an empty mask of
        zeros will be created with the same size as IA and IB
        """

        # if no mask is set, we should obtain an array of zeros of size img_size
        img = piv_image.PIVImage(self.IA, self.IB)
        expected = np.zeros(np.shape(self.IA))
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
        self.assertEqual(img.img_dim, [50, 100])

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
        self.assertTrue(np.allclose(mask, np.zeros((5, 5))))

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


if __name__ == "__main__":
    unittest.main(buffer=True)
