import unittest
import piv_image
import image_info
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
        img = piv_image.PIVImage(self.IA, self.IB, self.mask)

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

    def test_repr_reproduces_object(self):
        """Check that the __repr__ method recreates an equivalent object
        """

        # create dummy object
        img = piv_image.PIVImage(self.IA, self.IB, self.mask)

        # create object using __repr__
        img_repr = eval(repr(img))
        self.assertEqual(img, img_repr)


if __name__ == "__main__":
    unittest.main(buffer=False)
