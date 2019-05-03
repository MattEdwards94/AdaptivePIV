import unittest
import image_info as im


class TestImageInfo(unittest.TestCase):
    def test_initialisation(self):
        # extract information for the first flow type
        # in this case we'll test for the vortex array
        im_info = im.ImageInfo(22)
        # check that the data is all loaded in the right order
        info = im.get_image_information(22)
        self.assertEqual(im_info.flowtype, int(info[0]))
        self.assertEqual(im_info.description, info[1])
        self.assertEqual(im_info.folder, info[2])
        self.assertEqual(im_info.folder, 'vArray2x2')
        self.assertEqual(im_info.filename, info[3])
        self.assertEqual(im_info.mask_fname, info[4])
        self.assertEqual(im_info.vel_field_fname, info[5])
        self.assertEqual(im_info.img_dim_text, info[6])
        self.assertEqual(im_info.n_images, int(info[7]))
        self.assertEqual(im_info.is_synthetic, info[8])
        self.assertEqual(im_info.is_time_resolved, info[9])

    def test_initialisation_sets_dimensions(self):
        # checks that n_rows, n_cols and img_dim are all set
        # we don't


if __name__ == "__main__":
    unittest.main()
