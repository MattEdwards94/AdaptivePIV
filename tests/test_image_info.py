import unittest
import image_info
import piv_image


class TestImageInfo(unittest.TestCase):
    def test_initialisation(self):
        # extract information for the first flow type
        # in this case we'll test for the vortex array
        im_info = image_info.ImageInfo(22)
        # check that the data is all loaded in the right order
        info = image_info.get_image_information(22)
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
        """
        Checks that n_rows, n_cols and img_dim are all set
        we know the dimensions of the backwards facing step are:
        n_rows = 640
        n_cols = 1280
        it is also useful that the rows and cols are different to ensure we
        test that they don't get mixed up
        """
        im_info = image_info.ImageInfo(1)
        self.assertEqual(im_info.n_rows, 640)
        self.assertEqual(im_info.n_cols, 1280)
        self.assertEqual(im_info.img_dim, [640, 1280])

    def test_has_mask_is_set(self):
        """ensures that the setting 'has_mask' is set
        """
        im_info = image_info.ImageInfo(1)
        self.assertTrue(im_info.has_mask)
        im_info = image_info.ImageInfo(22)
        self.assertFalse(im_info.has_mask)

    def test_eq_evaluates_correctly(self):
        """
        Test that __eq__ evaluates correctly.
        Test that it returns true for equivalent objects,
        Test that it returns false for completely different objects
        Test that it returns false for slightly different objects
        """
        im = image_info.ImageInfo(1)
        im2 = image_info.ImageInfo(1)
        self.assertEqual(im, im2)  # this will invoke the __eq__ operator
        im3 = image_info.ImageInfo(2)
        self.assertNotEqual(im, im3)
        im2.n_images = 0
        self.assertNotEqual(im, im2)

    def test_eval_repr_creates_an_obj(self):
        """
        Check that eval(repr) at least runs and creates an object
        """
        im_info = image_info.ImageInfo(1)
        # create object using eval repr
        im_info_duplicate = eval(repr(im_info))
        self.assertEqual(im_info, im_info_duplicate)

    def test_print_all_details_runs(self):
        """Simply tests that printing all details runs without error
        """
        image_info.print_all_details()

    def test_list_available_flowtypes_runs(self):
        """Simply tests that listing available flow types runs without error
        """
        image_info.list_available_flowtypes()

    def test_print_table_header_runs(self):
        """Simply tests that printing the table header runs without error
        """
        image_info.print_table_header()

    def test_get_image_information_runs(self):
        """Simply tests that getting image information runs without error
        """
        row = image_info.get_image_information(5)

    def test_get_image_information_raises_value_error(self):
        """
        Tests that get_image_information raises a value error if the flow
        type is not found within the database
        """
        with self.assertRaises(ValueError):
            image_info.get_image_information(5000000)

    def test_formatted_filenames_raises_warning_if_im_number_too_big(self):
        """The database (csv file) stores the number of images in the ensemble
        if the required number is larger than this then a warning should be
        thrown
        It should NOT throw an error at this point, since it may be the
        database which is outdated - in this case attempting to use the returned
        filename would still work. If indeed this image does not exist then
        the code will error upon trying to access a non-existant file.
        """

        # n_images for bfs is 320
        im = image_info.ImageInfo(1)
        with self.assertWarns(UserWarning):
            im.formatted_filenames(321)

    def test_formatted_filenames_returns_none_for_no_mask(self):
        """If the has_mask property is false then the method should return None
        if there is no mask. This will allow for easy comparison later
        """

        # lamb oseen has no mask
        im = image_info.ImageInfo(23)
        fnames = im.formatted_filenames(1)
        self.assertIsNone(fnames[2], None)

    def test_all_flow_types(self):
        """
        Tests that all the flow types are valid
        """

        flow_types = image_info.all_flow_types()
        for item in flow_types:
            print(item)
            _, _, _ = piv_image.load_image_from_flow_type(item, im_number=1)


if __name__ == "__main__":
    unittest.main(buffer=True)
