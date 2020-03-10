import pytest
import PIV.image_info as image_info
import PIV.piv_image as piv_image


def test_initialisation():
    # extract information for the first flow type
    # in this case we'll test for the vortex array
    im_info = image_info.ImageInfo(22)
    # check that the data is all loaded in the right order
    info = image_info.get_image_information(22)
    assert im_info.flowtype == int(info[0])
    assert im_info.description == info[1]
    assert im_info.folder == info[2]
    assert im_info.folder == 'vArray2x2'
    assert im_info.filename == info[3]
    assert im_info.mask_fname == info[4]
    assert im_info._vel_field_fname == info[5]
    assert im_info.img_dim_text == info[6]
    assert im_info.n_images == int(info[7])
    assert im_info.is_synthetic == info[8]
    assert im_info.is_time_resolved == info[9]


def test_initialisation_sets_dimensions():
    """
    Checks that n_rows, n_cols and img_dim are all set
    we know the dimensions of the backwards facing step are:
    n_rows = 640
    n_cols = 1280
    it is also useful that the rows and cols are different to ensure we
    test that they don't get mixed up
    """
    im_info = image_info.ImageInfo(1)
    assert im_info.n_rows == 640
    assert im_info.n_cols == 1280
    assert im_info.img_dim == [640, 1280]


def test_has_mask_is_set():
    """ensures that the setting 'has_mask' is set
    """
    im_info = image_info.ImageInfo(1)
    assert im_info.has_mask
    im_info = image_info.ImageInfo(22)
    assert not im_info.has_mask


def test_eq_evaluates_correctly():
    """
    Test that __eq__ evaluates correctly.
    Test that it returns true for equivalent objects,
    Test that it returns false for completely different objects
    Test that it returns false for slightly different objects
    """
    im = image_info.ImageInfo(1)
    im2 = image_info.ImageInfo(1)
    assert im == im2  # this will invoke the __eq__ operato
    im3 = image_info.ImageInfo(2)
    assert not im == im3
    im2.n_images = 0
    assert not im == im2


def test_eval_repr_creates_an_obj():
    """
    Check that eval(repr) at least runs and creates an object
    """
    im_info = image_info.ImageInfo(1)
    # create object using eval repr
    im_info_duplicate = eval(repr(im_info))
    assert im_info == im_info_duplicate


def test_print_all_details_runs():
    """Simply tests that printing all details runs without error
    """
    image_info.print_all_details()


def test_list_available_flowtypes_runs():
    """Simply tests that listing available flow types runs without error
    """
    image_info.list_available_flowtypes()


def test_print_table_header_runs():
    """Simply tests that printing the table header runs without error
    """
    image_info.print_table_header()


def test_get_image_information_runs():
    """Simply tests that getting image information runs without error
    """
    image_info.get_image_information(5)


def test_get_image_information_raises_value_error():
    """
    Tests that get_image_information raises a value error if the flow
    type is not found within the database
    """
    with pytest.raises(ValueError):
        image_info.get_image_information(5000000)


def test_formatted_filenames_raises_warning_if_im_number_too_big():
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
    with pytest.warns(UserWarning):
        im.formatted_filenames(321)


def test_formatted_filenames_returns_none_for_no_mask():
    """If the has_mask property is false then the method should return None
    if there is no mask. This will allow for easy comparison later
    """

    # lamb oseen has no mask
    im = image_info.ImageInfo(23)
    fnames = im.formatted_filenames(1)
    assert fnames[2] is None
