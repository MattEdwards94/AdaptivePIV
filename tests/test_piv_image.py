import pytest
import PIV.piv_image as piv_image
import numpy as np
import PIV.dense_predictor as dense_predictor
import sym_filt
import PIV.image_info as image_info
import scipy.io as sio
import h5py


@pytest.fixture
def mock_IA(height=100, width=100):
    """Creates a random piv Image
    """
    return np.random.rand(height, width)


@pytest.fixture
def mock_IB(height=100, width=100):
    """Creates a random piv Image
    """
    return np.random.rand(height, width)


@pytest.fixture
def mock_mask(height=100, width=100):
    """Creates a mock image mask
    """
    return np.random.randint(0, 2, size=(height, width))


def test_initialisation_checks_image_size(mock_IA,
                                          mock_IB,
                                          mock_mask):
    """Checks that the inputs IA, IB must be of the same size otherwise
    a value error is thrown
    If mask is passed then this should also be the same size
    """

    assert np.alltrue(mock_IA != mock_IB)
    # mock_IA, IB, mask are all the same size so this should work fine
    piv_image.PIVImage(mock_IA, mock_IB, mock_mask)

    # now change the size of IB and check an error is thrown
    size = np.shape(mock_IA)
    bad_size_IB = np.random.rand(size[1] + 10, size[0] - 10)
    with pytest.raises(ValueError):
        piv_image.PIVImage(mock_IA, bad_size_IB, mock_mask)

    # check that the size of mask if given is checked
    bad_mask = np.random.randint(0, 2, size=(size[1] + 10, size[0] - 10))
    with pytest.raises(ValueError):
        piv_image.PIVImage(mock_IA, mock_IB, bad_mask)


def test_mask_set_to_ones_if_not_passed(mock_IA, mock_IB):
    """The default value for mask is None. In this case, an empty mask of
    ones will be created with the same size as IA and IB
    """

    # if no mask is set, we should obtain an array of ones of size img_size
    img = piv_image.PIVImage(mock_IA, mock_IB)
    expected = np.ones(np.shape(mock_IA))
    assert np.alltrue(img.mask == expected)


def test_initialisation_assigns_intensities_correctly(mock_IA, mock_IB):
    """
    Check that IA is assigned to mock_IA and that
    IB is assigned to mock_IB
    """
    img = piv_image.PIVImage(mock_IA, mock_IB)
    assert np.allclose(mock_IA, img.IA)
    assert np.allclose(mock_IB, img.IB)
    assert np.issubdtype(img.IA.dtype, np.float64)
    assert np.issubdtype(img.IB.dtype, np.float64)


def test_initialisation_saves_mask_status(mock_IA, mock_IB, mock_mask):
    """
    If there has been a mask passed we want to save this information as
    an easily checkable bool
    """

    img = piv_image.PIVImage(mock_IA, mock_IB)
    assert not img.has_mask
    img2 = piv_image.PIVImage(mock_IA, mock_IB, mock_mask)
    assert img2.has_mask


def test_image_dimensions_are_captured():
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
    assert img.n_rows == 50
    assert img.n_cols == 100
    assert img.dim == (50, 100)


def test_eq_method_evaluates_correctly():
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
    assert img1 == img1_copy
    assert not img1 == img2
    assert not img2 == img3

    # check that NotImplemented is raised if compared to another object
    assert img1.__eq__(4) == NotImplemented


def test_get_region_with_negative_x_raises_error(mock_IA, mock_IB):
    """
    negative x doesn't make sense.
    This needs to be captured here instead of letting e.g. numpy catch it
    because we squash/truncate the x-rad and y-rad locations
    """

    img = piv_image.PIVImage(mock_IA, mock_IB)
    x, y, rad = -5, 10, 4

    with pytest.raises(ValueError):
        img.get_region(x, y, rad)


def test_get_region_with_negative_y_raises_error(mock_IA, mock_IB):
    """
    negative x doesn't make sense.
    This needs to be captured here instead of letting e.g. numpy catch it
    because we squash/truncate the x-rad and y-rad locations
    """

    img = piv_image.PIVImage(mock_IA, mock_IB)
    x, y, rad = 5, -10, 4

    with pytest.raises(ValueError):
        img.get_region(x, y, rad)


def test_get_region_with_x_out_of_bounds_raises_error(mock_IA, mock_IB):
    """
    negative x doesn't make sense.
    This needs to be captured here instead of letting e.g. numpy catch it
    because we squash/truncate the x-rad and y-rad locations
    """

    img = piv_image.PIVImage(mock_IA, mock_IB)
    x, y, rad = 100, 10, 4

    with pytest.raises(ValueError):
        img.get_region(x, y, rad)


def test_get_region_with_y_out_of_bounds_raises_error(mock_IA, mock_IB):
    """
    negative x doesn't make sense.
    This needs to be captured here instead of letting e.g. numpy catch it
    because we squash/truncate the x-rad and y-rad locations
    """

    img = piv_image.PIVImage(mock_IA, mock_IB)
    x, y, rad = 5, 100, 4

    with pytest.raises(ValueError):
        img.get_region(x, y, rad)


def test_get_region_returns_correct_region():
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
    assert np.allclose(ia, exp_arr)
    assert np.allclose(ib, exp_arr)
    assert np.allclose(mask, exp_arr)

    # what happens if we truncate to the top left:
    ia, ib, mask = img.get_region(1, 1, 2)
    exp_arr = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4],
                        [0, 7, 8, 9, 10],
                        [0, 13, 14, 15, 16],
                        [0, 19, 20, 21, 22]])
    print(ia)
    assert np.allclose(ia, exp_arr)
    assert np.allclose(ib, exp_arr)
    assert np.allclose(mask, exp_arr)

    # what happens if we truncate to the bottom right:
    ia, ib, mask = img.get_region(4, 4, 2)
    exp_arr = np.array([[15, 16, 17, 18, 0],
                        [21, 22, 23, 24, 0],
                        [27, 28, 29, 30, 0],
                        [33, 34, 35, 36, 0],
                        [0, 0, 0, 0, 0]])
    print(ia)
    assert np.allclose(ia, exp_arr)
    assert np.allclose(ib, exp_arr)
    assert np.allclose(mask, exp_arr)

    # check the x and y are correct:
    ia, ib, mask = img.get_region(3, 4, 2)
    exp_arr = np.array([[14, 15, 16, 17, 18],
                        [20, 21, 22, 23, 24],
                        [26, 27, 28, 29, 30],
                        [32, 33, 34, 35, 36],
                        [0, 0, 0, 0, 0]])
    print(ia)
    assert np.allclose(ia, exp_arr)
    assert np.allclose(ib, exp_arr)
    assert np.allclose(mask, exp_arr)


def test_get_region_returns_mask_if_not_defined():
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
    assert np.allclose(ia, exp_arr)
    assert np.allclose(ib, exp_arr)
    assert np.allclose(mask, np.ones((5, 5)))


def test_load_mat_image_from_flowtype():
    """
    This test method tests that we can load .mat files in which are before
    version 7.3 as well as use the correct h5py library to load v7.3

    The vortex array and lamb oseen are both in format v7
    The Gaussian images are in v7.3
    """

    # overwrite utilities.root_path() such that we look in the data folder
    # instead of the main folder
    import PIV.utilities
    def replace_func():
        return "./PIV/data/"
    old = PIV.utilities.root_path
    PIV.utilities.root_path = replace_func

    # we just want to check that it loads without issue
    flowtype = 22  # vortex array v7
    IA_act, IB_act, _ = piv_image.load_images(flowtype, 1)
    # now check that it is transposed correctly by loading it manually and
    # checking the result
    im_info = image_info.ImageInfo(flowtype)
    filenames = im_info.formatted_filenames(1)
    img = sio.loadmat(filenames[0])
    IA_exp = np.array(img['IA'])
    img = sio.loadmat(filenames[1])
    IB_exp = np.array(img['IB'])
    assert np.allclose(IA_exp, IA_act)
    assert np.allclose(IB_exp, IB_act)

    flowtype = 24  # gaussian smoothed v7.3
    IA_act, IB_act, mask = piv_image.load_images(flowtype, 1)
    im_info = image_info.ImageInfo(flowtype)
    filenames = im_info.formatted_filenames(1)
    img = h5py.File(filenames[0])
    IA_exp = np.transpose(np.array(img['IA']))
    img = h5py.File(filenames[1])
    IB_exp = np.transpose(np.array(img['IB']))
    assert np.allclose(IA_exp, IA_act)
    assert np.allclose(IB_exp, IB_act)

    PIV.utilities.root_path = old

def test_load_image_file():
    """
    Want to test that the method loads images if the requested file is a
    regular image file
    """

    # overwrite utilities.root_path() such that we look in the data folder
    # instead of the main folder
    import PIV.utilities
    def replace_func():
        return "./PIV/data/"
    old = PIV.utilities.root_path
    PIV.utilities.root_path = replace_func
    
    # just checking it loads without issue
    flowtype = 1  # bfs
    _, _, _ = piv_image.load_images(flowtype, 1)
    PIV.utilities.root_path = old


def test_load_image_loads_mask_file_if_no_file():
    """
    If the mask stored in the database is 'None' then set array with
    all ones
    vortex array (22) doesn't load a mask
    """

    # overwrite utilities.root_path() such that we look in the data folder
    # instead of the main folder
    import PIV.utilities
    def replace_func():
        return "./PIV/data/"
    old = PIV.utilities.root_path
    PIV.utilities.root_path = replace_func

    flowtype = 22  # vortex array
    IA, IB, mask = piv_image.load_images(flowtype, 1)
    assert np.allclose(mask, np.ones(np.shape(IA)))

    PIV.utilities.root_path = old

def test_load_image_builds_object():
    """
    Created as a result of Issue #4

    When loading images, we are getting a multi layered array for the mask
    This seems to be because of trying to load a colour image.

    Test loading a single image from several flow types and ensuring that
    it can create a PIVImage object without raising an error
    """

    # overwrite utilities.root_path() such that we look in the data folder
    # instead of the main folder
    import PIV.utilities
    def replace_func():
        return "./PIV/data/"
    old = PIV.utilities.root_path
    PIV.utilities.root_path = replace_func

    # get flowtypes
    flowtypes = [1, 2, 22, 23]
    for item in flowtypes:
        im_number = 1
        IA, IB, mask = piv_image.load_images(item, im_number)
        # test that the object creates just fine
        piv_image.PIVImage(IA, IB, mask)

    PIV.utilities.root_path = old


def test_deformation_is_done_on_filtered_images(mock_IA,
                                                mock_IB):
    """
    just checks that the correct process is taken
    """
    print(np.shape(mock_IA))
    img = piv_image.PIVImage(mock_IA, mock_IB)

    # deform image by 8 in x and 4 in y
    dp = dense_predictor.DensePredictor(
        np.ones((100, 100)) * 8, np.ones((100, 100)) * 4)
    img_def = img.deform_image(dp)

    # check process
    # filter images
    IA_filt = piv_image.quintic_spline_image_filter(mock_IA)
    IB_filt = piv_image.quintic_spline_image_filter(mock_IB)

    # get new pixel locations
    npx, npy = np.meshgrid(np.r_[1:101.], np.r_[1:101.])
    npx_a = npx - 4
    npy_a = npy - 2
    npx_b = npx + 4
    npy_b = npy + 2

    # deform images
    IA_def = sym_filt.bs5_int(IA_filt, 100, 100, npx_a, npy_a)
    IB_def = sym_filt.bs5_int(IB_filt, 100, 100, npx_b, npy_b)

    assert np.allclose(IA_def, img_def.IA)
    assert np.allclose(IB_def, img_def.IB)


def test_load_PIVImage():
    """
    Tests that the image loaded is equivalent to loading the image in the
    manual way
    """

    # overwrite utilities.root_path() such that we look in the data folder
    # instead of the main folder
    import PIV.utilities
    def replace_func():
        return "./PIV/data/"
    old = PIV.utilities.root_path
    PIV.utilities.root_path = replace_func

    # 'manual' way
    flowtype, im_number = 1, 20
    IA, IB, mask = piv_image.load_images(flowtype, im_number)
    exp = piv_image.PIVImage(IA, IB, mask)

    # test method
    act = piv_image.load_PIVImage(flowtype, im_number)
    assert exp, act
    PIV.utilities.root_path = old


def test_quintic_spline_image_filt_all_ones():
    """All ones input should give all ones output
    """

    a = np.ones((50, 50))
    b = piv_image.quintic_spline_image_filter(a)

    assert np.allclose(a, b)


def test_get_binary_img_part_locations():
    """Check the output for an expected input
    """

    img_dim, npart = (75, 50), 25
    xp = np.random.uniform(0, img_dim[1]-1, npart)
    yp = np.random.uniform(0, img_dim[0]-1, npart)

    exp = np.zeros(img_dim)
    x_ind, y_ind = np.round(xp).astype(int), np.round(yp).astype(int)
    exp[y_ind, x_ind] = 1

    act = piv_image.get_binary_image_particle_locations(xp, yp, img_dim)

    assert np.allclose(act, exp)


def test_particle_detection_perf_n_particles():
    """
    Tests that the true number of particles is correctly identified
    """

    img_dim = (50, 50)

    for n_particles_in in range(1, 101, 10):
        img_in = np.zeros(img_dim)
        x = np.random.randint(0, img_dim[1], n_particles_in)
        y = np.random.randint(0, img_dim[0], n_particles_in)

        # it's possible that the coordinates are duplicated, and we can only
        # handle one particle per pixel
        xy = np.unique(np.vstack((x, y)).T, axis=0)

        img_in[xy[:, 1], xy[:, 0]] = 1
        (n_particles, _, 
        _, _) = piv_image.particle_detection_perf(img_in,
                                                           np.zeros_like(img_in))
        assert n_particles == np.shape(xy)[0]


def test_particle_detection_perf_n_valid_is_correct():
    """
    Tests that the number of correctly identified particles is correct. 

    For this we will pass in a binary image which is similar to the img_in with
    a known number of correctly detected particle locations
    """

    img_dim = (50, 50)

    for n_particles_in in range(1, 101, 10):
        img_true, img_test = np.zeros(img_dim), np.zeros(img_dim)

        x = np.random.randint(0, img_dim[1], n_particles_in)
        y = np.random.randint(0, img_dim[0], n_particles_in)

        # it's possible that the coordinates are duplicated, and we can only
        # handle one particle per pixel
        xy = np.unique(np.vstack((x, y)).T, axis=0)

        img_true[xy[:, 1], xy[:, 0]] = 1

        for i in range(0, np.shape(xy)[0]):
            img_test[xy[:i, 1], xy[:i, 0]] = 1
            (_, 
            n_detect_valid, 
            _, _) = piv_image.particle_detection_perf(img_true,
                                                                    img_test)
            assert n_detect_valid == i


def test_particle_detection_perf_n_invalid_correct():
    """
    Tests that the number of number of 'detected' particle images which 
    don't line up with a true particle image is correct
    """

    img_dim = (50, 50)

    img_true, img_test = np.zeros(img_dim), np.zeros(img_dim)

    # manually create a list of coordinates for the true particle locations
    xy_true = np.array([[1, 7],
               [2, 5],
               [3, 9],
               [3, 4],
               [4, 6],
               [5, 8],
               [6, 6]])
    xy_add = xy_true + 1

    img_true[xy_true[:, 1], xy_true[:, 0]] = 1
    img_test[xy_add[:, 1], xy_add[:, 0]] = 1
    
    (_, _, n_detect_invalid, _) = piv_image.particle_detection_perf(img_true,
                                                                img_test)

    assert n_detect_invalid == 7


def test_particle_detection_perf_n_undetected():
    """
    Tests the number of particles which were not detected by the 
    detection routine
    """

    img_dim = (50, 50)

    img_true, img_test = np.zeros(img_dim), np.zeros(img_dim)

    # manually create a list of coordinates for the true particle locations
    xy_true = np.array([[1, 7],
               [2, 5],
               [3, 9],
               [3, 4],
               [4, 6],
               [5, 8],
               [6, 6]])
    xy_add = xy_true[:4, :]

    img_true[xy_true[:, 1], xy_true[:, 0]] = 1
    img_test[xy_add[:, 1], xy_add[:, 0]] = 1
    
    (_, _, _, n_undetected) = piv_image.particle_detection_perf(img_true,
                                                                img_test)

    # minus 4 since we haven't added have added 4 particles to the image
    assert n_undetected == np.shape(xy_true)[0] - 4


def test_detect_particles_max_filter():
    """
    Tests the particle detection for a simple cases
    """

    # create mock image
    img_dim = (50, 50)
    xp, yp = [4, 15, 25, 35, 42], [5, 12, 4, 45, 15]
    d_tau, Ip = [3]*5, [0.9]*5
    img = piv_image.render_synthetic_PIV_image(img_dim, xp, yp, 
                                               d_tau, Ip, 
                                               noise_mean=0, noise_std=0)

    exp = np.zeros(img_dim)
    exp[yp, xp] = 1
    
    act = piv_image.detect_particles_max_filter(img)


    assert np.allclose(exp, act)


def test_detect_particles_max_filter_considers_mask():
    """If a mask is given the the particle detection routine should return
    0's in this region
    """

    # create mock image
    img_dim = (150, 150)
    (xp1, yp1, 
    dtau, Ip) = piv_image.gen_uniform_part_locations(img_dim, 0.05,
                                                     int_mean=0.8, int_std=0.1)
    img = piv_image.render_synthetic_PIV_image(img_dim, xp1, yp1,
                                               dtau, Ip,
                                               noise_mean=0.4, noise_std=0.05)
    
    # define the mask on the left half of the image
    mask_lim = int(img_dim[1] / 2)
    mask = np.ones(img_dim)
    mask[:, 0:mask_lim] = 0

    # this is to make sure that the threshold will be different if the mask
    # is not considered
    img[:, 0:mask_lim] = 0

    # we know that we have particle images over the whole image
    # we want to check that only one half of the image is being considered 
    # in terms of the identified particles, and the threshold used.
    # apply the maximum filter
    import scipy.ndimage.filters as im_filter
    import skimage.filters

    mf = im_filter.maximum_filter(img, size=3)

    # obtain the threshold
    thr = skimage.filters.threshold_otsu(img[mask==1])
    
    # get the particle locations
    exp = (img == mf) & (img >= thr) & (mask == 1)

    act = piv_image.detect_particles_max_filter(img, mask)

    assert np.allclose(act, exp)

def test_detect_particles_with_mask_shape_wrong():
    """
    Test that a mask with the wrong size is caught
    """

    with pytest.raises(ValueError):
        piv_image.detect_particles(np.random.rand(100, 100), 
                                   mask=np.random.randint(0, 2, (50, 50)))


def test_detect_particles_method_not_implemented():
    """
    If a bad method is given, raise not implemented error
    """ 
    
    with pytest.raises(NotImplementedError):
        piv_image.detect_particles(np.random.rand(100, 100), 
                                   method='not_implemented')

def test_detect_particles_output_shape():
    """
    Test that the output of detect particles maintains the same shape as the
    input image
    """    

    img = np.random.rand(100, 100)
    out = piv_image.detect_particles(img)
    assert np.shape(img) == np.shape(out)
    


