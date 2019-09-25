import image_info
import scipy.io as sio
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import sym_filt
import dense_predictor
import matplotlib.pyplot as plt


class PIVImage:
    """
    Stores the information for a PIV image pair and provides functionality to
    select sub regions/perform pre-processing/image deformation etc.

    Attributes:
        has_mask (bool): Defines whether the image has a non-clear mask
        IA (np array): The first image in the pair
        IB (np array): The second image in the pair.
                   Must have the same dimensions as IA
        mask (np array): Image mask indicating regions not to be considered
                     in the analysis
        n_rows (int): Number of rows in the image
        n_cols (int): Number of columns in the image
        dim (tuple): (n_rows, n_cols)
        is_filtered (bool): Indicates whether the image has already been
                            filtered and my now be interpolated
        IA_filt (ndarray): Filtered image to be used for re-interpolation
        IB_filt (ndarray): Filtered image to be used for re-interpolation

    Examples:
        >>>IA = np.random.rand(100, 100)
        >>>IB = np.random.rand(100, 100)
        >>>mask = np.random.randint(0, 2, (100, 100))
        >>>obj = PIVImage(IA, IB, mask)

    Deleted Attributes:
        n_rows(int): The number of rows in the image
        n_cols(int): The number of columns in the image
        dim(tuple): The dimensions of the image [n_rows, n_cols]
    """

    def __init__(self, IA, IB, mask=None):
        """
        Stores the two images IA and IB along with the associated mask.

        Args:
            IA (array like): First image in the pair
            IB (array like): The second image in the pair.
                       Must have the same dimensions as IA
            mask (array like, optional): Image mask indicating regions not to
                                         be considered in the analysis

        Raises:
            ValueError: If the shapes of IA, IB, and mask are not the same
        """

        if np.shape(IA) != np.shape(IB):
            raise ValueError("shape of IA must match the shape of IB")

        if mask is not None:
            if np.shape(IA) != np.shape(mask):
                raise ValueError("The shape of the mask must match IA and IB")
            self.has_mask = True
        else:
            mask = np.ones(np.shape(IA))
            self.has_mask = False

        self.IA = np.array(IA, dtype=np.float64)
        self.IB = np.array(IB, dtype=np.float64)
        self.mask = np.array(mask, dtype=np.float64)
        self.n_rows = np.shape(IA)[0]
        self.n_cols = np.shape(IA)[1]
        self.dim = (self.n_rows, self.n_cols)
        self.is_filtered = False

    def __eq__(self, other):
        """
        Add method to PIVImage to check for equality

        PIVImage objects are considered equal if IA, IB, and mask all match

        Only compares equality to other PIVImage objects.
        Will return NotImplemented if the classes are not of the same type
            e.g.
            obj1 = PIVImage(IA, IB)
            obj2 = MyClass(a, b)
            obj1 == obj2
                returns NotImplemented

        Args:
            other (PIVImage): The object to be compared against

        Returns:
            Bool: True or False depending on object equality

        Examples:
            >>> obj1 = PIVImage(IA, IB)
            >>> obj2 = PIVImage(IA, IB)
            >>> obj1 == obj2
            ... returns True

            >>> obj3 = PIVImage(IA2, IB2)
            >>> obj3 == obj1
            ... returns False

            >>> obj4 = MyOtherClass(a, b, c)
            >>> obj4 == obj1
            ... returns NotImplemented
        """
        # print(other)
        # print(isinstance(other, PIVImage))
        if not isinstance(other, PIVImage):
            return NotImplemented

        if not np.alltrue(self.IA == other.IA):
            return False

        if not np.alltrue(self.IB == other.IB):
            return False

        if not np.alltrue(self.mask == other.mask):
            return False

        return True

    def get_region(self, x, y, rad):
        """
        Retrieves the pixel intensities for the region requested.
        If the region requested extends beyond the image dimensions, then
        such pixels will be set to 0 intensity.

        Matrix access is base 0 and is ROW major
        -------------
        | 0| 1| 2| 3|
        -------------
        | 4| 5| 6| 7|
        -------------
        | 8| 9|10|11|
        -------------

        Args:
            x (int): x coord of the centre of the region to be extracted
            y (int): y coord of the centre of the region to be extracted
            rad (int): the number of pixels to extend in each directions

        Returns:
            ia (ndarray): Intensity values from the first image in the region
                           [x-rad:x+rad, y-rad:y+rad]
                           np.shape(ia) = ((2*rad+1), (2*rad+1))
            ib (ndarray): Intensity values from the second image in the region
                           [x-rad:x+rad, y-rad:y+rad]
                           np.shape(ib) = ((2*rad+1), (2*rad+1))
            mask (ndarray): Mask flag values in the region
                           [x-rad:x+rad, y-rad:y+rad]
                           np.shape(mask) = ((2*rad+1), (2*rad+1))

        Examples:
        >>> ia, ib, mask = PIVImage.get_region(20, 15, 9)
        >>> np.shape(ia)
        ... (18, 18)

        """

        """
        extract what region we can within the image.
        to do this we need to know where we are with respect to the limits
        of the image
        """

        # check whether x and y are within the image region
        if x < 0:
            raise ValueError("x can't be negative")
        if y < 0:
            raise ValueError("y can't be negative")
        if x > self.n_cols - 1:  # -1 for 0 index
            raise ValueError("x out of bounds")
        if y > self.n_rows - 1:
            raise ValueError("y out of bounds")

        # if x - rad is < 0, set to 0, if > n_cols, set to n_cols - 1
        # the minus 1 is because of 0 indexing
        left = max(min(x - rad, self.n_cols - 1), 0)
        right = max(min(x + rad, self.n_cols - 1), 0)
        bottom = max(min(y - rad, self.n_rows - 1), 0)
        top = max(min(y + rad, self.n_rows - 1), 0)

        # pad with 0's where the requested region is outside of the image
        # domain. If the requested sub region is entirely within the image
        # then there wont be padding
        lStart = max(left - (x - rad), 0)
        rEnd = lStart + (right - left)
        bStart = max(bottom - (y - rad), 0)
        tEnd = bStart + top - bottom

        ia, ib = np.zeros((2 * rad + 1, 2 * rad + 1)
                          ), np.zeros((2 * rad + 1, 2 * rad + 1))

        # extract this region out of the images/mask
        # note the +1 is because left:right is not inclusive of right
        ia[bStart:tEnd + 1, lStart:rEnd +
            1] = self.IA[bottom:top + 1, left:right + 1]
        ib[bStart:tEnd + 1, lStart:rEnd +
            1] = self.IB[bottom:top + 1, left:right + 1]
        if self.has_mask:
            mask = np.zeros((2 * rad + 1, 2 * rad + 1))
            mask[bStart:tEnd + 1, lStart:rEnd +
                 1] = self.mask[bottom:top + 1, left:right + 1]
        else:
            mask = np.ones((2 * rad + 1, 2 * rad + 1))

        return ia, ib, mask

    def deform_image(self, dp):
        """
        Deforms the images according to the displacment field dp

        Performs a central differencing scheme such that:
            IA --> -0.5*dp
            IB --> +0.5*dp

        Args:
            dp (Densepredictor): Densepredictor object

        Returns:
            PIVImage: The image deformed according to dp

        Raises:
            ValueError: If the dimensions of the Densepredictor and the image
                        don't match
        """

        # check that the image and densepredictor are the same size
        if not np.all(self.dim == dp.dim):
            raise ValueError("dimensions of image and dp must match")

        # check whether the images have already been filtered
        if not self.is_filtered:
            self.IA_filt = quintic_spline_image_filter(self.IA)
            self.IB_filt = quintic_spline_image_filter(self.IB)
            self.is_filtered = True

        # calculate pixel locations
        xx, yy = np.meshgrid(np.r_[1:self.n_cols + 1],
                             np.r_[1:self.n_rows + 1])

        IA_new = sym_filt.bs5_int(
            self.IA_filt, self.n_rows, self.n_cols,
            xx - 0.5 * dp.u, yy - 0.5 * dp.v)
        IB_new = sym_filt.bs5_int(
            self.IB_filt, self.n_rows, self.n_cols,
            xx + 0.5 * dp.u, yy + 0.5 * dp.v)

        return PIVImage(IA_new, IB_new, self.mask)

    def plot_images(self):
        """Summary
        """
        plt.figure(1)
        plt.imshow(self.IA)
        plt.title("IA")
        plt.figure(2)
        plt.imshow(self.IB)
        plt.title("IB")
        plt.show()


def load_images(flowtype, im_number):
    """
    Loads the PIV image pair associated with the specified flowtype and the
    image number
    Also loads the mask associated, if there is one.
    If not mask is stored for the specified flowtype, then mask is returned
    as zeros(shape(IA))

    Args:
        flowtype (Int): The flowtype of the desired piv images.
                        For more information call
                        image_info.list_available_flowtypes()
        im_number (Int): The number in the ensemble to load into memory.
                         If im_number is greater than the known number of images
                         for the specified flowtype, a warning will be raised
                         The method will still try to open the requested file
                         If the file does not exist then an error will
                         be raised

    Returns:
        IA (ndarray): Image intensities for the first in the image pair
        IB (ndarray): Image intensities for the second in the image pair
        mask (ndarray): mask values with 0 for no mask and 1 for mask

    Examples:
        >>> import image_info
        >>> image_info.list_available_flowtypes() # to obtain options
        >>> IA, IB, mask = piv_image.load_image_from_flow_type(1, 20)
    """
    # first load the image information
    im_info = image_info.ImageInfo(flowtype)

    # get the formatted filename with the correct image number inserted
    filenames = im_info.formatted_filenames(im_number)
    print(filenames)

    # try to load image A
    if filenames[0][-4:] == ".mat":
        try:
            # mat files <7.3
            img = sio.loadmat(filenames[0])
            IA = np.array(img['IA'])
            pass
        except NotImplementedError:
            # mat files v7.3
            img = h5py.File(filenames[0])
            IA = np.transpose(np.array(img['IA']))
    else:
        # IA = Image.open(filenames[0])
        # IA.load()
        IA = np.asarray(Image.open(filenames[0])).copy()

    # image B
    if filenames[1][-4:] == ".mat":
        try:
            # mat files <7.3
            img = sio.loadmat(filenames[1])
            IB = np.array(img['IB'])
            pass
        except NotImplementedError:
            # mat files v7.3
            img = h5py.File(filenames[1])
            IB = np.transpose(np.array(img['IB']))
    else:
        IB = np.asarray(Image.open(filenames[1])).copy()

    # mask
    if filenames[2] is None:
        mask = np.ones(np.shape(IA))
    else:
        mask = np.asarray(Image.open(filenames[2]).convert('L')).copy()
        mask[mask > 0] = 1

    return IA, IB, mask


def load_PIVImage(flowtype, im_number):
    """Creates a PIVimage object for the specified flowtype and image number

    Args:
        flowtype (Int): The flowtype of the desired piv images.
                        For more information call
                        image_info.list_available_flowtypes()
        im_number (Int): The number in the ensemble to load into memory.
                         If im_number is greater than the known number of images
                         for the specified flowtype, a warning will be raised
                         The method will still try to open the requested file
                         If the file does not exist then an error will
                         be raised

    Examples:
        >>> import image_info
        >>> image_info.list_available_flowtypes() # to obtain options
        >>> img_obj = piv_image.load_PIVImage(flowtype=1, im_number=20)
        >>> IA, IB, mask = piv_image.load_image_from_flow_type(1, 20)
        >>> img_obj2 = piv_image.PIVImage(IA, IB, mask)
        >>> img_obj == img_obj2
        ... True

    No Longer Returned:
        IA (ndarray): Image intensities for the first in the image pair
        IB (ndarray): Image intensities for the second in the image pair
        mask (ndarray): mask values with 0 for no mask and 1 for mask
    """

    # load images
    IA, IB, mask = load_images(flowtype, im_number)
    return PIVImage(IA, IB, mask)


def quintic_spline_image_filter(IA):
    """
    Performs a quintic spline causal and anti-causal filter

    Refer to:
        Unser M., Aldroubi A., Eden M., 1993,
            "B-Spline Signal Processing: Part I - Theory",
            IEEE Transactions on signal processing, Vol. 41, No.2, pp.821-822
        Unser M., Aldroubi A., Eden M., 1993,
            "B-Spline Signal Processing: Part II -
            Efficient Design and Applications",
            IEEE Transactions on signal processing, Vol. 41, No.2, pp.834-848
        uk.mathworks.com/matlabcentral/fileexchange/19632-n-dimensional-bsplines


    Args:
        IA (ndarray): Image intensities to be filtered

    Returns:
        ndarray: The quintic splint filtered image

    Raises:
        ValueError: Image dimensions must be at least 43px
    """

    # doesn't work if the image is less than 43pixels wide/high
    if np.shape(IA)[0] < 43:
        raise ValueError("number of pixels in x and y must be at least 43")
    if np.shape(IA)[1] < 43:
        raise ValueError("number of pixels in x and y must be at least 43")

    # define coefficients
    scale = 120
    z = [-0.430575347099973, -0.0430962882032647]  # poles
    K0_tol = np.spacing(1)

    # initialise output
    C = IA * scale * scale
    dims = np.shape(C)
    C_rows = dims[0]
    C_cols = dims[1]

    # start = time.time()

    for i in range(2):
        K0 = math.ceil(math.log(K0_tol) / math.log(np.absolute(z[i])))
        indices = np.arange(K0)

        # scaling term for current pole
        C0 = -z[i] / (1 - z[i]**2)

        # column wise for each pole
        # apply symmetric filter over each column
        for k in range(C_cols):
            C[:, k] = sym_filt.sym_exp_filt(
                C[:, k], C_rows, C0, z[i], K0, indices)

        # row-wise for each pole
        # apply symmetric filter over each column
        for k in range(C_rows):
            C[k, :] = sym_filt.sym_exp_filt(
                C[k, :], C_cols, C0, z[i], K0, indices)

    # print("time: {}".format(time.time() - start))

    return C


if __name__ == "__main__":
    img = load_PIVImage(1, 1)
    u, v = 5 * np.ones((640, 1280)), 5 * np.ones((640, 1280))
    dp = dense_predictor.DensePredictor(u, v)
    img_def = img.deform_image(dp)

    # save into mat file
    IA, IB = img.IA, img.IB
    IAf, IBf = img_def.IA, img_def.IB
    mdict = {"IA": IA,
             "IB": IB,
             "IAf": IAf,
             "IBf": IBf,
             "u": u,
             "v": v,
             }
    sio.savemat("test_file.mat", mdict)
