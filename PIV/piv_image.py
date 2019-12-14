import PIV.image_info as image_info
import scipy.io as sio
from scipy import interpolate as interp
from scipy.special import erf
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import numpy as np
import math
import sym_filt
import PIV.dense_predictor as dense_predictor


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

    def plot_mask(self):
        """
        Plots the mask as a black and white image, where the mask it black
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(self.mask)


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
    # print(filenames)

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
    C_rows = int(dims[0])
    C_cols = int(dims[1])
    # print(type(C_rows))

    # start = time.time()

    for i in range(2):
        K0 = math.ceil(math.log(K0_tol) / math.log(np.absolute(z[i])))
        indices = np.arange(K0, dtype=np.int32)
        # print(type(indices))

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


def generate_particle_locations(img_dim, seed_density,
                                u, v,
                                d_tau_mean=2.5, d_tau_std=0.25,
                                int_mean=0.9, int_std=0.05,
                                **kwargs):
    """Generates particles for synthetic generation

        Arguments:
            img_dim {int, tuple} -- The height and width of the image in pixels
            seed_density {float} -- Approximate number of particles per pixel
            u {float, ndarray} -- Horizontal displacement field 
                                  defined pixelwise
            v {float, ndarray} -- Vertical displacement field, defined pixelwise
            d_tau_mean {float} -- The average particle image diameter px
            d_tau_std {float} -- The standard deviation of particle 
                                 image diameters
            int_mean {float} -- The mean intensity of particle images, 0-1
            int_std {float} -- The standard deviation of intensities 
                               about the mean

        Returns:
            xp1, yp1, xp2, yp2 (float, list) -- The locations of particles in
                                                the first and second timestep 
            d_tau, Ip (float, list) -- Properties of the particle images
    """
    # extend the particle seed beyond the edge of the domain, equal to the
    # maximum displacement and the particle diameter, such that particles
    # can enter the domain in the second iteration as they would in reality
    max_disp = 10

    # create a random distribution of particles over the domain
    n_part = int((img_dim[0]+d_tau_mean+max_disp) *
                 (img_dim[1]+d_tau_mean+max_disp) *
                 seed_density)

    xp1 = np.random.uniform(-d_tau_mean-max_disp,
                            img_dim[1]+d_tau_mean+max_disp,
                            n_part)
    yp1 = np.random.uniform(-d_tau_mean-max_disp,
                            img_dim[0]+d_tau_mean+max_disp,
                            n_part)
    d_tau = np.random.normal(d_tau_mean, d_tau_std, n_part)
    Ip = np.random.normal(int_mean, int_std, n_part)

    # interpolate the displacement field to obtain a function
    # this allows us to calculate the sub-pixel displacement
    f_u = interp.interp2d(np.arange(img_dim[1]),
                          np.arange(img_dim[0]), u)
    f_v = interp.interp2d(np.arange(img_dim[1]),
                          np.arange(img_dim[0]), v)
    up1 = [f_u(x, y)[0] for x, y in zip(xp1, yp1)]
    vp1 = [f_v(x, y)[0] for x, y in zip(xp1, yp1)]

    # diplace the particle images
    xp2 = xp1 + up1
    yp2 = yp1 + vp1

    return xp1, yp1, xp2, yp2, d_tau, Ip


def render_synthetic_PIV_image(height, width,
                               x_part, y_part,
                               d_tau, part_intens,
                               bit_depth=8, fill_factor=0.85,
                               noise_mean=0.05, noise_std=0.025,
                               **kwargs):
    """Renders a single PIV image based on the specified 
    particle locations and intensities

    Arguments:
        height (Int) -- The size of the image
        width (Int) -- The size of the image
        x_part (ndarray) -- The x location of the particles
        y_part (ndarray) -- The y location of the particles
        d_tau (ndarray) -- The diameters of the particle images, pixels
        part_intens (ndarray) -- The intensities of the particles, 0-1
        bit_depth (Int) -- The bit depth of the output image. 

    Returns:
        Image: numpy array
            The desired synthetic image.

    """

    # prepare output
    im_out = np.zeros([height, width])

    # calculate some constant terms outside of the loop for efficiency
    sqrt8 = np.sqrt(8)
    ccd_fill = fill_factor * 0.5
    one32 = 1/32

    for i in range(x_part.size):
        par_diam = d_tau[i]
        x = x_part[i]
        y = y_part[i]

        bl = int(max(x - par_diam, 0))
        br = int(min(x + par_diam, width))
        bd = int(max(y - par_diam, 0))
        bu = int(min(y + par_diam, height))

        # Equation 6 from europiv SIG documentation has:
        # d_particle^2 * r_tau ^ 2 * pi/8
        # the dp^2 is to reflect the fact that bigger particles
        # scatter more light proportional to dp^2
        # this is implicitly governed by part_intens[ind]
        # the r_tau^2 is actually r_tau_x * r_tau_y
        # this appears to come from the integration of the continuous
        # equation
        scale_term = par_diam * par_diam * np.pi * part_intens[i] * one32

        for c in range(bl, br):
            for r in range(bd, bu):
                im_out[r, c] = im_out[r, c] + scale_term * (
                    # assumes a fill factor of 1 -> the 0.5 comes
                    # from fill_factor * 0.5
                    # sqrt8 comes from the erf( ... / (sqrt2 * par_radius))
                    # hence 2 * ... / sqrt2 * par_diam
                    # hence sqrt8 * ... / par_diam
                    erf(sqrt8 * (c - x - ccd_fill) / par_diam) -
                    erf(sqrt8 * (c - x + ccd_fill) / par_diam)
                ) * (
                    erf(sqrt8 * (r - y - ccd_fill) / par_diam) -
                    erf(sqrt8 * (r - y + ccd_fill) / par_diam)
                )

    # calculate the noise to apply to the image
    noise = np.random.normal(noise_mean, noise_std, (height, width))

    # cap at 0 - 1
    im_out = np.maximum(np.minimum(im_out + noise, 1), 0)

    # return the quantized image
    return (im_out*(2**bit_depth - 1)).astype(int)


def create_synthetic_image_pair(img_dim, seed_dens, u, v, **kwargs):
    """Helper function to generate synthetic PIV images

    For additional arguments, refer to:
        generate_particle_locations
        render_synthetic_PIV_image

    Arguments:
        img_dim {int, tuple} -- The height and width of the images, pixels
        seed_dens {float} -- The density of particle images in particles per px
        u {float, ndarray} -- Pixelwise horizontal displacement field
        v {float, ndarray} -- Pixelwise vertical displacement field
    """

    # get particle image locations
    (xp1, yp1,
     xp2, yp2,
     d_tau, Ip) = generate_particle_locations(img_dim, seed_dens,
                                              u, v,
                                              **kwargs)

    img_a = render_synthetic_PIV_image(img_dim[0], img_dim[1],
                                       xp1, yp1,
                                       d_tau, Ip, **kwargs)
    img_b = render_synthetic_PIV_image(img_dim[0], img_dim[1],
                                       xp2, yp2,
                                       d_tau, Ip, **kwargs)

    return img_a, img_b


def plot_pair_images(ia, ib, fig_size=(20, 10), n_bits=None):
    """Plot PIV images side by side

    Note that the fig_size parameter does not seem to be very precise

    Arguments:
        ia {ndarray} -- The first image in the snapshot
        ib {ndarray} -- The second image in the snapshot

    Keyword Arguments:
        fig_size {float, tuple} -- Size of the figure, supposedly in inches
                                   (default: {(20, 10)})
        n_bits {int} -- The number of bits in the image. If nothing
                        is passed then the number of
                        bits will be estimated from the 
                        peak intensity within ia and ib (default: {None})

    Returns:
        fig, ax1, ax2 -- figure and axes handles
    """

    # create the figure
    fig = plt.figure(figsize=fig_size)
    # create a 1 by 2 grid for the images to go side by side
    grid = axgrid1.ImageGrid(fig, 121,
                             nrows_ncols=(1, 2), axes_pad=0.1,
                             share_all=True,  # means that the axes are shared
                             cbar_location="right",
                             cbar_mode="single")

    # determine how many bits are being used to scale the colormap
    n_bits = math.ceil(math.log2(np.max(np.maximum(ia, ib))))

    a = grid[0]
    ima = a.imshow(ia, vmin=0, vmax=2**n_bits-1, cmap='gray')
    a.set_title('Frame A')
    a.invert_yaxis()

    b = grid[1]
    imb = b.imshow(ib, vmin=0, vmax=2**n_bits-1, cmap='gray')
    b.set_title('Frame B')
    b.invert_yaxis()

    grid.cbar_axes[0].colorbar(ima)

    return fig, a, b


if __name__ == "__main__":
    # img = load_PIVImage(1, 1)
    # u, v = 5 * np.ones((640, 1280)), 5 * np.ones((640, 1280))
    # dp = dense_predictor.DensePredictor(u, v)
    # img_def = img.deform_image(dp)

    # # save into mat file
    # IA, IB = img.IA, img.IB
    # IAf, IBf = img_def.IA, img_def.IB
    # mdict = {"IA": IA,
    #          "IB": IB,
    #          "IAf": IAf,
    #          "IBf": IBf,
    #          "u": u,
    #          "v": v,
    #          }
    # sio.savemat("test_file.mat", mdict)

    (x_part, y_part,
     d_tau, part_intens,
     x_part2, y_part2) = generate_particle_locations((50, 50), 0.01,
                                                     5*np.ones((50, 50)), np.zeros((50, 50)))

    render_synthetic_PIV_image(50, 50, x_part, y_part, d_tau, part_intens)
