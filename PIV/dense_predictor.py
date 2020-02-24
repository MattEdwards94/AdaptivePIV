import numpy as np
import matplotlib.pyplot as plt
import PIV.utilities as utils
import PIV.image_info as im_info
import PIV.piv_image as piv_image
import scipy.io


class DensePredictor:
    """Class to contain the displacement fields

    Attributes:
        has_mask (bool): Indicates if a mask is present
        u (ndarray): The horizontal displacement field
        v (ndarray): The vertical displacement field
        mask (ndarray): Mask flag array, 1 indicates no mask, 0 indicates mask
        n_rows (int): Number of rows
        n_cols (int): Number of columns
        dim (tuple): (n_rows, n_cols)

    """

    def __init__(self, u, v, mask=None):
        """
        constructs the densepredictor object from the input displacement fields
        u, v, mask must all be the same size

        Args:
            u (ndarray): The horizontal displacement field
            v (ndarray): The vertical displacement field
            mask (ndarry, optional): mask flag array. 1 is no mask, 0 is mask
                                     If not specified then an array of 1's is
                                     created

        Examples:
            >>> uIn = np.random.rand(100, 100)
            >>> vIn = np.random.rand(100, 100)
            >>> mask = np.random.randint(0, 2, (100, 100))
            >>> dp = dense_predictor.DensePredictor(uIn, vIn, mask)

        Raises:
            ValueError: If the sizes of u, v, or mask are not consistent

        """

        # check that the inputs are the same size
        if np.shape(u) != np.shape(v):
            raise ValueError("shape of u must match the shape of v")

        if mask is not None:
            if np.shape(u) != np.shape(mask):
                raise ValueError("The shape of the mask must match u and v")
            self.has_mask = True
        else:
            mask = np.ones(np.shape(u))
            self.has_mask = False

        self.u = np.array(u, dtype=np.float_)
        self.v = np.array(v, dtype=np.float_)
        self.mask = mask
        self.apply_mask()
        self.n_rows = np.shape(u)[0]
        self.n_cols = np.shape(u)[1]
        self.dim = (self.n_rows, self.n_cols)

        # create sat for u, v, and mask
        self.u_sat = utils.SummedAreaTable(self.u)
        self.v_sat = utils.SummedAreaTable(self.v)
        self.mask_sat = utils.SummedAreaTable(self.mask)

    @staticmethod
    def from_dimensions(dim, value=None):
        """Provides a mechanism to initialise a densepredictor from just the 
        dimesions

        Arguments:
            dim {tuple, int} -- The height and width, respectively, of the 
                                desired denspredictor
            value {tuple, float} -- The u and v values, respectively, to 
                                    initialise the displacement field to. 
                                    By default sets the values to 0 everywhere.

        Returns:
            DensePredictor 
        """

        if value is None:
            value = (0, 0)

        u, v = np.ones(dim)*value[0], np.ones(dim)*value[1]
        dp_out = DensePredictor(u, v)

        return dp_out

    @staticmethod
    def load_true(flowtype):
        """Loads the true displacement field for a given flow type


        Arguments:
            flowtype {int} -- Integer value representing the flow 'id' as per
                              index.csv. 

        Returns: 
            DensePredictor
        """

        info = im_info.ImageInfo(flowtype)
        true_filename = info.vel_field_fname

        if true_filename == None:
            raise ValueError("No displacement field "
                             "defined for flowtype{}".format(flowtype))
        else:
            uv = scipy.io.loadmat(true_filename)

        mask = piv_image.load_mask(flowtype)

        dp_out = DensePredictor(uv["u"], uv["v"], mask)

        return dp_out

    def get_region(self, x, y, rad, truncate=True):
        """
        Retrieves the displacements for the region requested.
        If the region requested extends beyond the image dimensions then the
        values are either truncated or padded with 0's

        Matrix access is base 0 and is ROW major:
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
            truncate (bool, optional): True - truncates the returned values
                                       outside of the domain
                                       False - pads with zeros beyong the domain

        Returns:
            u (ndarray): Horizontal displacement values around (x, y)+-rad
            v (ndarray): Vertical displacement values around (x, y)+-rad
            mask (ndarray): Mask flag values around (x, y)+-rad

        Examples:
            >>> uIn = np.random.rand(100, 100)
            >>> dp = dense_predictor.DensePredictor(uIn, uIn)
            >>> u, v, mask = dense_predictor.DensePredictor(3, 3, 2)
            >>> np.shape(u)
            ... (5, 5)
            >>> np.allclose(uIn[0:5, 0:5], u)
            ... True
        """

        """
        extract what region we can within the image.
        to do this we need to know where we are with respect to the limits
        of the image
        """
        # if x - rad is < 0, set to 0, if > n_cols, set to n_cols - 1
        # the minus 1 is because of 0 indexing
        left = max(min(x - rad, self.n_cols - 1), 0)
        right = max(min(x + rad, self.n_cols - 1), 0)
        bottom = max(min(y - rad, self.n_rows - 1), 0)
        top = max(min(y + rad, self.n_rows - 1), 0)

        # extract this region out of the images/mask
        # note the +1 is because left:right is not inclusive of right
        u = self.u[bottom:top + 1, left:right + 1]
        v = self.v[bottom:top + 1, left:right + 1]

        # decide whether to pad or truncate
        if truncate:
            if self.has_mask:
                mask = self.mask[bottom:top + 1, left:right + 1]
            else:
                mask = np.ones(np.shape(u))
        else:
            # now pad the image with 0's if ctr +- rad overlaps the edge
            pl = max(rad - x, 0)
            pr = max(x + rad - self.n_cols + 1, 0)
            pb = max(rad - y, 0)
            pt = max(y + rad - self.n_rows + 1, 0)
            pad = ((pb, pt), (pl, pr))

            u = np.pad(u, pad, 'constant', constant_values=0)
            v = np.pad(v, pad, 'constant', constant_values=0)
            if self.has_mask:
                mask = np.pad(
                    self.mask[bottom:top + 1, left:right + 1], pad,
                    'constant', constant_values=0)

        return u, v, mask

    def __eq__(self, other):
        """
        Compares two DensePredictor objects to determine equality

        Considered equal if:
            self.u == other.u
            self.v == other.v
            self.mask == other.mask

        Args:
            other (DensePredictor): The other DensePredictor to be compared to

        Returns:
            Bool: Wheter the two Densepredictors are considered equal
        """

        if not isinstance(other, DensePredictor):
            return NotImplemented

        if not np.alltrue(self.u == other.u):
            return False

        if not np.alltrue(self.v == other.v):
            return False

        if not np.alltrue(self.mask == other.mask):
            return False

        return True

    def __add__(self, other):
        """
        Overloads the operator for addition

        The masks must be identical otherwise a ValueError is raised

        Args:
            other (DensePredictor): Another Densepredictor object, must have the
                                    same dimensions as self
                                    Must have the same mask

        Returns:
            DensePredictor: DensePredictor(self.u + other.u,
                                           self.v + other.v,
                                           self.mask)

        Raises:
            ValueError: If the input dimensions of self and other are not
                        the same
            ValueError: If the masks are not identical
        """

        # don't provide functionality for any other class
        if not isinstance(other, DensePredictor):
            return NotImplemented

        # check that the dimensions are the same
        if not np.alltrue(self.dim == other.dim):
            raise ValueError("DensePredictors must be the same size")

        # check if the masks are not the same
        if not np.alltrue(self.mask == other.mask):
            raise ValueError("The two masks are not identical")

        # calculate the sum
        newU = self.u + other.u
        newV = self.v + other.v
        dp = DensePredictor(newU, newV, self.mask)
        dp.apply_mask()

        return dp

    def __sub__(self, other):
        """
        Overloads the operator for subtraction

        The masks must be identical otherwise a ValueError is raised

        Args:
            other (DensePredictor): Another Densepredictor object, must have the
                                    same dimensions as self
                                    Must have the same mask

        Returns:
            DensePredictor: DensePredictor(self.u - other.u,
                                           self.v - other.v,
                                           self.mask)

        Raises:
            ValueError: If the input dimensions of self and other are not
                        the same
            ValueError: If the masks are not identical
        """

        # don't provide functionality for any other class
        if not isinstance(other, DensePredictor):
            return NotImplemented

        # check that the dimensions are the same
        if not np.alltrue(self.dim == other.dim):
            raise ValueError("DensePredictors must be the same size")

        # check if the masks are not the same
        if not np.alltrue(self.mask == other.mask):
            raise ValueError("The two masks are not identical")

        # calculate the sum
        newU = self.u - other.u
        newV = self.v - other.v
        dp = DensePredictor(newU, newV, self.mask)
        dp.apply_mask()

        return dp

    def __mul__(self, other):
        """
        Overloads the operator for multiplication

        The masks must be identical otherwise a ValueError is raised

        Args:
            other (DensePredictor): Another Densepredictor object, must have the
                                    same dimensions as self
                                    Must have the same mask

        Returns:
            DensePredictor: DensePredictor(self.u * other.u,
                                           self.v * other.v,
                                           self.mask)

        Raises:
            ValueError: If the input dimensions of self and other are not
                        the same
            ValueError: If the masks are not identical
        """

        # don't provide functionality for any other class
        if not isinstance(other, DensePredictor):
            return NotImplemented

        # check that the dimensions are the same
        if not np.alltrue(self.dim == other.dim):
            raise ValueError("DensePredictors must be the same size")

        # check if the masks are not the same
        if not np.alltrue(self.mask == other.mask):
            raise ValueError("The two masks are not identical")

        # calculate the sum
        newU = self.u * other.u
        newV = self.v * other.v
        dp = DensePredictor(newU, newV, self.mask)
        dp.apply_mask()

        return dp

    def __truediv__(self, other):
        """
        Overloads the operator for division

        The masks must be identical otherwise a ValueError is raised

        Args:
            other (DensePredictor): Another Densepredictor object, must have the
                                    same dimensions as self
                                    Must have the same mask

        Returns:
            DensePredictor: DensePredictor(self.u / other.u,
                                           self.v / other.v,
                                           self.mask)

        Raises:
            ValueError: If the input dimensions of self and other are not
                        the same
            ValueError: If the masks are not identical
        """

        # don't provide functionality for any other class
        if not isinstance(other, DensePredictor):
            return NotImplemented

        # check that the dimensions are the same
        if not np.alltrue(self.dim == other.dim):
            raise ValueError("DensePredictors must be the same size")

        # check if the masks are not the same
        if not np.alltrue(self.mask == other.mask):
            raise ValueError("The two masks are not identical")

        # calculate the sum
        with np.errstate(divide='ignore', invalid='ignore'):
            newU = self.u / other.u
            newV = self.v / other.v
        dp = DensePredictor(newU, newV, self.mask)
        dp.apply_mask()

        return dp

    def get_local_avg_disp(self, x, y, rad):
        """Return the local average displacement of the dp, within the 
        non-masked region only

        Args:
            x (int): The horizontal location to center the extraction around
            y (int): The vertical location to center the extraction around
            rad (int): The distance either side of (x,y) to average the 
                       displacement over, such that the total region is
                       rad*2 + 1
        """

        # get the local sum, noting that masked values are 0
        sum_u = self.u_sat.get_area_sum(x-rad, x+rad, y-rad, y+rad)
        sum_v = self.v_sat.get_area_sum(x-rad, x+rad, y-rad, y+rad)
        sum_mask = self.mask_sat.get_area_sum(x-rad, x+rad, y-rad, y+rad)

        return sum_u / sum_mask, sum_v / sum_mask

    def apply_mask(self):
        """
        Method to apply the mask
        locations where the mask is 0 will have the
        u, v displacements set to 0
        """

        inter = self.mask == 0
        self.u[inter] = 0
        self.v[inter] = 0

    def magnitude(self):
        """
        Returns the magnitude of the densepredictor
        """
        return np.sqrt(self.u * self.u + self.v * self.v)

    def plot_displacement_field(self, ax=None, spacing=16, **kwargs):
        """
        Plots the displacement field

        Args:
            axes (None, optional): Handle to the axes to plot in.
                                   If none given a new set of axes
                                   shall be drawn
            spacing (int, optional): The spacing over vectors in the quiver.
        """

        xv, yv = (np.arange(0, self.n_cols, spacing),
                  np.arange(0, self.n_rows, spacing))
        u, v = (self.u[0::spacing, 0::spacing], self.v[0::spacing, 0::spacing])

        if ax is None:
            fig, ax = plt.subplots()

        if np.sum(self.mask) != np.prod(self.dim):
            ax.imshow(self.mask)

        ax.quiver(xv, yv, u, v, **kwargs)
        ax.set_title("displacement")

    def plot_u_magnitude(self, ax=None, **kwargs):
        """Plot the horizontal component of the velocity field as a contour

        Args:
            ax (None, optional): Optional axes handle to plot to
                                 If none given a new set of axes
                                 shall be drawn
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        im = ax.imshow(self.u, **kwargs)
        ax.figure.colorbar(im, ax=ax)
        ax.set_title("horizontal component")

    def plot_v_magnitude(self, ax=None, **kwargs):
        """Plot the vertical component of the velocity field as a contour

        Args:
            ax (None, optional): Optional axes handle to plot to
                                 If none given a new set of axes
                                 shall be drawn
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        im = ax.imshow(self.v, **kwargs)
        ax.figure.colorbar(im, ax=ax)
        ax.set_title("vertical component")

    def plot_contour_magnitude(self, **kwargs):
        fig, ax = plt.subplots()
        im = ax.imshow(self.magnitude(), **kwargs)
        ax.figure.colorbar(im)
