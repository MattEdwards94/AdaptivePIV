import os
import numpy as np
import math
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1

_verbosity = 4


class MeanAndVarCalculator():

    def __init__(self, init_values):
        """Initialises the updating algorithm with the first values

        Args:
            init_values (ndarray): Initial displacement values
        """

        self.mean = init_values
        self.S = np.zeros_like(init_values)
        self.N = 1

    def __eq__(self, other):
        """
        Allow for comparing equality between MeanAndVarCalculator classes

        Args:
            other (MeanAndVarCalculator): The other MeanAndVarCalculator
                                          to be compared to

        Returns:
            Bool: Whether the two MeanAndVarCalculator match
        """

        if not isinstance(other, MeanAndVarCalculator):
            return NotImplemented

        for s, o in zip(self.__dict__.values(), other.__dict__.values()):
            if not np.allclose(s, o, equal_nan=True):  # check for equality
                return False

        return True

    @property
    def dim(self):
        """Gets the dimensions of the displacement field being calculated

        Returns:
            tuple: shape of the displacement field.
        """
        return np.shape(self.mean)

    @property
    def variance(self):
        """Gets the sample variane of the population over the domain

        Returns:
            ndarray: Sample variance of population
        """
        if self.N == 1:
            return np.zeros_like(self.mean)
        else:
            return self.S / (self.N - 1)

    def add_values(self, values):
        """Adds another set of sample data to the distribution

        Updates the mean and the variance

        Args:
            values (ndarray): new displacement values to add to the ensemble
        """

        # check first that the input values are the same dimensions as the
        # stored values
        if not np.all(np.shape(values) == self.dim):
            print("Shape of input: {}".format(np.shape(values)))
            print("Shape of stored values: {}".format(self.dim))
            raise ValueError("Dimensions must match")

        self.N += 1
        bf = values - self.mean
        self.mean += bf / self.N
        self.S += bf * (values - self.mean)


def elementwise_diff(A):
    """Returns the list of differences i.e. A[i+1] - A[i]

    Args:
        A (list, numeric): the input list to calculate the diff of

    Returns:
        (list, numeric): list of differences of next element to current element

    Raises:
        ValueError: If the input has only 0 or 1 elements
    """
    if len(A) < 2:
        raise ValueError("More than 2 elements needed to perform diff")

    return [nxt - curr for curr, nxt in zip(A, A[1:])]


def auto_reshape(x, y, f1=None, f2=None):
    """
    Returns 2D structured data of unknown dimensions, from the flattened data

    If the dimensions are known then reshape should be used.

    Args:
        x (list): flattened structured data locations
        y (list): flattened structured data locations
        f1 (list, optional): flattened structured data values
        f2 (list, optional): flattened structured data values. Optional to allow
                            for same functionality for either a single set of
                            data values or for two sets of values, e.g. u,v
    """

    # get spacing in x by finding the first value which is different from the
    # preceeding value
    # this should be the first element if we are flattening row wise
    x_diff = elementwise_diff(x)
    for ii, item in enumerate(x_diff):
        if not item == 0:
            x_spacing = item
            break

    # get spacing in y as above
    # also we need to grab the location where the value is different, so that
    # we can work out where the correct dimensions to return
    y_diff = elementwise_diff(y)
    for ii, item in enumerate(y_diff):
        if not item == 0:
            y_spacing = item
            # get the length of the row, i.e. the x_dim
            x_dim = ii + 1
            break

    # get y dim by seeing how many rows fit into the total number of elements
    y_dim = int(len(x) / x_dim)

    # now reshape the input arrays and data
    x_2d = np.reshape(x, (y_dim, x_dim))
    y_2d = np.reshape(y, (y_dim, x_dim))

    # check that the array is valid - i.e. equivalent to a meshgrid
    # get the first and last values of x and y
    x_strt, y_strt = x[0], y[0]
    x_end, y_end = x_2d[0, -1], y_2d[-1, 0]

    # now create equivalent meshgrid
    X_check, Y_check = np.meshgrid(np.arange(x_strt, x_end + 1, x_spacing),
                                   np.arange(y_strt, y_end + 1, y_spacing), )

    if not (np.allclose(X_check, x_2d) and np.allclose(Y_check, y_2d)):
        raise ValueError(
            "Input must be sorted, i.e. just flattened along rows")

    # determine outputs
    if (f1 is not None) and (f2 is None):  # f1 defined, f2 is not
        f1_2d = np.reshape(f1, (y_dim, x_dim))
        return x_2d, y_2d, f1_2d
    elif (f1 is None) and (f2 is not None):  # f2 is defined, f1 is not
        f2_2d = np.reshape(f2, (y_dim, x_dim))
        return x_2d, y_2d, f2_2d
    elif (f1 is not None) and (f2 is not None):  # both f1 and f2 are defined
        f1_2d = np.reshape(f1, (y_dim, x_dim))
        f2_2d = np.reshape(f2, (y_dim, x_dim))
        return x_2d, y_2d, f1_2d, f2_2d
    else:  # neither f1 or f2 are defined
        return x_2d, y_2d


def lin_extrap_edges(f, n_pad=1):
    """
    Extends the values of f by n_pad rows/columns around the array.
    By default n_pad is 1
    If the input is one dimensional then the output is only padded in the same
    dimension.

    Args:
        f (ndarray): array of values to extrapolate
        n_pad (int, optional): number or rows/columns to extend the input by.
                               Default is 1

    Returns:
        ndarray: padded output array
    """

    f = np.asarray(f)

    if f.ndim == 1:
        # calculate the gradient at the start and end
        start_grad = f[1] - f[0]
        # deliberately this way such that +dx is obtained
        end_grad = f[-1] - f[-2]

        # extend
        prepend = [f[0] - ii * start_grad for ii in range(n_pad, 0, -1)]
        append = [f[-1] + ii * end_grad for ii in range(1, n_pad + 1)]

        out = np.concatenate((prepend, f, append))
        return out
    else:
        # calculate gradients left and right
        l_grad, r_grad = f[:, 1] - f[:, 0], f[:, -1] - f[:, -2]

        # extend in x and y
        prepend = [f[:, 0] - ii * l_grad for ii in range(n_pad, 0, -1)]
        append = [f[:, -1] + ii * r_grad for ii in range(1, n_pad + 1)]

        wide = np.hstack((np.transpose(prepend), f, np.transpose(append)))

        # repeat for up and down
        u_grad, d_grad = wide[1, :] - wide[0, :], wide[-1, :] - wide[-2, :]

        # extend in x and y
        prepend = [wide[0, :] - ii * u_grad for ii in range(n_pad, 0, -1)]
        append = [wide[-1, :] + ii * d_grad for ii in range(1, n_pad + 1)]

        out = np.vstack((prepend, wide, append))

        return out


def round_to_odd(val):
    """
    Rounds the input value to the nearest odd integer
    Even values round up to the next odd

    Args:
        WS (float): non-odd-integer value to be rounded up
    """

    # round down to nearest integer
    bf = math.floor(val)

    # if even, increment
    return bf + (1 - (bf % 2))


def nice_print_dict(d):
    """Prints a dictionary nicely

    Args:
        values (dict): Dictionary of data to output nicely
    """

    # find the longest word in the list of keys
    length = 0
    for word in d.keys():
        if len(word) > length:
            length = len(word)

    for key, value in d.items():
        print(("{:>" + str(length + 1) + "}: {}").format(key, value))


def root_path():
    """Gets the path to the root.

    For my machine this will be
        "C:/Users/me12288/Documents"
    For BC3 it will be
        "/newhome/me12288"
    """

    # get current path
    cwd = os.getcwd()

    if "c:" in cwd.lower():
        if "me12288" in cwd:
            return "C:/Users/me12288/Documents/"
        else:
            return "C:/Users/Matt/MyDocuments/General/PhD/"
    else:
        return "/newhome/me12288/"


def plot_adjacent_images(ia, ib,
                         title_a, title_b,
                         vminmax_a=[None, None],
                         vminmax_b=[None, None],
                         cmap_a="gray",
                         cmap_b="gray",
                         fig=None, figsize=(20, 10),
                         share_all=True,
                         axes_pad=0.1, share_all_axes=True,
                         cbar_location="right", cbar_mode="single",
                         **kwargs):
    """Plots two figures nicely side by side, 
    and returns the figure and axes handles
    """

    # create the figure is one is not given
    if fig is None:
        fig = plt.figure(figsize=figsize)

    grid = axgrid1.ImageGrid(fig, 121, nrows_ncols=(1, 2),
                             axes_pad=axes_pad, share_all=share_all,
                             cbar_location=cbar_location,
                             cbar_mode=cbar_mode)

    ax1 = grid[0]
    im_a = ax1.imshow(ia, vmin=vminmax_a[0], vmax=vminmax_a[1], cmap=cmap_a)
    ax1.set_title(title_a)
    ax1.invert_yaxis()

    ax2 = grid[1]
    ax2.imshow(ib, vmin=vminmax_b[0], vmax=vminmax_b[1], cmap=cmap_b)
    ax2.set_title(title_b)
    ax2.invert_yaxis()

    grid.cbar_axes[0].colorbar(im_a)

    return fig, ax1, ax2


class SummedAreaTable():
    """Creates a summed area table to allow for rapid extraction of the
    summation of a submatrix of an array
    """

    def __init__(self, IA):
        """Initialises the summed area table for an input array IA

        Arguments:
            IA {ndarray} -- Input array to create the SAT from
        """

        # sum the rows and then the columns
        self.SAT = IA.cumsum(axis=1).cumsum(axis=0)
        self.img_dim = np.shape(IA)

    def get_area_sum(self, left, right, bottom, top):
        """Gets the sum of the region defined by left/right/bottom/top

        The sum is inclusive of all pixels defined by l:r:b:t
        This is DIFFERENT to the standard behaviour of numpy indexing. 
        For example:
            The following is the sum of rows 1-9, inclusive, and 
            columns 4-7 inclusive.
            a = np.sum(A[1:10, 4:8])

            For equivalent behaviour using a summed area table
            st = SummedAreaTable(A)
            a = st.get_area_sum(4, 7, 1, 9)


        Arguments:
            left {int} -- The left most coordinate of the region to search
            right {int} -- The rightmost coordinate of the region to search.
                        This must be greater than the left side
            bottom {int} -- The bottom of the region to search
            top {int} -- The top of the region to search. This must be greater
                        than the bottom of the region
        """
        if right < left:
            raise ValueError("The right must be >= left")
        if top < bottom:
            raise ValueError("The top must be >= bottom")

        # bounds check the inputs
        top = min(max(top, 0), self.img_dim[0]-1)
        right = min(max(right, 0), self.img_dim[1]-1)

        # define the square as
        # A -- B
        # |    |
        # |    |
        # C -- D
        # The sum of the region, including B, excluding the rest, is thus:
        # B - A - D + C
        # note that C is added due to it being doubly subtracted by A and D
        # refer to https://en.wikipedia.org/wiki/Summed-area_table for more
        # information
        #
        # also note that if A or C are on the first column, then they should
        # be 0 in the SAT, likewise if C or D are below the first row
        A = self.SAT[top, left-1] if left > 0 else 0
        B = self.SAT[top, right]
        C = self.SAT[bottom-1, left-1] if left > 0 and bottom > 0 else 0
        D = self.SAT[bottom-1, right] if bottom > 0 else 0

        return (B - A - D + C)

    def get_total_sum(self):
        """Returns the sum of values over the whole domain

        i.e. the top right value
        """
        return self.SAT[-1, -1]

    def fixed_filter_convolution(self, filt_size):
        """Gets the effective unity weighted fixed convolution of a filter over
        the whole domain. 

        Is equivalent to looping over every pixel and working out the sum within
        a region equal to filter, centered on each pixel. 
        Values outside of the domain are assumed to be 0

        Must be an odd filter size

        Arguments:
            filt_size {int, odd} -- The size of the filter to apply over 
                                    the domain
        """

        if not filt_size % 2:
            raise ValueError("The filter size must be odd")

        rad = int((filt_size - 1) / 2)

        # using pad in this way shifts the elements of the array, and fills in
        # to the correct size, using the edge value as the fill.
        # this makes it such that the window will effectively sum 0 values
        # outside of the image

        # note that the comments define the direction that we want to move
        # the desired reference pixel in. This is the opposite to the direction
        # that the actual array is moving in.

        # MODIFY THIS CODE WITH CAUTION

        # shift the top right down and to the left, keeping the values
        # at the edges
        tr = np.pad(self.SAT, ((0, rad), (0, rad)),
                    mode='edge')[rad:, rad:]

        # shift the top left down and to the right. Keep the values along the
        # top edge, set new values to 0 along the left edge
        tl = np.pad(self.SAT, ((0, rad), (0, 0)),
                    mode='edge')[rad:, :]
        tl = np.pad(tl, ((0, 0), (rad+1, 0)),
                    mode='constant')[:, :-(rad+1)]

        # shift the bottom right up and to the left. Keep the values along the
        # right hand edge, set new values to 0 along the bottom
        br = np.pad(self.SAT, ((0, 0), (0, rad)),
                    mode='edge')[:, rad:]
        br = np.pad(br, ((rad+1, 0), (0, 0)),
                    mode='constant')[:-(rad+1), :]

        # shift the bottom left up and to the right.
        # Set all new values to 0
        bl = np.pad(self.SAT, ((rad+1, 0), (rad+1, 0)),
                    mode='constant')[:-(rad+1), :-(rad+1)]

        return tr - tl - br + bl


def vprint(thr, *args, **kwargs):
    """Checks if the _verbosity is equal to or above the threshold,
    otherwise it doesn't print

    Args:
        thr (int) -- Threshold for verbosity. If _verbosity is >= thr, then 
                     vprint will behave as the builtin `print'. Otherwise
                     nothing will happen.
    """
    global _verbosity

    if _verbosity >= thr:
        print(*args, **kwargs)


if __name__ == '__main__':
    pass
