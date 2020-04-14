import numpy as np
from sklearn.neighbors import NearestNeighbors
import PIV.utilities as utilities
from PIV.utilities import vprint
import PIV.corr_window as corr_window
from scipy import interpolate as interp
import skimage.segmentation as sk_seg
from scipy.spatial import Delaunay, ConvexHull, Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import ais_module
import time


class Distribution:
    def __init__(self, init_locations=None):
        """
        Initialises the distribution object with 0, 1, or many correlation
        window objects

        Args:
            init_locations ([list] CorrWindow, optional): list of correlation
                                                          windows to initialise
                                                          the distribution with
        """

        # if there is nothing passed, then initialise empty list
        if init_locations is None:
            self.windows = []
        elif type(init_locations) == list:
            # if it is only a single or multiple inputs we need to add
            self.windows = init_locations.copy()
        else:
            self.windows = [init_locations]

    def __eq__(self, other):
        """
        Allow for comparing equality

        Args:
            other (Distribution): The other Distribution obj to be compared to

        Returns:
            Bool: Whether the two Distributions match
        """

        if not isinstance(other, Distribution):
            return NotImplemented

        for s, o in zip(self.__dict__.values(), other.__dict__.values()):
            if s != o:
                if not np.all(np.isnan((s, o))):
                    return False

        return True

    def n_windows(self):
        """
        Returns the number of windows currently stored in the distribution
        """
        return len(self.windows)

    @staticmethod
    def from_AIS(*args, **kwargs):
        """Helper method to initialise a Distribution object from AIS
        """
        xy = AIS(*args, **kwargs)
        return Distribution.from_locations(xy[:, 0], xy[:, 1])

    @staticmethod
    def from_locations(x, y, WS=None):
        """
        Creates a distribution of CorrWindow objects, made for each
        item in x, y and WS

        Inputs are unrolled, using np.ravel() into a one dimensional array
        before creating any CorrWindows which assumes row-major indexing.

        Args:
            x (list, int): The x location of the windows
            y (list, int): The y location of the windows
            WS (list, odd int): The window sizes

        Returns:
            Distribution: Distribution object containing the specified locations
        """

        if WS is None:
            WS = [None] * len(x)

        x, y, WS = np.array(x), np.array(y), np.array(WS)

        cwList = list(map(corr_window.CorrWindow,
                          x.ravel(),
                          y.ravel(),
                          WS.ravel()))

        return Distribution(cwList)

    @property
    def x(self):
        return np.array([cw.x for cw in self.windows])

    @property
    def y(self):
        return np.array([cw.y for cw in self.windows])

    @property
    def u(self):
        return np.array([cw.u for cw in self.windows])

    @property
    def v(self):
        return np.array([cw.v for cw in self.windows])

    @property
    def WS(self):
        return np.array([cw.WS for cw in self.windows])

    def get_all_xy(self):
        """
        Returns a (N, 2) array of all the stored locations where N is the
        total number of corr windows

        Returns:
            ndarray: (N, 2) array of all locations [x, y]
        """
        return np.array([[cw.x, cw.y] for cw in self.windows])

    def get_unmasked_win(self):
        """
        Returns a (N, 1) array of all unmasked stored locations where N is the
        number of unmasked CorrWindows

        If "is_masked" is not set then an error is raised.

        Returns:
            ndarray: (N, 2) array of all unmasked locations [x, y]
        """
        if self.windows[0].is_masked is None:
            raise ValueError("Mask status not known")
        out_list = []
        for cw in self.windows:
            if cw.is_masked is False and cw.is_halo is None:
                out_list.append(cw)

        return np.array(out_list)

    def get_unmasked_xy(self):
        """
        Returns a (N, 2) array of all unmasked stored locations where N is the
        number of unmasked CorrWindows

        If "is_masked" is not set then an error is raised.

        Returns:
            ndarray: (N, 2) array of all unmasked locations [x, y]
        """
        if self.windows[0].is_masked is None:
            raise ValueError("Mask status not known")
        out_list = []
        for cw in self.windows:
            if cw.is_masked is False and cw.is_halo is None:
                out_list.append([cw.x, cw.y])

        return np.array(out_list)

    def get_all_uv(self):
        """
        Returns a (N, 2) array of all the stored vectors

        Returns:
            ndarray: (N, 2) array of all vectors [u, v]
        """
        return np.array([[cw.u, cw.v] for cw in self.windows])

    def get_unmasked_uv(self):
        """
        Returns a (N, 2) array of all unmasked stored vectors where N is the
        number of unmasked CorrWindows

        If "is_masked" is not set then an error is raised.

        Returns:
            ndarray: (N, 2) array of all unmasked vectors [x, y]
        """
        if self.windows[0].is_masked is None:
            raise ValueError("Mask status not known")
        out_list = []
        for cw in self.windows:
            if cw.is_masked is False and cw.is_halo is None:
                out_list.append([cw.u, cw.v])

        return np.array(out_list)

    def get_all_WS(self):
        """
        Returns a (N, 1) array of all the stored window sizes

        Returns:
            ndarray: (N, 1) array of all window sizes
        """
        return np.array([cw.WS for cw in self.windows])

    def get_unmasked_WS(self):
        """
        Returns a (N, 2) array of all unmasked stored WS's where N is the
        number of unmasked CorrWindows

        If "is_masked" is not set then an error is raised.

        Returns:
            ndarray: (N, 2) array of all unmasked window sizes [x, y]
        """
        if self.windows[0].is_masked is None:
            raise ValueError("Mask status not known")
        out_list = []
        for cw in self.windows:
            if cw.is_masked is False and cw.is_halo is None:
                out_list.append(cw.WS)

        return np.array(out_list)

    def get_flag_values(self):
        """
        Returns:
            Returns an (N, 1) array of flag values
        """
        return np.array([cw.flag for cw in self.windows])

    def validation_NMT_8NN(self, threshold=2, eps=0.1, idw=False):
        """
        Performs a normalised median threshold test by comparing each vector to
        it's 8 nearest neighbours

        Invalid vectors are replaced by the median of the surrounding valid
        vectors within the 8 selected neighbours. If any of the surrounding
        8 neighbours are invalid, then this is not considered for the
        replacement. If all 8 neighbours are invalid, the centre vector is
        replaced by a 0.

        See:
            "Universal Outlier Detection for PIV Data" - Westerweel and Scarano


        Parameters
        ----------
        threshold : int, optional
            The threshold above which the norm indicates an outlier.
            Refer to paper
            "Universal Outlier Detection for PIV Data" - Westerweel and Scarano
        eps : float, optional
            The assumed background noise in px.
            Refer to paper.
        idw : Boolean, optional
            Indicates whether to perform distance weighting on validation
        """

        # detection
        # find neighbours
        xy, uv = self.get_unmasked_xy(), self.get_unmasked_uv()
        u, v = uv[:, 0], uv[:, 1]
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(xy)
        nb_dist, nb_ind = nbrs.kneighbors(xy)

        if not idw:
            norm = NMT_detection(u, v, nb_ind, eps)
        else:
            norm = NMT_idw_detection(u, v, nb_ind, nb_dist)
        flag = norm > threshold
        invalid = np.sum(flag)
        try:
            vprint(2, f"  {invalid}/{self.n_windows()} vectors replaced")
        except:
            vprint(2, f"  {invalid}/{self.n_windows} vectors replaced")

        # replacement
        u, v = outlier_replacement(flag, u, v, nb_ind)

        # update values in CorrWindow objects
        for (cw, outlier, u_i, v_i) in zip(self.get_unmasked_win(), flag, u, v):
            if cw.is_masked is True:
                continue
            if outlier == 1:
                cw.u_pre_validation, cw.v_pre_validation = cw.u, cw.v
                cw.u, cw.v = u_i, v_i
                cw.flag = outlier

        return flag

    def interp_to_densepred(self, method, eval_dim, inter_h=1):
        """
        Interpolates the displacement vectors held by the distribution object
        onto a pixel grid from 0 to the output dimensions
        The values are linearly extrapolated at the boundaries

        Args:
            method (string): type of interpolation, 'linear' or 'cubic'
            eval_dim (tuple): dimensions of the pixel grid to be evaluated

        Returns:
            (u, v): ndarrays of the u and v interpolated displacements, of
                    dimenions equal to eval_dim

        Raises:
            ValueError: if the method or evaluation dimensions are not valid
        """
        # check the input method
        acc_options = ["struc_lin", "struc_cub", "unstruc_cub"]
        if method not in acc_options:
            raise ValueError("Method not handled")

        # check the output dimensions - must be positive integers
        for dim in eval_dim:
            if (not np.all(dim == int(dim))) or np.any(dim < 1):
                raise ValueError("Dimensions must be positive integer")

        xy, uv = self.get_all_xy(), self.get_all_uv()

        if method in ["struc_lin", "struc_cub"]:
            # reshape the data onto a structured grid
            x, y, u, v = xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1]
            x2d, y2d, u2d, v2d = utilities.auto_reshape(x, y, u, v)

            u_int, v_int = interp_disp_structured(x2d, y2d, u2d, v2d,
                                                  eval_dim, method)
        else:
            # unstructured

            # extend convex hull
            ex_points, ex_vals = self.extend_convex_hull(eval_dim)
            xy = np.append(xy, ex_points, axis=0)
            uv = np.append(uv, ex_vals, axis=0)

            xe, ye = np.meshgrid(np.arange(0, eval_dim[1], inter_h),
                                 np.arange(0, eval_dim[0], inter_h))

            u_int, v_int = interp_cub_unstruc(xy, uv,
                                              eval_dim, inter_h,
                                              extend_hull=False)

            if inter_h > 1:
                u_int, v_int = interp_disp_structured(xe, ye, u_int, v_int,
                                                      eval_dim, "struc_cub")

        return u_int, v_int

    def correlate_all_windows(self, img, dp):
        """
        Loops through all the CorrWindow locations and correlates the two images


        Args:
            img (PIVImage): The PIVImage object containing the images and the
                            mask
            dp (DensePredictor): DensePredictor object containing the underlying
                                 displacment field, to be combined with the
                                 correlation result
        """

        for cw in self.windows:
            if cw.is_halo is True:
                continue
            cw.correlate(img, dp)

    def plot_locations(self, handle=None, mask=None, *args, **kwargs):
        """Plots the locations of all the windows within the distribution
        """

        if handle is None:
            fig, ax = plt.subplots(1)
        else:
            ax = handle

        if mask is not None:
            x, y = [], []
            for _x, _y in self.get_unmasked_xy():
                try:
                    if mask[_y, _x] == 1:
                        x.append(_x)
                        y.append(_y)
                except IndexError:
                    continue

        else:
            x, y = self.x, self.y

        ax.plot(x, y, 'r*', *args, **kwargs)
    def plot_distribution(self, handle=None):

        if handle is None:
            fig, ax = plt.subplots(1)
        else:
            ax = handle

        xy, uv = self.get_all_xy(), self.get_all_uv()
        ax.quiver(xy[:, 0],
                  xy[:, 1],
                  uv[:, 0],
                  uv[:, 1])
        plt.show()

    def AIW(self, img, dp,
            step_size=6, SNR_thr=1.4,
            max_WS=117, store_hist=True):
        """
        Analyses the contained distribution using Adaptive Initial Window sizing. 
        Correlates the windows and stores the results inside the class, as it
        would for correlate_all_windows

        Args:
            img (PIVImage): PIVImage object containing the images to be analysed
            NI_target (int, optional): The number of particles to target for 
                                       each correlation window. 
                                       Defaults to 25.
            step_size (int, optional): If a correlation fails, the amount to 
                                       increase the window size by. 
                                       Must be even integer. 
                                       Defaults to 6.
        """

        for cw in self.windows:
            if store_hist:
                history = np.empty((0, 4))
            # if WS is greater than max_WS to begin, we should just correlate
            # this indicates the seeding is extremely poor and is likely to
            # yield a poor correlation
            if cw.WS >= max_WS:
                cw.correlate(img, dp)
                continue

            while cw.WS < max_WS:
                cw.correlate(img, dp)
                if store_hist and cw.disp_mag() <= cw.WS*0.25:
                    history = np.vstack((history, [cw.WS, cw.u, cw.v, cw.SNR]))

                # now check validity of result
                if cw.SNR == 0:
                    # the location is masked
                    break
                elif (cw.SNR <= SNR_thr) or (cw.disp_mag() > cw.WS*0.25):
                    # if SNR is too low, or the displacement violates 1/4 rule
                    cw.WS += step_size
                    if cw.WS >= max_WS:
                        if store_hist:
                            # only update the values if we need to
                            ind = np.argmax(history, axis=0)
                            (cw.WS, cw.u, cw.v, cw.SNR) = history[ind[3], :]
                        break
                else:
                    # WS is ok
                    break

    def interp_WS_unstructured(self, mask):
        """Performs nearest neighbhour interpolation to get the WS

        Parameters
        ----------
        mask : ndarray
            Array containing zeros at the locations of the domain 
            not to be considered. 

        Returns
        -------
        WS_array : ndarray
            The interpolated WS over the domain, zero where the mask is zero
        """
        xe, ye = np.meshgrid(np.arange(np.shape(mask)[1]),
                             np.arange(np.shape(mask)[0]))
        f_ws = interp.NearestNDInterpolator(self.get_all_xy(),
                                            self.get_all_WS())
        return f_ws((xe.ravel(), ye.ravel())).reshape(np.shape(mask))

    def interp_WS(self, mask):
        """Interpolates the windows sizes onto a domain with the same dimensions
        as the mask.       
        Assumes input is a grid with uniform spacing in each direction equal
        to the spacing between the first and second points in each direction.
        interpolation is linear. 
        Extrapolation is performed via nearest neighbour

        Arguments:
            mask {ndarray} -- Array containing zeros at the locations of the 
                              domain not to be considered. 

        Returns:
            WS {ndarray} -- the interpolated WS over the domain. 
                            zero where the mask is zero
        """

        # the y value will be all zeros for each row, with a jump equal to the
        # vertical spacing each row.
        y_size = int([i for i, x in enumerate(np.diff(self.y)) if x != 0][0]
                     + 1)
        x_size = int(np.size(self.x) / y_size)

        # now reshape the vectors
        xv_grid = np.reshape(self.x, (x_size, y_size))
        yv_grid = np.reshape(self.y, (x_size, y_size))
        ws_grid = np.reshape(self.WS, (x_size, y_size))
        # ws_grid = np.nan_to_num(ws_grid, nan=97)

        f_ws_interp = interp.interp2d(xv_grid[0],
                                      yv_grid[:, 0],
                                      ws_grid)

        return f_ws_interp(np.arange(np.shape(mask)[1]),
                           np.arange(np.shape(mask)[0])) * mask

    def extend_convex_hull(self, dim):
        """Extends the convex hull beyond the edge of the domain so that we 
        don't have to extrapolate

        By reflecting in x AND y, we guarantee that the whole domain will be 
        within the convex hull of the new distribution

        Parameters
        ----------
        dim : (int, int)
            Dimensions of the domain in pixels

        Returns
        -------
        ex_points : ndarray
            The locations [x, y] of the points added beyond the domain
        ex_vals : ndarray
            The reflected values beyond the domain
        """

        ex_points, ex_vals = np.empty((0, 2)), np.empty((0, 2))
        vor = Voronoi(self.get_all_xy())
        uv = self.get_all_uv()

        # loop over regions relating to each point
        # ind is the index of the region
        # ii is the index of the point
        for ii, ind in enumerate(vor.point_region):
            region = vor.regions[ind]
            # region contains the indices of the vertices forming the region
            # if the region is open, it will contain a -1
            if -1 in region:
                # if the region is open, then we need to reflect the current
                # point in both x and y. The current point has index ii
                x, y = vor.points[ii, :]

                # reflect in x
                if x <= dim[1]/2:
                    ex_points = np.append(ex_points, [[-x, y]], axis=0)
                else:
                    ex_points = np.append(ex_points, [[2*dim[1] - x, y]],
                                          axis=0)
                ex_vals = np.append(ex_vals, [uv[ii, :]], axis=0)

                # reflect in y
                if y <= dim[0]/2:
                    ex_points = np.append(ex_points, [[x, -y]], axis=0)
                else:
                    ex_points = np.append(ex_points, [[x, 2*dim[0] - y]],
                                          axis=0)
                ex_vals = np.append(ex_vals, [uv[ii, :]], axis=0)

        return ex_points, ex_vals


def interp_cub_unstruc(xy, f, eval_dim, eval_h=1, extend_hull=False):
    """Interpolates unstructured sample data onto a structured grid

    Parameters
    ----------
    xy : N_in x 2 array
        list of coordinates of all sample locations
    f : N_in x ndim ndarray
        The values at each of the locations in xy.
    eval_dim : tuple
        The dimensions of the domain to interpolate to. (height, width)
    eval_h : int
        The spacing between evaluation locations in the evaluation grid. 
        Useful for reducing the computational cost of evaluation.
    extend_hull : Bool
        Indicates whether the convex hull should be reflected in the evaluation
        domain first. 
        If False, values outside the convex hull (but within the eval domain)
        will become NaN
        If True, values on the convex hull of the input locations are reflected
        in the domain boundary, effectively resulting in a 'constant' 
        extrapolation. 
        Note that the extrapolation in this case is NOT strictly constant, since
        the cubic interpolant will produce some oscilations here.

    Returns
    -------
    f_int : list of ndarrays
        Returns the interpolated values as an ndarray per ndim in f
    """

    # extend hull if needed
    if extend_hull is True:
        ex_points, ex_vals = extend_convex_hull(xy, f, eval_dim)
        _xy = np.append(xy, ex_points, axis=0)
        _f = np.append(f, ex_vals, axis=0)

    else:
        _xy, _f = xy, f

    if _f.ndim == 1:
        _f = _f[:, np.newaxis]

    # interpolate each 'value' list in f
    _f_out = []
    for _fi in _f.T:
        _f_out.append(interp.CloughTocher2DInterpolator(_xy, _fi.T))

    xe, ye = np.meshgrid(np.arange(0, eval_dim[1], eval_h),
                         np.arange(0, eval_dim[0], eval_h))
    eval_coord_list = np.array([xe.ravel(), ye.ravel()]).T

    # evaluate each value list
    f_out = []
    for _fi in _f_out:
        f_out.append(_fi(eval_coord_list).reshape(np.shape(xe)))

    return f_out


def extend_convex_hull(xy, f, dim):
    """Extends the convex hull by reflecting points in the domain boundaries

    Parameters
    ----------
    xy : N_in x 2 array
        list of coordinates of all sample locations
    f : N_in x ndim ndarray
        The values at each of the locations in xy.
    dim : tuple
        The dimensions of the domain to extend beyond. (height, width)

    Returns
    -------
    ex_points : N_out x 2 array
        list of additional points outside of the domain.
        Does not include the input locations
    ex_vals : ndarray
        Array of reflected values.
        Has shape of N_out x ndim
    """

    try:
        ndim = np.shape(f)[1]
    except IndexError:
        ndim = 1
        f = f[:, np.newaxis]
    ex_points, ex_vals = np.empty((0, 2)), np.empty((0, ndim))
    vor = Voronoi(xy)

    # loop over regions relating to each point
    # ind is the index of the region
    # ii is the index of the point
    for ii, ind in enumerate(vor.point_region):
        region = vor.regions[ind]
        # region contains the indices of the vertices forming the region
        # if the region is open, it will contain a -1
        if -1 in region:
            # if the region is open, then we need to reflect the current
            # point in both x and y. The current point has index ii
            x, y = vor.points[ii, :]

            # reflect in x
            if x <= dim[1]/2:
                ex_points = np.append(ex_points, [[-x, y]], axis=0)
            else:
                ex_points = np.append(ex_points, [[2*dim[1] - x, y]],
                                      axis=0)
            ex_vals = np.append(ex_vals, [f[ii, :]], axis=0)

            # reflect in y
            if y <= dim[0]/2:
                ex_points = np.append(ex_points, [[x, -y]], axis=0)
            else:
                ex_points = np.append(ex_points, [[x, 2*dim[0] - y]],
                                      axis=0)
            ex_vals = np.append(ex_vals, [f[ii, :]], axis=0)

    if ndim == 1:
        ex_vals = ex_vals[:, 0]

    return ex_points, ex_vals


def interp_disp_structured(x, y, u, v, eval_dim, method):
    """Interpolates displacement values onto a pixel grid

    Extrapolates values linearly beyond the convex hull of x,y 

    Parameters
    ----------
    x : 2d array
        Array of x coordinates of samples
    y : 2d array
        Array of y coordinates of samples
    u : 2d array
        Array of horizontal displacements
    v : 2d array
        Array of vertical displacements
    eval_dim : (2 x 1) int
        Tuple of dimensions indicating the number of pixels in each direction
        that the displacement should be interpolated
        Values beyond the convex hull of x, y are linearly extrapolated
    method: str
        Indicates how the values should be interpolated
        Options: "struc_lin", "struc_cub"

    Returns
    -------
    u, v : ndarray
        Pixelwise displacement values
    """

    # now we need to handle extrapolation
    x, y, u, v = (utilities.lin_extrap_edges(x),
                  utilities.lin_extrap_edges(y),
                  utilities.lin_extrap_edges(u),
                  utilities.lin_extrap_edges(v), )

    # calculate evaluation range
    xe = np.arange(eval_dim[1])
    ye = np.arange(eval_dim[0])

    if method == "struc_lin":
        # interpolate using scipy
        f_u = interp.interp2d(x[0, :], y[:, 0], u, kind='linear')
        f_v = interp.interp2d(x[0, :], y[:, 0], v, kind='linear')
        u_int = f_u(xe, ye)
        v_int = f_v(xe, ye)
    elif method == "struc_cub":
        # interpolate using scipy
        f_u = interp.interp2d(x[0, :], y[:, 0], u, kind='cubic')
        f_v = interp.interp2d(x[0, :], y[:, 0], v, kind='cubic')
        u_int = f_u(xe, ye)
        v_int = f_v(xe, ye)
    else:
        raise ValueError("Bad method")

    return u_int, v_int


def NMT_detection(u, v, nb_ind, eps=0.1):
    """
    Detects outliers according to the normalised median threshold test of
    Westerweel and Scarano
    Returns the norm value

    Args:
        u (list, float): list of the u displacement values
        v (list, float): list of the v displacement values
        nb_ind (ndarray, int): list of neighbour indices for each location
        eps (float, optional): background noise level, in px
        thr (int, optional): threshold for an outlier
    """

    # calculate the median of all neighbours
    # nb_ind is (N, 9), u/v_med is (N, 1)
    u_med, v_med = (np.nanmedian(u[nb_ind[:, 1:]], axis=1),
                    np.nanmedian(v[nb_ind[:, 1:]], axis=1))

    # fluctuations
    # u_fluct_all is (N, 9)
    u_fluct, v_fluct = (u[nb_ind] - u_med[:, np.newaxis],
                        v[nb_ind] - v_med[:, np.newaxis])

    # residual is (N, 1)
    resu, resv = (np.nanmedian(np.abs(u_fluct[:, 1:]), axis=1) + eps,
                  np.nanmedian(np.abs(v_fluct[:, 1:]), axis=1) + eps)

    u_norm, v_norm = (np.abs(u_fluct[:, 0] / resu),
                      np.abs(v_fluct[:, 0] / resv))
    norm = np.sqrt(u_norm**2 + v_norm**2)

    return norm


def NMT_idw_detection(u, v, nb_ind, nb_dist, eps=0.1):
    """
    Detects outliers according to the normalised median threshold test of
    Duncan, Dabiri, and Hove.
    Returns the norm value

    Parameters
    ----------
    u : list, float
        list of the u displacement values
    v : list, float
        list of the v displacement values
    nb_ind : ndarray, int
        list of neighbour indices for each location
    nb_dist : ndarray, float
        list of distances to each neighbour defined nb_ind
    eps : float, optional
        background noise level, in px
    thr : int, optional
        threshold for an outlier
    """

    # calculate the denominator term, the median of neighbour distances plus ea
    md_neighb = np.nanmedian(nb_dist[:, 1:], axis=1)
    ea = 0.5*(-md_neighb + np.sqrt(md_neighb**2 + 4*eps))

    # calculate the median of all neighbours
    di_ea = (nb_dist[:, 1:] + ea[:, np.newaxis])
    # nb_ind is (N, 9), u/v_med is (N, 1)
    # bf_u = (nb_dist[:, 1:] + ea)[:, np.newaxis]
    u_med, v_med = (np.nanmedian(u[nb_ind[:, 1:]]/di_ea, axis=1),
                    np.nanmedian(v[nb_ind[:, 1:]]/di_ea, axis=1))

    u_0_norm, v_0_norm = (u[nb_ind[:, 0]] / (md_neighb + ea),
                          v[nb_ind[:, 0]] / (md_neighb + ea))

    # fluctuations
    # u_fluct_all is (N, 9)
    u_fluct, v_fluct = (u[nb_ind[:, 1:]]/di_ea - u_med[:, np.newaxis],
                        v[nb_ind[:, 1:]]/di_ea - v_med[:, np.newaxis])

    # residual is (N, 1)
    resu, resv = (np.nanmedian(np.abs(u_fluct), axis=1) + ea,
                  np.nanmedian(np.abs(v_fluct), axis=1) + ea)

    u_norm, v_norm = (np.abs(u_0_norm-u_med) / resu,
                      np.abs(v_0_norm-v_med) / resv)
    norm = np.sqrt(u_norm**2 + v_norm**2)

    return norm

def outlier_replacement(flag, u, v, nb_ind):
    """
    Replaces all outliers with the median of neighbouring valid vectors
    If all neighbouring vectors are invalid then the replaced value is 0

    Args:
        flag (boolean): flag indicating which vectors are outliers
        u (list, float): list of the u displacement values, including outliers
        v (list, float): list of the v displacement values, including outliers
        nb_ind (ndarry, int): list of neighbour indices for each location
    """

    for ii in range(np.shape(flag)[0]):
        if flag[ii]:  # if outlier
            # get the neighbouring values, including outliers at this point
            u_neigh, v_neigh = u[nb_ind[ii, 1:]], v[nb_ind[ii, 1:]]

            # remove outliers from neighbour list
            u_neigh = u_neigh[~flag[nb_ind[ii, 1:]]]
            v_neigh = v_neigh[~flag[nb_ind[ii, 1:]]]

            # calculate replacement, unless all neighbours are outliers
            if len(u_neigh) > 0:
                u[ii], v[ii] = np.median(u_neigh), np.median(v_neigh)
            else:
                u[ii], v[ii] = 0, 0
        else:
            continue

    return u, v


class Disk():
    """
    Class for adaptive incremental stippling to provide functionality for
    determining whether a disk is valid or not
    """

    def __init__(self, x, y, r):
        """
        Initialise a Disk with a specific location and radius

        Args:
            x (int): Horizontal location in the domain in pixels
            y (int): Vertical location in the domain in pixels
            r (float): Radius of the disk
        """

        self.x, self.y = x, y
        self.r = r
        # a list of arcs
        self.avail_range = [[0, 2*np.pi]]

    def is_range_available(self):
        """
        Returns whether or not there is space around the perimeter of the disk
        to place another disk
        """
        return len(self.avail_range) > 0

    def random_avail_angle(self):
        """
        Returns a random angle from within the range of available arcs, as
        defined by self.avail_range
        """
        rand_arc = random.choice(self.avail_range)
        return np.random.uniform(rand_arc[0], rand_arc[1])

    def overlaps_in_buffer(self, buffer, bf_refine):
        """
        Determines whether the current disk would overlap any existing disk in
        the buffer.

        To improve the accuracy of whether a disk overlaps of not, the buffer
        may have been 'refined', that is, multiple pixels in the buffer_array
        may refer to a single pixel in the 'actual' buffer.

        Args:
            buffer (ndarray): boolean array indicating where disks already exist
            bf_refine (int): Ratio of number of pixels in the buffer array,
                             to the number of pixels in the domain. Effectively,
                             shape(buffer) = bf_refine*(dim_y, dim_x)
        """

        if int(bf_refine) != bf_refine:
            raise ValueError("bf_refine must be an integer")

        # get the properties of the disk in the buffer array
        x_bf, y_bf = self.x*bf_refine, self.y*bf_refine
        r_bf = self.r*bf_refine

        n_rows_bf, n_cols_bf = np.shape(buffer)*bf_refine

        # get the coordinates of the square of size 2r x 2r
        l = int(max(0, np.floor(x_bf - r_bf)))
        r = int(min(n_cols_bf, np.ceil(x_bf + r_bf) + 1))
        b = int(max(0, np.floor(y_bf - r_bf)))
        t = int(min(n_rows_bf, np.ceil(y_bf + r_bf) + 1))

        # select the points in buffer which are within the radius of the disk
        # if any of these points are unity, then the disk overlaps
        xx, yy = (np.arange(l, r)-x_bf)**2, (np.arange(b, t)-y_bf)**2
        return np.any(buffer[b:t, l:r][xx + yy[:, np.newaxis] <= r_bf**2])

    def update_available_range(self, other_disk):
        """
        Adjusts the available range to reflect the presence of a new disk

        Args:
            other_disk (Disk): The new disk being added
        """

        # work out angles
        cos_beta = max(-1, min(1, (self.r + other_disk.r)/(4*other_disk.r)))
        beta = np.arccos(cos_beta)

        if beta < 1e-3:
            # if beta is too small, then break out early.
            return

        dx, dy = other_disk.x - self.x, other_disk.y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/dist, dy/dist
        alpha = np.arctan2(dy, dx)

        # define 2 pi for simplicity later
        two_pi = np.pi*2

        if alpha < 0:
            alpha += two_pi

        clippers = []
        _from, to = alpha - beta, alpha + beta

        if _from >= 0 and to <= two_pi:
            # simple case, from and to is entirely within 0 and 2pi
            clippers.append([_from, to])
        else:
            # clipper crosses 0, so need to split into two clippings
            if _from < 0:
                if to > 0:
                    # if to == 2pi, then we only need one clipper
                    clippers.append([0, to])
                clippers.append([_from + two_pi, two_pi])

            if to > two_pi:
                if _from < two_pi:
                    # see above comment
                    clippers.append([_from, two_pi])
                clippers.append([0, to - two_pi])

        remaining = []
        for clipper in clippers:
            for arc in self.avail_range:
                if arc[0] >= clipper[0] and arc[1] <= clipper[1]:
                    # arc is completely culled, remove
                    continue
                elif arc[1] < clipper[0] or arc[0] > clipper[1]:
                    # untouched
                    remaining.append(arc)
                elif (arc[0] <= clipper[0] and
                      arc[1] >= clipper[0] and
                      arc[1] <= clipper[1]):
                    # if the clipper starts within this arc, and the arc ends
                    # within the clipper
                    _from, to = arc[0], clipper[0]
                    remaining.append([_from, to])
                elif (arc[0] >= clipper[0] and
                      arc[0] <= clipper[1] and
                      arc[1] >= clipper[1]):
                    # if the arc starts within the clipper, and the arc ends
                    # outside the clipper
                    _from, to = clipper[1], arc[1]
                    remaining.append([_from, to])
                else:
                    # clipper is entirely in the arc, split
                    _from, to = arc[0], clipper[0]
                    remaining.append([_from, to])
                    _from, to = clipper[1], arc[1]
                    remaining.append([_from, to])

        self.avail_range = remaining

    def approximate_local_density(self, pdf_sat, mask_sat):
        """
        Returns an estimate of the local pdf density around the disk.

        The pdf is summed in a square centred on the disk with linear edges
        equal the the Disk's diameter

        Args:
            pdf_sat (SummedAreaTable): The pdf as a summed area table
            mask_sat (SummedAreaTable): The mask as a summed area table
        """

        if self.r < 1:
            density = pdf_sat.get_area_sum(self.x, self.x, self.y, self.y)
        else:
            l = int(max(0, np.floor(self.x - self.r)))
            r = int(min(pdf_sat.img_dim[1], np.ceil(self.x + self.r)))
            b = int(max(0, np.floor(self.y - self.r)))
            t = int(min(pdf_sat.img_dim[0], np.ceil(self.y + self.r)))

            pdf_val = pdf_sat.get_area_sum(l, r, b, t)
            mask_val = mask_sat.get_area_sum(l, r, b, t)
            # scale according to how much of the area was masked
            # area of square / non-masked area
            # also scale according to area of cirle in area of square
            # density = pdf_val * (4*self.r**2) / mask_val) * np.pi / 4
            density = pdf_val * self.r**2 * np.pi / mask_val

        if density == 0:
            density += 1e-6

        return density

    def draw_onto_buffer(self, buffer, bf_refine):
        """
        Draws a binary representation of the disk onto the buffer.

        Pixels which lie within disk.r of the disk centre will be set to 1 in the
        buffer

        Args:
            buffer (ndarray): Current disk buffer with ones indicating the
                              location of existing disks.
                              Will be modified in place
            disk (Disk): The disk to add to the buffer
            bf_refine (int, optional): Ratio of number of pixels in the disk
                                       buffer to the number of pixels in the
                                       domain. Allows for more precise
                                       evaluation of disk overlap at the
                                       expense of computational cost.
                                       Defaults to 1.
        """

        if int(bf_refine) != bf_refine:
            raise ValueError("bf_refine must be an integer")

        # get the properties of the disk in the buffer array
        x_bf, y_bf = self.x*bf_refine, self.y*bf_refine
        r_bf = self.r*bf_refine

        n_rows_bf, n_cols_bf = np.shape(buffer)*bf_refine

        # get the coordinates of the square of size 2r x 2r
        l = int(max(0, np.floor(x_bf - r_bf)))
        r = int(min(n_cols_bf, np.ceil(x_bf + r_bf)))
        b = int(max(0, np.floor(y_bf - r_bf)))
        t = int(min(n_rows_bf, np.ceil(y_bf + r_bf)))

        # get dist squared to center
        xx, yy = (np.arange(l, r)-x_bf)**2, (np.arange(b, t)-y_bf)**2

        new_disk = np.zeros((t-b, r-l))
        new_disk[xx + yy[:, np.newaxis] <= r_bf**2] = 1

        buffer[b:t, l:r] = new_disk

    def change_radius(self, Q, angle, K, pdf_sat, mask_sat):
        """
        Changes the disks radius such that it contains the desired amount of
        underlying pdf

        Args:
            Q (Disk): The central disk. The current disk will be varied in size
                      while maintaining contact with the central disk
            angle (float): Random angle along which the new disk is varied in
                           size. In radians
            K (float): The amount of the underlying pdf to contain
            pdf_sat (SummedAreaTable): SAT representing the pdf function
        """
        dens = self.approximate_local_density(pdf_sat, mask_sat)
        radius_ratio = np.sqrt(K / dens)
        eps, count, limit = 0.001, 0, 13
        n_rows, n_cols = pdf_sat.img_dim

        while abs(radius_ratio-1) > eps and count < limit:
            rn = max(0.5, self.r*radius_ratio)
            self.x = min(max(0, round(Q.x + (rn+Q.r)*np.cos(angle))), n_cols-1)
            self.y = min(max(0, round(Q.y + (rn+Q.r)*np.sin(angle))), n_rows-1)
            self.r = rn
            dens = self.approximate_local_density(pdf_sat, mask_sat)
            radius_ratio = np.sqrt(K/dens)
            count += 1


def AIS_py(pdf, mask, n_points, bf_refine=1, ex_points=None):
    """
    Distributes approximately n_points samples with a local density similar to
    that described by the pdf. Points will not be placed in the masked region.

    Args:
        pdf (ndarray): Probability density function describing the local target
                       target density of the sample distribution
        mask (ndarray): Binary mask indicating where points should not be placed
                        A mask value of 0 indicates that points should ne be
                        placed here.
                        Must have the same dimensions as the input pdf
        n_points (int): Approximate number of samples to place in the domain
        bf_refine (int, optional): Ratio of number of pixels in the disk
                                   buffer to the number of pixels in the domain.
                                   Allows for more precise evaluation of disk
                                   overlap at the expense of computational cost.
                                   Defaults to 1.
        ex_points (2D list int, optional): List of coordinates to seed the
                                           distribution process with. Should
                                           be a 2D array_like object (i.e. a
                                           list or tuple of lists or tuples
                                           containing the x and y location,
                                           alternatively, a 2D numpy array).
                                           If no seed points are given, the
                                           seed point is randomly chosen.
                                           Defaults to None.
    """

    if not np.any(mask):
        raise ValueError("Mask can't be all 0")

    n_rows, n_cols = np.shape(pdf)

    # initialise the queue, output list, and the disk buffer
    q, out_list, disk_buf = [], [], np.zeros((n_rows, n_cols)*bf_refine)

    pdf *= mask
    # create summed area table for pdf
    pdf_sat = utilities.SummedAreaTable(pdf)
    mask_sat = utilities.SummedAreaTable(mask)

    # determine the initial estimate for r1
    K = pdf_sat.get_total_sum() / n_points
    r1 = np.sqrt(np.size(pdf)/(np.pi*n_points))

    # initialise AIS
    if ex_points is None:
        # create a seed point
        while True:
            xr, yr = np.random.randint(0, n_cols), np.random.randint(0, n_rows)
            if mask[yr, xr]:  # if not masked
                break

        D = Disk(xr, yr, r1)
        dens = D.approximate_local_density(pdf_sat, mask_sat)
        ratioR = np.sqrt(K/dens)
        eps, count, limit = 0.001, 0, 20

        while np.abs(ratioR-1) > eps:
            D.r = max(0.5, D.r*ratioR)
            dens = D.approximate_local_density(pdf_sat, mask_sat)
            ratioR = np.sqrt(K/dens)
            count += 1
            if count > limit:
                break

        # Add the disk to the queue, and add it to the final points list
        q.append(D)
        out_list.append([D.x, D.y])

        # draw the disk onto the buffer, note that disk_buf is updated inplace
        D.draw_onto_buffer(disk_buf, bf_refine)

    else:
        for point in ex_points:
            D = Disk(point[0], point[1], r1*0.25)
            q.append(D)
            D.draw_onto_buffer(disk_buf, bf_refine)
            out_list.append([D.x, D.y])

    # main AIS loop
    while len(q) > 0:
        # get the last added disk
        Q = q.pop()
        attempts, limit = 0, 20

        while Q.is_range_available() and attempts < limit:
            attempts += 1
            # create disk at random angle with init radius r1
            # checking it isn't masked or out of the domain
            alpha = Q.random_avail_angle()
            xn = int(np.round(Q.x + (Q.r+r1)*np.cos(alpha)))
            yn = int(np.round(Q.y + (Q.r+r1)*np.sin(alpha)))
            if (yn >= n_rows or xn >= n_cols or
                yn < 0 or xn < 0 or
                    mask[yn, xn] == 0):
                continue
            else:
                P = Disk(xn, yn, r1)

            P.change_radius(Q, alpha, K, pdf_sat, mask_sat)

            if ((P.x >= 0 and P.x < n_cols) and (P.y >= 0 and P.y < n_rows)
                    and not P.overlaps_in_buffer(disk_buf, bf_refine)):
                # point is valid, accept it
                q.append(P)
                out_list.append([P.x, P.y])

                P.draw_onto_buffer(disk_buf, bf_refine)
                Q.update_available_range(P)

    return out_list


def AIS(pdf, mask, n_points, bf_refine=1, ex_points=None):
    """
    Distributes approximately n_points samples with a local density similar to
    that described by the pdf. Points will not be placed in the masked region.

    Args:
        pdf (ndarray): Probability density function describing the local target
                       target density of the sample distribution
        mask (ndarray): Binary mask indicating where points should not be placed
                        A mask value of 0 indicates that points should ne be
                        placed here.
                        Must have the same dimensions as the input pdf
        n_points (int): Approximate number of samples to place in the domain
        bf_refine (int, optional): Ratio of number of pixels in the disk
                                   buffer to the number of pixels in the domain.
                                   Allows for more precise evaluation of disk
                                   overlap at the expense of computational cost.
                                   Defaults to 1.
        ex_points (2D list int, optional): List of coordinates to seed the
                                           distribution process with. Should
                                           be a 2D array_like object (i.e. a
                                           list or tuple of lists or tuples
                                           containing the x and y location,
                                           alternatively, a 2D numpy array).
                                           If no seed points are given, the
                                           seed point is randomly chosen.
                                           Defaults to None.
    Returns:
        out_list (N-by-2 ndarray): The x and y coordinates of the distribution's 
                                samples
    """

    dim = np.shape(pdf)
    mb = sk_seg.find_boundaries(mask, mode='inner')
    # get the locations in terms of [y, x], and then switch to [x, y]
    mask_bounds = np.argwhere(mb == True)[:, [1, 0]]
    # get the locations of the image boundaries
    bottom = [[x, 0] for x in range(dim[1])]
    top = [[x, dim[0]-1] for x in range(dim[1])]
    left = [[0, y] for y in range(dim[0])]
    right = [[dim[1]-1, y] for y in range(dim[0])]
    all_locs = np.concatenate((mask_bounds, left, right, top, bottom), axis=0)
    # remove locations IN the mask
    borders = all_locs[mask[all_locs[:, 1], all_locs[:, 0]] == True]
    n_borders = np.shape(borders)[0]

    # calculate the value of r1 - the initial disk size for AIS
    r1 = np.sqrt(np.sum(mask, axis=None)/(np.pi * n_points))
    # calculate how many samples to place around the borders
    n = int(np.maximum(2, np.round(n_borders/(3*2*r1), 0)))

    # place some points around these borders
    ind = pdf_transform_1d(np.ones((n_borders, 1)), n)

    ex_points = np.array(borders[ind, :]).astype(np.float64)

    return np.array(ais_module.AIS(pdf.astype(np.float64),
                                   mask.astype(np.float64),
                                   n_points, bf_refine, ex_points))


def pdf_transform_1d(pdf, n_points):
    """Distributes samples according to the inverse of a one dimensional pdf

    Parameters
    ----------
    pdf : list or Nx1 ndarray
        probability density function indicating how dense the output locations
        should be
    n_points : int
        How many points should be distributed along the line
    """

    # calculate the normalised cdf
    cdf = np.cumsum(pdf) / np.sum(pdf)
    inv_y = np.linspace(np.min(cdf)+1e-9, 1, n_points)
    x_out = interp.interp1d(cdf, np.arange(len(pdf)))
    return x_out(inv_y).astype(int)


def pdf_transform(pdf, mask, n_points, smoothing=True):
    """
    Create a 2D distribution of points using the pdf transform method from
    Raf

    Args:
        pdf (ndarray): The probability density function which defines the local
                       relative density
        mask (ndarray): Binary mask indicating whether a region should or should
                        not have a point placed in
        n_points (int): Approximate number of points to place in the domain
        smoothing (bool): Whether to apply laplacian smoothing or not  

    """

    # get the possible locations of the windows in x and y
    dim = np.shape(pdf)
    y_pos, x_pos = np.arange(dim[0]), np.arange(dim[1])
    x, y = [], []

    # normalise the pdf between 0 and 1
    pdf /= np.sum(pdf)

    # get the sum and the normalised cumulative sum of the pdf along each row
    m = np.sum(pdf, axis=1)
    M = np.cumsum(m) / np.sum(m)

    # now evenly space points in this range
    prob_y = np.linspace(np.min(M)+1e-9, np.max(M)-1e-9, n_points)
    q_y = interp.interp1d(M, y_pos)(prob_y)
    m_q = interp.interp1d(y_pos, m)(q_y)

    # Calculate x-positions
    for n in range(n_points):
        p_xq = pdf[int(np.ceil(q_y[n])), :]
        c_xq = p_xq / m_q[n]
        C = np.cumsum(c_xq)/np.sum(c_xq)
        prob_x = min(C) + np.random.uniform()*(max(C)-min(C))
        q_x = min(dim[1], max(1, interp.interp1d(C, x_pos)(prob_x)))

        x.append(min(dim[1]-1, max(0, np.round(q_x))))
        y.append(min(dim[0]-1, max(0, np.round(q_y[n]))))

    out_list = np.unique(np.array([x, y]), axis=1).T.astype(int)

    if smoothing:
        out_list = laplacian_smoothing(out_list)

    mask_val = mask[out_list[:, 0], out_list[:, 1]]
    out_list = out_list[mask_val == 1]

    return out_list


def laplacian_smoothing(coordinates):
    """
    Smooths the input coordinates using laplacian smoothing

    Args:
        coordinates (ndarray): Contains a list_like of coordinate locations
    """

    # calculate the delaunay triangulation
    tri = Delaunay(coordinates)

    # get list of neighbouring points
    indices, indptr = tri.vertex_neighbor_vertices

    # initialise output list
    xy_out = np.empty_like(coordinates)

    hull = ConvexHull(coordinates)
    ind = np.unique(hull.simplices)

    for k in range(len(indices)-1):
        if k in ind:
            xy_out[k, :] = coordinates[k, :]
        else:
            xy_neighbs = coordinates[indptr[indices[k]:indices[k+1]], :]
            xy_out[k, :] = np.average(xy_neighbs, axis=0)

    return xy_out


if __name__ == '__main__':

    # # create meshgrid
    # strt, fin, step = 1, 31 + 1, 10
    # x = np.arange(strt, fin, step)
    # y = np.arange(strt, fin + 5, step / 2)
    # xx, yy = np.meshgrid(x, y)
    # U = np.exp(-(2 * xx / 101)**2 - (yy / (2 * 101))**2)

    # # interpolate U on x and y
    # vals = interp.interp2d(xx[0, :], yy[:, 0], U)

    # xe = np.arange(strt, fin, 1)
    # ye = np.arange(strt, fin, 1)
    # xxe, yye = np.meshgrid(xe, ye)
    # u_int = vals(xxe.ravel(), yye.ravel())

    # fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
    # ax[0].plot_wireframe(xx, yy, U, color='b')
    # ax[0].plot_wireframe(yye.ravel(), xxe.ravel(), u_int, color='r')

    # # ax[1].plot(x, U[0, :], 'ro-', xe, u_int[0, :], 'b-')
    # plt.show()

    import cProfile
    import pstats

    def run_code(func, pdf, mask, n_points, n_repeats):
        for i in range(n_repeats):
            func(pdf, mask, n_points)

    pdf = np.arange(1000)[:, np.newaxis] * np.ones((1000, 1000))
    pdf /= np.sum(pdf)
    mask = np.ones((1000, 1000))
    n_repeats, n_points = 3, 1000

    cProfile.runctx('run_code(ais_module.AIS, pdf, mask, n_points, n_repeats)',
                    globals(), locals(), 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats('tottime').print_stats(25)

    cProfile.runctx('run_code(AIS, pdf, mask, n_points, n_repeats)',
                    globals(), locals(), 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats('tottime').print_stats(25)
    # start = time.time()
    # for i in range(n_repeats):
    #     points = ais_module.AIS(pdf, mask, n_points=1000)
    # end = time.time()
    # print((end-start)/n_repeats)

    # start = time.time()
    # for i in range(n_repeats):
    #     points = AIS(pdf, mask, n_points=1000)
    # end = time.time()
    # print((end-start)/n_repeats)
