import numpy as np
from sklearn.neighbors import NearestNeighbors
import PIV.utilities as utilities
import PIV.corr_window as corr_window
from scipy import interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    def from_locations(x, y, WS):
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
            if cw.is_masked is False:
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
            if cw.is_masked is False:
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
            if cw.is_masked is False:
                out_list.append(cw.WS)

        return np.array(out_list)

    def get_flag_values(self):
        """
        Returns:
            Returns an (N, 1) array of flag values
        """
        return np.array([cw.flag for cw in self.windows])

    def validation_NMT_8NN(self, threshold=2, eps=0.1):
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


        Args:
            threshold (int, optional): The threshold above which the norm
                                       indicates an outlier. Refer to paper
            eps (float, optional): The assumed background noise in px.
                                   Refer to paper
        """

        # detection
        # find neighbours
        xy, uv = self.get_unmasked_xy(), self.get_unmasked_uv()
        u, v = uv[:, 0], uv[:, 1]
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(xy)
        nb_dist, nb_ind = nbrs.kneighbors(xy)

        norm = NMT_detection(u, v, nb_ind, eps)
        flag = norm > threshold
        invalid = np.sum(flag)
        try:
            print(f"  {invalid}/{self.n_windows()} vectors replaced")
        except:
            print(f"  {invalid}/{self.n_windows} vectors replaced")

        # replacement
        u, v = outlier_replacement(flag, u, v, nb_ind)

        # update values in CorrWindow objects
        for (cw, outlier, u_i, v_i) in zip(self.windows, flag, u, v):
            if cw.is_masked is True:
                continue
            if outlier == 1:
                cw.u_pre_validation, cw.v_pre_validation = cw.u, cw.v
                cw.u, cw.v = u_i, v_i
                cw.flag = outlier

        return flag

    def interp_to_densepred(self, method, eval_dim):
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
        acc_options = ["struc_lin", "struc_cub"]
        if method not in acc_options:
            raise ValueError("Method not handled")

        # check the output dimensions - must be positive integers
        for dim in eval_dim:
            if (not np.all(dim == int(dim))) or np.any(dim < 1):
                raise ValueError("Dimensions must be positive integer")

        # reshape the data onto a structured grid
        xy, uv = self.get_all_xy(), self.get_all_uv()
        x, y, u, v = xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1]
        x2d, y2d, u2d, v2d = utilities.auto_reshape(x, y, u, v)

        # now we need to handle extrapolation
        x, y, u, v = (utilities.lin_extrap_edges(x2d),
                      utilities.lin_extrap_edges(y2d),
                      utilities.lin_extrap_edges(u2d),
                      utilities.lin_extrap_edges(v2d), )

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
            cw.correlate(img, dp)

    def plot_locations(self, *args, **kwargs):
        """Plots the locations of all the windows within the distribution
        """
        fig = plt.figure()
        plt.plot(self.x, self.y, *args, **kwargs)
        fig.show()

    def plot_distribution(self):
        fig, ax = plt.subplots()
        xy, uv = self.get_all_xy(), self.get_all_uv()
        ax.quiver(xy[:, 0],
                  xy[:, 1],
                  uv[:, 0],
                  uv[:, 1])
        plt.show()

    def AIW(self, img, dp, step_size=6, SNR_thr=1.3, max_WS=245):
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
            while cw.WS < max_WS:
                cw.correlate(img, dp)
                # now check validity of result
                if cw.SNR == 0:
                    # the location is masked
                    break
                elif (cw.SNR <= SNR_thr) or (cw.disp_mag() > cw.WS*0.25):
                    # if SNR is too low, or the displacement violates 1/4 rule
                    cw.WS += step_size
                    if cw.WS > max_WS:
                        break
                else:
                    # WS is ok
                    break

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


if __name__ == '__main__':

    # create meshgrid
    strt, fin, step = 1, 31 + 1, 10
    x = np.arange(strt, fin, step)
    y = np.arange(strt, fin + 5, step / 2)
    xx, yy = np.meshgrid(x, y)
    U = np.exp(-(2 * xx / 101)**2 - (yy / (2 * 101))**2)

    # interpolate U on x and y
    vals = interp.interp2d(xx[0, :], yy[:, 0], U)

    xe = np.arange(strt, fin, 1)
    ye = np.arange(strt, fin, 1)
    xxe, yye = np.meshgrid(xe, ye)
    u_int = vals(xxe.ravel(), yye.ravel())

    fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
    ax[0].plot_wireframe(xx, yy, U, color='b')
    ax[0].plot_wireframe(yye.ravel(), xxe.ravel(), u_int, color='r')

    # ax[1].plot(x, U[0, :], 'ro-', xe, u_int[0, :], 'b-')
    plt.show()
