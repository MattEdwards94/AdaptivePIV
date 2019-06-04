import corr_window
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors


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

    def n_windows(self):
        """
        Returns the number of windows currently stored in the distribution
        """
        return len(self.windows)

    def get_values(self, prop):
        """
        Returns a list of property values from the list of CorrWindows
        corresponding to the requested property 'prop'

        Args:
            prop (str): The property of self.windows to retrieve

        Returns:
            list: list of properties 'prop' from self.windows

        Example:
            >>> import corr_window
            >>> x = [10, 20, 30]
            >>> y = [15, 25, 35]
            >>> WS = [31, 41, 51]
            >>> cwList = []
            >>> for i in range(3)
            >>>     cwList.append(corr_window.CorrWindow(x[i], y[i], WS[i]))
            >>> dist = Distribution(cwList)
            >>> x_vals = dist.get_values("x")
            >>> print(x_vals)
            ... [10, 20, 30]
        """
        return [cw.__dict__[prop] for cw in self.windows]

    def set_values(self, prop, values):
        """
        Set's the values of the CorrWindow objects properties specified by
        'prop' to be 'values'

        Args:
            prop (str): The property of self.windows to update
            values (TYPE): Description
        """

        if not (np.shape(values)[0] == self.n_windows()):
            raise ValueError('values must be a column vector with the same \
                number of entries as the number of windows')

        if prop not in self.windows[0].__dict__:
            raise KeyError('Property ', prop, ' does not exist')

        for cw, ii in zip(self.windows, range(self.n_windows())):
            cw.__dict__[prop] = values[ii]

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
        x = self.get_values('x')
        y = self.get_values('y')
        u = self.get_values('u')
        v = self.get_values('v')
        xy = np.transpose(np.array([x, y]))
        nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(xy)
        nb_dist, nb_ind = nbrs.kneighbors(xy)

        norm = NMT_detection(u, v, nb_ind, eps)
        flag = norm > threshold

        # replacement
        u, v = outlier_replacement(flag, u, v, nb_ind)

        # update values in CorrWindow objects
        for (cw, outlier, u_i, v_i) in zip(self.windows, flag, u, v):
            if outlier == 1:
                cw.u_pre_validation, cw.v_pre_validation = cw.u, cw.v
                cw.u, cw.v = u_i, v_i
                cw.flag = outlier

    def interpolate_onto_densepredictor(self, method, out_dim):

        # check the input method
        acc_options = ["str_lin", "str_cub"]
        if method not in acc_options:
            raise ValueError("Method not handled")


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

    norm = np.empty(np.shape(u))
    for row in nb_ind:
        # get the median of u and v from the neighbours
        # ignore first element (itself)
        u_nb, v_nb = u[row[1:]], v[row[1:]]
        u_med, v_med = np.median(u_nb), np.median(v_nb)

        # difference of all vectors to median
        u_fluct, v_fluct = u[row] - u_med, v[row] - v_med

        # difference of central vector
        u_ctr_fluct, v_ctr_fluct = u_fluct[0], v_fluct[0]

        # calculate norm
        u_norm, v_norm = (np.abs(u_ctr_fluct /
                                 (np.median(np.abs(u_fluct[1:])) + eps)),
                          np.abs(v_ctr_fluct /
                                 (np.median(np.abs(v_fluct[1:])) + eps)), )
        norm[row[0]] = np.sqrt(u_norm**2 + v_norm**2)

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
    # create long list of corrWindows
    cwList = []
    for i in range(10):
        cwList.append(corr_window.CorrWindow(i, 2 * i, 31))

    dist = Distribution(cwList)
    print(dist.get_values('x'))
    # print(dist.get_values('WrongKey'))

    start = time.time()
    for i in range(100):
        dist.get_values('x')
    print("plain list", time.time() - start)
