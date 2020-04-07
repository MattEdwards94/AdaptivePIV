import numpy as np
import PIV.distribution as distribution
import PIV.utilities as utilities
import math
import PIV.corr_window as corr_window
import PIV.dense_predictor as dense_predictor
import PIV.ensemble_solution as es
import PIV.multiGrid as mg
import PIV
import matplotlib.pyplot as plt
from PIV.utilities import vprint, WS_for_iter

ESSENTIAL, BASIC, TERSE = 1, 2, 3


def adapt_multi_grid(img):
    """Analyses an image using the multi_grid approach
    """

    init_WS = 129
    final_WS = 65

    dp = PIV.DensePredictor(
        np.zeros(img.dim), np.zeros(img.dim), img.mask)

    amg = mg.MultiGrid(img.dim, spacing=64, WS=init_WS)
    print("Grid created")

    # correlate all windows
    print("Correlating windows")
    amg.correlate_all_windows(img, dp)

    print("Validate vectors")
    amg.validation_NMT_8NN()

    print("Interpolating")
    dp = amg.interp_to_densepred()
    dp.mask = img.mask
    dp.apply_mask()

    print("Deforming image")
    img_def = img.deform_image(dp)

    print("Spitting all cells")
    amg.split_all_cells()
    print(amg.grids[1].x_vec)
    print(amg.grids[1].y_vec)
    print("Setting all windows to 65 pixel windows")
    for window in amg.windows:
        window.WS = final_WS

    # correlate all windows
    print("Correlating windows")
    amg.correlate_all_windows(img_def, dp)

    print("Validate vectors")
    amg.validation_NMT_8NN()

    print("Interpolating")
    dp = amg.interp_to_densepred()
    dp.mask = img.mask
    dp.apply_mask()

    return dp, amg


class MultiGridSettings():
    def __init__(self,
                 init_WS=None, final_WS=None,
                 init_N_windows=2500, final_N_windows=10000,
                 n_iter_main=3, n_iter_ref=2,
                 distribution_method='AIS',
                 vec_val='NMT', idw=True,
                 interp='unstruc_cub',
                 part_detect='simple',
                 sd_P_target=20,
                 target_init_NI=20, target_fin_NI=8,
                 verbosity=2):
        """

        Parameters
        ----------
        init_WS : int or str, optional
            Initial window size, if numeric, must be odd and 5 <= init_WS <= 245
            Otheriwse 'auto', where the window size will be calculated using the
            adaptive initial window routine.
            Default 'auto'.
        final_WS : int or str, optional
            Final window size, must be odd and 5 <= final_WS <= 245
            Otheriwse 'auto', where the window size will be calculated
            according to the seeding density
            Default 'auto'
        init_N_windows : int, optional
            Initial number of windows to be used in the first iteration.
            The number of windows will increase linearly from init to final
            (see below) over the first n_iter_main iterations.
            Default 2,500
        final_N_windows : int, optional
            Final number of windows to have in the analysis after n_iter_main
            iterations.
            Can be more or less than init_N_windows.
            Default 10,000
        n_iter_main : int, optional
            Number of main iterations, wherein the WS and spacing will reduce
            from init_WS to final_WS Must be 1 <= n_iter_main <= 10
            If the number of main iterations is 1 then the final_WS and
            final_N_windows is ignored
            Default 3
        n_iter_ref : int, optional
            Number of refinement iterations, where the WS and locations remain
            fixed, however, subsequent iterations are performed to improve
            the solution. Must be 0 <= n_iter_ref <= 10
            Default 2
        vec_val : str, optional
            Type of vector validation to perform.
            Options: 'NMT', None
            Default: 'NMT'
        idw : bool, optional
            Whether to use inverse distance weighting for vector validation
            Default True.
        interp (str, optional
            Type of interpolation to perform
            Options: 'struc_lin', 'struc_cub'
            Default: 'struc_cub'
        part_detect : str, optional
            The type of particle detection to use
            Options: 'simple', 'local_thr'
            Default: 'simple'
        sd_P_target : int, optional
            The number of particles to target per kernel when estimating
            the seeding density.
            Refer to piv_image.calc_seeding_density for more information
            Default = 20
        target_init_NI : int, optional
            The number of particles to target per correlation window in the
            first iteration. Considering AIW, it is possible the resulting
            window will be significantly larger depending on the underlying
            displacement.
            Default = 20.
        target_fin_NI (int, optional
            The number of particles to target per correlation window in the last
            iteration. Unlike the initial target, the final WS should contain
            approximately this many particles, depending on the accuracy of
            particle detection and seeding density estimation
            Default = 8.
        """

        self.init_WS = init_WS
        self.final_WS = final_WS
        self.init_N_windows = init_N_windows
        self.final_N_windows = final_N_windows
        self.distribution_method = distribution_method
        self.n_iter_main = n_iter_main
        self.n_iter_ref = n_iter_ref
        self.vec_val = vec_val
        self.idw = idw
        self.interp = interp
        self.part_detect = part_detect
        self.sd_P_target = sd_P_target
        self.target_init_NI = target_init_NI
        self.target_fin_NI = target_fin_NI
        self.verbosity = verbosity

    def __eq__(self, other):
        """
        Allow for comparing equality between settings classes

        Parameters
        ----------
        other : WidimSettings
            The other WidimSettings to be compared to

        Returns
        -------
            Bool:
                Whether the two WidimSettings match
        """

        if not isinstance(other, MultiGridSettings):
            return NotImplemented

        for s, o in zip(self.__dict__.values(), other.__dict__.values()):
            if s != o:
                if not np.all(np.isnan((s, o))):
                    return False

        return True

    def __repr__(self):
        output = f" init_WS: {self.init_WS}\n"
        output += f" final_WS: {self.final_WS}\n"
        output += f" init_N_windows: {self.init_N_windows}\n"
        output += f" final_N_windows: {self.final_N_windows}\n"
        output += f" n_iter_main: {self.n_iter_main}\n"
        output += f" n_iter_ref: {self.n_iter_ref}\n"
        output += f" vec_val: {self.vec_val}\n"
        output += f" interp: {self.interp}\n"
        output += f" part_detect: {self.part_detect}\n"
        output += f" sd_P_target: {self.sd_P_target}\n"
        output += f" target_init_NI: {self.target_init_NI}\n"
        output += f" target_fin_NI: {self.target_fin_NI}\n"
        output += f" verbosity: {self.verbosity}\n"
        return output

    @property
    def init_WS(self):
        return self._init_ws

    @init_WS.setter
    def init_WS(self, value):
        """Sets the value of initial window size checking its validity

        Parameters
        ----------
        value : int or string
            Initial window size, if numeric, must be odd and 5 <= init_WS <= 245
            Otheriwse 'auto', where the window size will be calculated using the
            adaptive initial window routine.
        """

        if value is None or value == 'auto':
            self._init_ws = 'auto'
        elif type(value) is str and value != 'auto':
            raise ValueError("If non-numeric input, must be 'auto'")
        elif int(value) != value:
            raise ValueError("Initial WS must be integer")
        elif (value < 5) or (value > 245):
            raise ValueError("Initial WS must be 5 <= WS <= 245")
        elif value % 2 != 1:
            raise ValueError("Initial WS must be odd")
        else:
            self._init_ws = int(value)

    @property
    def final_WS(self):
        return self._final_WS

    @final_WS.setter
    def final_WS(self, value):
        """Sets the value of the final window size, checking validity

        Parameters
        ----------
        value : int or str
            Final window size, must be odd and 5 <= final_WS <= 245
            Otheriwse 'auto', where the window size will be calculated
            according to the seeding density
            Default 'auto'
        """

        if value is None or value == 'auto':
            self._final_WS = 'auto'
        elif type(value) is str and value != 'auto':
            raise ValueError("If non-numeric input, must be 'auto'")
        elif int(value) != value:
            raise ValueError("Final WS must be integer")
        elif (value < 5) or (value > 245):
            raise ValueError("Final WS must be 5 <= WS <= 245")
        elif value % 2 != 1:
            raise ValueError("Final WS must be odd")
        else:
            self._final_WS = value

    @property
    def init_N_windows(self):
        return self._init_N_windows

    @init_N_windows.setter
    def init_N_windows(self, value):
        """Sets the value of initial number of windows

        Parameters
        ----------
        value : int
            Initial number of windows to be used in the first iteration.
        """

        if int(value) != value:
            raise ValueError("Initial number of windows must be integer")
        else:
            self._init_N_windows = int(value)

    @property
    def final_N_windows(self):
        return self._final_N_windows

    @final_N_windows.setter
    def final_N_windows(self, value):
        """Sets the value of the final number of windows

        Parameters
        ----------
        value : int
            Final number of windows to be used
        """

        if int(value) != value:
            raise ValueError("Final number of windows must be integer")
        else:
            self._final_N_windows = value

    @property
    def n_iter_main(self):
        return self._n_iter_main

    @n_iter_main.setter
    def n_iter_main(self, value):
        """Sets the number of main iterations, checking validity

        Parameters
        ----------
        value : int
            Number of main iterations, wherein the WS and spacing will reduce
            from init_WS to final_WS 1 <= n_iter_main <= 10
            If the number of main iterations is 1 then the final_WS is ignored
        """
        if int(value) != value:
            raise ValueError("Number of iterations must be integer")
        if value < 1:
            raise ValueError("Number of iterations must be at least 1")
        if value > 10:
            raise ValueError(
                "Number of main iterations must be at most 10")

        self._n_iter_main = value

    @property
    def n_iter_ref(self):
        return self._n_iter_ref

    @n_iter_ref.setter
    def n_iter_ref(self, value):
        """Sets the number of refinement iterations, checking validity

        Parameters
        ----------
        value : int
            Number of refinement iterations, where the WS and locations remain
            fixed, but subsequent iterations are performed to improve the soln
            0 <= n_iter_ref <= 10
        """

        if int(value) != value:
            msg = "Number of refinement iterations must be integer"
            raise ValueError(msg)
        if value < 0:
            msg = "Number of refinement iterations must be at least 0"
            raise ValueError(msg)
        if value > 10:
            msg = "Number of refinement iterations must be at most 10"
            raise ValueError(msg)

        self._n_iter_ref = value

    @property
    def vec_val(self):
        return self._vec_val

    @vec_val.setter
    def vec_val(self, value):
        """Sets the type of vector validation, checking validity

        Parameters
        ----------
        value : str
            Type of vector validation to perform.
            Options: 'NMT', None
        """

        options = ['NMT', None]

        if value not in options:
            raise ValueError("Vector validation method not handled")

        self._vec_val = value

    @property
    def interp(self):
        return self._interp

    @interp.setter
    def interp(self, value):
        """Sets the type of interpolation, checking validity

        Parameters
        ----------
        value : int
            Type of interpolation to perform
            Options: 'struc_lin', 'struc_cub'
        """

        options = ['struc_lin', 'struc_cub', 'unstruc_cub']
        if value not in options:
            raise ValueError("Interpolation method not handled")

        self._interp = value
