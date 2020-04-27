import numpy as np
from PIV.distribution import Distribution
import PIV.utilities as utilities
import math
import PIV.corr_window as corr_window
from PIV.dense_predictor import DensePredictor
import PIV.ensemble_solution as es
import PIV.multiGrid as mg
from PIV.multiGrid import MultiGrid
import PIV
import matplotlib.pyplot as plt
from PIV.utilities import vprint, WS_for_iter

ESSENTIAL, BASIC, TERSE = 1, 2, 3


def adaptive_analysis(img, settings):
    """Analyse the image specified by img using an adaptive analysis approach

    Parameters
    ----------
    img : PIVImage
        The piv image object containing the image intensities and the mask
        information
    settings : AdaptSettings
        Settings class instructing how to analyse the image
    """

    # set the verbosity level
    prev_verb = PIV.utilities._verbosity
    PIV.utilities._verbosity = settings.verbosity

    img_def = img
    dp = DensePredictor(np.zeros(img.dim), np.zeros(img.dim), img.mask)

    # if one of init/final WS are auto, we will need the seeding info
    img.calc_seed_density(method=settings.part_detect,
                          P_target=settings.sd_P_target)
    min_sd = np.minimum(img.sd_IA, img.sd_IB)
    # if ends up being 0, just assume it is some low value
    min_sd[min_sd == 0] = 0.0021

    if settings.final_WS == 'auto':
        # get WS based on seeding only.
        ws_final = utilities.round_to_odd(np.sqrt(settings.target_fin_NI /
                                                  min_sd))
    else:
        ws_final = settings.final_WS * np.ones(img_def.dim)

    phi = np.ones(img.dim)*img.mask

    for _iter in range(1, settings.n_iter_main+1):
        vprint(BASIC, "Starting main iteration, {}".format(_iter))

        delta = settings.final_N_windows - settings.init_N_windows

        n_windows = (settings.init_N_windows +
                     np.floor(delta *
                              (_iter-1)/(settings.n_iter_main-1)))

        vprint(BASIC, "Creating sampling distribution")
        if _iter > 1:
            [u_var, v_var] = dp.spatial_variance(33, 5)
            phi_flow = np.sqrt(u_var + v_var)
            phi_flow /= np.max(phi_flow)
            bf = min_sd
            bf[np.isnan(min_sd)] = 0
            phi_seed = min_sd / (np.max(bf)*4)
            phi_seed[img.mask == 0] = 0
            phi = (phi_flow + phi_seed)
        if settings.distribution_method == "AIS":
            dist = Distribution.from_AIS(phi, img.mask, n_windows)
        elif settings.distribution_method == "MG":
            dist = MultiGrid.from_obj_func(phi, n_windows, img.mask)
        else:
            raise NotImplementedError("Distribution method not implemented")

        vprint(BASIC, "{} windows".format(dist.n_windows()))

        # correlate using adaptive initial window size.
        if _iter == 1:
            # if the initial WS is auto, calculate this
            if settings.init_WS == 'auto':
                # get WS based on seeding only.
                ws_seed_init = utilities.round_to_odd(
                    np.sqrt(settings.target_init_NI / min_sd))
                ws_seed_init[np.isnan(ws_seed_init)] = 97

                # initialise window size from seeding
                for win in dist:
                    win.WS = ws_seed_init[win.y, win.x]

                # analyse the windows
                vprint(BASIC, "Analysing first iteration with AIW")
                dist.AIW(img_def, dp)

                # need to store the actual initial WS for subsequent iterations
                ws_first_iter = dist.interp_WS_unstructured(img_def.mask)
            else:
                # set windows to a uniform size
                for win in dist:
                    win.WS = settings.init_WS

                ws_first_iter = np.ones(img_def.dim) * settings.init_WS
                vprint(BASIC, "Analysing first iteration with uniform window size")
                dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                vprint(BASIC, "Validate vectors")
                dist.validation_NMT_8NN(idw=settings.idw)

            vprint(BASIC, "Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim,
                                            inter_h=4)
            dp = DensePredictor(u, v, img_def.mask)
            # dp.plot_displacement_field()

            vprint(BASIC, "Deforming image")
            img_def = img.deform_image(dp)

        else:
            # get WS for current iteration,
            ws = ws_first_iter + ((_iter-1) / (settings.n_iter_main - 1)) * \
                (ws_final - ws_first_iter)
            ws = utilities.round_to_odd(ws)
            ws[np.isnan(ws)] = 5

            for win in dist:
                win.WS = ws[win.y, win.x]

            # correlate windows
            dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                vprint(BASIC, "Validate vectors")
                dist.validation_NMT_8NN(idw=settings.idw)

            vprint(BASIC, "Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim,
                                            inter_h=4)
            dp = DensePredictor(u, v, img_def.mask)

            vprint(BASIC, "Deforming image")
            img_def = img.deform_image(dp)

    vprint(BASIC, "Refinement iterations")
    for _iter in range(1, settings.n_iter_ref + 1):

        vprint(BASIC, "Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            vprint(BASIC, "validate vectors")
            dist.validation_NMT_8NN(idw=settings.idw)

        vprint(BASIC, "Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim,
                                        inter_h=4)
        dp = DensePredictor(u, v, img_def.mask)

        vprint(BASIC, "Deforming image")
        img_def = img.deform_image(dp)

    # reset verbosity
    PIV.utilities._verbosity = prev_verb

    return dp, dist


class AdaptSettings():
    """
    Class to hold the settings required for an unstructured, adaptive, analysis.
    """

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

        if not isinstance(other, AdaptSettings):
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
