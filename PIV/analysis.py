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
from PIV.utilities import vprint

ESSENTIAL, BASIC, TERSE = 1, 2, 3


def ensemble_analysis(method, flowtype, im_start, im_stop, settings):
    """Analyses an ensemble of images and returns an EnsembleSolution object

    Parameters
    ----------
    method : function handle
        Handle to the function which is supposed to be used for the ensemble
        analysis. 
        Examples: 'widim', 'structured_adaptive_analysis' or 'adaptive_analysis'
    flowtype : int
        The flow type id. See image_info.all_flow_types()
    im_start : int
        The number of the first image in the series to analyse
    im_stop : int
        The number of the last image in the series to analyse 
        Inclusive. i.e. 1-3 will analyse 1, 2, and 3
    settings : SettingsClass
        Settings to analyse the images with. Must agree with the method
        e.g. 
        'widim' uses WidimSettings
        'structured_adaptive_analysis' uses AdaptStructSettings
        'adaptive_analysis' uses AdaptSettings

    Returns
    solution: EnsembleSolution
        An EnsembleSolution object. 
        See ensemble_solution.py
    """

    # set the verbosity level
    prev_verb = PIV.utilities._verbosity
    PIV.utilities._verbosity = settings.verbosity

    ensR = es.EnsembleSolution(settings, flowtype)

    for i in range(im_start, im_stop + 1):
        vprint(1, "Analysing image {}".format(i))
        dp = method(PIV.PIVImage.from_flowtype(flowtype, i),
                    settings)
        if type(dp) is tuple:
            dp, dist = dp
        ensR.add_displacement_field(dp)

    # reset verbosity
    PIV.utilities._verbosity = prev_verb

    return ensR


def multi_grid_analysis(img):
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


def widim(img, settings):
    """
    Performs a widim analysis on the PIVImage object, img, with the settings
    defined in settings

    Args:
        img (PIVImage): PIVImage object containing the images to be analysed
        settings (dict): dictionary of settings obtained by using
                         'widim_settings()'
    """

    # set the verbosity level
    prev_verb = PIV.utilities._verbosity
    PIV.utilities._verbosity = settings.verbosity

    img_def = img
    dp = PIV.DensePredictor(
        np.zeros(img.dim), np.zeros(img.dim), img.mask)

    # main iterations
    for _iter in range(1, settings.n_iter_main + 1):
        vprint(BASIC, "Starting main iteration, {}".format(_iter))

        # calculate spacing and create sample grid
        vprint(BASIC, "Calculating WS and spacing")
        WS = WS_for_iter(_iter, settings)
        vprint(BASIC, "WS: {}".format(WS))
        h = max(1, math.floor((1 - settings.WOR) * WS))
        vprint(BASIC, h)

        vprint(BASIC, "Creating grid and windows")
        xv, yv = (np.arange(0, img.n_cols, h),
                  np.arange(0, img.n_rows, h))
        xx, yy = np.meshgrid(xv, yv)
        ws_grid = np.ones_like(xx) * WS
        vprint(BASIC, "{} windows".format(len(xx.ravel())))

        # create distribution of correlation windows
        dist = PIV.Distribution.from_locations(xx, yy, ws_grid)

        vprint(BASIC, "Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            vprint(BASIC, "Validate vectors")
            dist.validation_NMT_8NN()

        vprint(BASIC, "Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
        dp = PIV.DensePredictor(u, v, img_def.mask)

        vprint(BASIC, "Deforming image")
        img_def = img.deform_image(dp)

    vprint(BASIC, "Starting refinement iterations")

    for _iter in range(1, settings.n_iter_ref + 1):

        vprint(BASIC, "Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            vprint(BASIC, "validate vectors")
            dist.validation_NMT_8NN()

        vprint(BASIC, "Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
        dp = PIV.DensePredictor(u, v, img_def.mask)

        vprint(2, "Deforming image")
        img_def = img.deform_image(dp)

    # reset verbosity
    PIV.utilities._verbosity = prev_verb

    return dp


def WS_for_iter(_iter, settings):
    """
    Returns the WS to be used for iteration _iter for the current settings

    The window size is calculated by finding what reduction factor (RF) would
    need to be applied to init_WS, n_iter_main times
    such that at the end

    WS = init_WS*(RF^n_iter_main) = final_WS

    _iter = 1 returns init_WS
        UNLESS
        iter_ = 1 and n_iter_main == 1, which returns final_WS
    _iter >= n_iter_main returns final_WS

    Args:
        iter_ (int): The iteration to calculate the WS for.
                     Must be 1 <= iter_ <= n_iter_main + n_iter_ref
        settings (dict): Settings to be used, see 'widim_settings()'

    Returns:
        WS: Returns the WS rounded to the nearest odd integer
    """

    # check inputs for special cases
    if _iter == 1:
        if settings.n_iter_main == 1:
            return settings.final_WS
        else:
            return settings.init_WS

    if _iter >= settings.n_iter_main:
        return settings.final_WS

    # now calculate intermediate WS value
    reduction_fact = np.exp(
        np.log(settings.final_WS / settings.init_WS)
        / (settings.n_iter_main - 1)
    )
    WS = settings.init_WS * (reduction_fact ** (_iter - 1))

    # return the nearest odd integer
    return utilities.round_to_odd(WS)


class WidimSettings():

    def __init__(self, init_WS=97, final_WS=33, WOR=0.5,
                 n_iter_main=3, n_iter_ref=2,
                 vec_val='NMT', interp='struc_cub',
                 verbosity=2):
        """

        Args:
            init_WS (int, optional): Initial window size, must be odd and
                                     5 <= init_WS <= 245
            final_WS (int, optional): Final window size, must be odd and
                                      5 <= final_WS <= 245
            WOR (float, optional): Window overlap ratio, must be 0 <= WOR < 1
            n_iter_main (int, optional): Number of main iterations, wherein the
                                         WS and spacing will reduce from init_WS
                                         to final_WS
                                         Must be 1 <= n_iter_main <= 10
                                         If the number of main iterations is 1
                                         then the final_WS is ignored
            n_iter_ref (int, optional): Number of refinement iterations, where
                                        the WS and locations remain fixed,
                                        however, subsequent iterations are
                                        performed to improve the solution
                                        Must be 0 <= n_iter_ref <= 10
            vec_val (str, optional): Type of vector validation to perform.
                                     Options: 'NMT', None
                                     Default: 'NMT'
            interp (str, optional): Type of interpolation to perform
                                    Options: 'struc_lin', 'struc_cub'
                                    Default: 'struc_cub'
        """

        self.init_WS = init_WS
        self.final_WS = final_WS
        self.WOR = WOR
        self.n_iter_main = n_iter_main
        self.n_iter_ref = n_iter_ref
        self.vec_val = vec_val
        self.interp = interp
        self.verbosity = verbosity

    def __eq__(self, other):
        """
        Allow for comparing equality between settings classes

        Args:
            other (WidimSettings): The other WidimSettings to be compared to

        Returns:
            Bool: Whether the two WidimSettings match
        """

        if not isinstance(other, WidimSettings):
            return NotImplemented

        for s, o in zip(self.__dict__.values(), other.__dict__.values()):
            if s != o:
                if not np.all(np.isnan((s, o))):
                    return False

        return True

    def __repr__(self):
        output = f" init_WS: {self.init_WS}\n"
        output += f" final_WS: {self.final_WS}\n"
        output += f" WOR: {self.WOR}\n"
        output += f" n_iter_main: {self.n_iter_main}\n"
        output += f" n_iter_ref: {self.n_iter_ref}\n"
        output += f" vec_val: {self.vec_val}\n"
        output += f" interp: {self.interp}\n"
        return output

    @property
    def init_WS(self):
        return self._init_ws

    @init_WS.setter
    def init_WS(self, value):
        """Sets the value of initial window size checking its validity

        Args:
            value (int): Initial window size,
                         must be odd
                         5 <= init_WS <= 245
                         init_WS >= final_WS (Checked in final_WS)
        """

        if int(value) != value:
            raise ValueError("Initial WS must be integer")
        if (value < 5) or (value > 245):
            raise ValueError("Initial WS must be 5 <= WS <= 245")
        if value % 2 != 1:
            raise ValueError("Initial WS must be odd")

        self._init_ws = int(value)

    @property
    def final_WS(self):
        return self._final_WS

    @final_WS.setter
    def final_WS(self, value):
        """Sets the value of the final window size, checking validity

        Args:
            value (int): Final window size,
                         Must be odd
                         5 <= final_WS <= 245
                         final_WS <= init_WS
        """

        if int(value) != value:
            raise ValueError("Final WS must be integer")
        if (value < 5) or (value > 245):
            raise ValueError("Final WS must be 5 <= WS <= 245")
        if value % 2 != 1:
            raise ValueError("Final WS must be odd")
        if self.init_WS < value:
            raise ValueError("Final WS must be <= init WS")
        self._final_WS = value

    @property
    def WOR(self):
        return self._WOR

    @WOR.setter
    def WOR(self, value):
        """Sets the window overlap ratio, checking validity

        Args:
            value (float): overlap factor as decimal 0 <= WOR < 1
                           0 represents no overlap
                           1 represents full overlap
                              This would result in all windows in the same
                              location and is hence invalid
        """
        if value < 0:
            raise ValueError("WOR must be greater than 0")
        if value >= 1:
            raise ValueError("WOR must be strictly less than 1")
        self._WOR = value

    @property
    def n_iter_main(self):
        return self._n_iter_main

    @n_iter_main.setter
    def n_iter_main(self, value):
        """Sets the number of main iterations, checking validity

        Args:
            value (float): Number of main iterations, wherein the WS and
                           spacing will reduce from init_WS to final_WS
                           1 <= n_iter_main <= 10
                           If the number of main iterations is 1
                           then the final_WS is ignored
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

        Args:
            value (float): Number of refinement iterations, where the
                           WS and locations remain fixed, however,
                           subsequent iterations are performed to
                           improve the solution
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

        Args:
            value (float): Type of vector validation to perform.
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

        Args:
            value (float): Type of interpolation to perform
                            Options: 'struc_lin', 'struc_cub'
        """

        options = ['struc_lin', 'struc_cub']
        if value not in options:
            raise ValueError("Interpolation method not handled")

        self._interp = value


def structured_adaptive_analysis(img, settings, **kwargs):
    """
    Analyses the PIV image using a structured grid, but allowing for
    adaptively sized windows

    Args:
        img (PIVImage): PIVImage object containing the images to be analysed
        settings (dict): dictionary of settings obtained by using
                        'AdaptStructSettings()'

    """

    # set the verbosity level
    prev_verb = PIV.utilities._verbosity
    PIV.utilities._verbosity = settings.verbosity

    img_def = img
    dp = dense_predictor.DensePredictor(
        np.zeros(img.dim), np.zeros(img.dim), img.mask)

    # For now, lets forget about auto spacing - we will consider this later
    if 'auto' in [settings.init_spacing, settings.final_spacing]:
        raise NotImplementedError("Auto spacing is not implemented yet")

    # if one of init/final WS/spacing are auto, we will need the seeding info
    if 'auto' in [settings.init_WS, settings.final_WS,
                  settings.init_spacing, settings.final_spacing]:
        img.calc_seed_density(method=settings.part_detect,
                              P_target=settings.sd_P_target)
        min_sd = np.minimum(img.sd_IA, img.sd_IB)
        min_sd[min_sd == 0] = 0.0021

    if settings.final_WS == 'auto':
        # get WS based on seeding only.
        ws_final = utilities.round_to_odd(np.sqrt(settings.target_fin_NI /
                                                  min_sd))
    else:
        ws_final = settings.final_WS * np.ones(img_def.dim)

    for _iter in range(1, settings.n_iter_main+1):
        vprint(BASIC, "Starting main iteration, {}".format(_iter))

        vprint(BASIC, "Creating sampling grid")
        init_h, fin_h = settings.init_spacing, settings.final_spacing
        if settings.n_iter_main == 1:
            h = init_h
        else:
            h = fin_h + ((_iter-1) / (settings.n_iter_main - 1)) * \
                (fin_h - init_h)
        vprint(BASIC, f"  sample spacing: {h}")
        xv, yv = (np.arange(0, img.n_cols, h),
                  np.arange(0, img.n_rows, h))
        xx, yy = np.meshgrid(xv, yv)
        vprint(BASIC, "{} windows".format(len(xx.ravel())))

        # correlate using adaptive initial window size.
        if _iter == 1:
            # if the initial WS is auto, calculate this
            if settings.init_WS == 'auto':
                # get WS based on seeding only.
                ws_seed_init = utilities.round_to_odd(
                    np.sqrt(settings.target_init_NI / min_sd))
                ws_seed_init[np.isnan(ws_seed_init)] = 97

                # create correlation windows
                dist = distribution.Distribution.from_locations(xx,
                                                                yy,
                                                                ws_seed_init[yy, xx])

                # analyse the windows
                vprint(BASIC, "Analysing first iteration with AIW")
                dist.AIW(img_def, dp, **kwargs)

                # need to store the actual initial WS for subsequent iterations
                ws_first_iter = dist.interp_WS(img_def.mask)
                # vprint(BASIC, ws_first_iter)
                # fig = plt.figure(figsize=(20, 10))
            else:
                # just create and correlate the windows
                dist = distribution.Distribution.from_locations(xx,
                                                                yy,
                                                                settings.init_WS)

                ws_first_iter = np.ones(img_def.dim) * settings.init_WS
                vprint(BASIC, "Analysing first iteration with uniform window size")
                dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                vprint(BASIC, "Validate vectors")
                dist.validation_NMT_8NN()

            vprint(BASIC, "Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
            dp = dense_predictor.DensePredictor(u, v, img_def.mask)
            # dp.plot_displacement_field()

            vprint(BASIC, "Deforming image")
            img_def = img.deform_image(dp)

        else:
            # get WS for current iteration,
            ws = ws_first_iter + ((_iter-1) / (settings.n_iter_main - 1)) * \
                (ws_final - ws_first_iter)
            ws = utilities.round_to_odd(ws)
            ws[np.isnan(ws)] = 5

            # create correlation windows
            dist = distribution.Distribution.from_locations(xx, yy, ws)

            # correlate windows
            dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                vprint(BASIC, "Validate vectors")
                dist.validation_NMT_8NN()

            vprint(BASIC, "Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
            dp = dense_predictor.DensePredictor(u, v, img_def.mask)

            vprint(BASIC, "Deforming image")
            img_def = img.deform_image(dp)

    vprint(BASIC, "Refinement iterations")
    for _iter in range(1, settings.n_iter_ref + 1):

        vprint(BASIC, "Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            vprint(BASIC, "validate vectors")
            dist.validation_NMT_8NN()

        vprint(BASIC, "Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
        dp = PIV.DensePredictor(u, v, img_def.mask)

        vprint(BASIC, "Deforming image")
        img_def = img.deform_image(dp)

    # reset verbosity
    PIV.utilities._verbosity = prev_verb

    return dp, ws_first_iter


class AdaptStructSettings():
    """
    Class to hold the settings required for a structured, adaptive, analysis.
    """

    def __init__(self,
                 init_WS=None,
                 final_WS=None,
                 init_spacing=None,
                 final_spacing=None,
                 n_iter_main=3,
                 n_iter_ref=2,
                 vec_val='NMT',
                 interp='struc_cub',
                 part_detect='simple',
                 sd_P_target=20,
                 target_init_NI=20,
                 target_fin_NI=8,
                 verbosity=2):
        """

        Args:
            init_WS (int/str, optional): Initial window size, if numeric,
                                         must be odd and
                                         5 <= init_WS <= 245
                                         Otheriwse 'auto', where the window
                                         size will be calculated using the
                                         adaptive initial window routine.
                                         Default 'auto'.
            final_WS (int/str, optional): Final window size, must be odd and
                                          5 <= final_WS <= 245
                                          Otheriwse 'auto', where the window
                                          size will be calculated according to
                                          the seeding density
            init_spacing (int/str, optional): The initial spacing between
                                              samples, in px. The spacing will
                                              decline linearly from init to
                                              final over the first n_iter_main
                                              where possible.
                                              Must be x >= 2.
                                              Alternatively 'Auto' to calculate
                                              based on the number of particles
            final_spacing (int/str, optional): The final spacing between
                                               corr windows, in px.
                                               If only one iteration is
                                               requested, the initial grid
                                               spacing will be used.
                                               Must be x >= 2
                                               Alternatively 'Auto' to calculate
                                               based on the number of particles
            n_iter_main (int, optional): Number of main iterations, wherein the
                                         WS and spacing will reduce from init_WS
                                         to final_WS
                                         Must be 1 <= n_iter_main <= 10
                                         If the number of main iterations is 1
                                         then the final_WS is ignored
            n_iter_ref (int, optional): Number of refinement iterations, where
                                        the WS and locations remain fixed,
                                        however, subsequent iterations are
                                        performed to improve the solution
                                        Must be 0 <= n_iter_ref <= 10
            vec_val (str, optional): Type of vector validation to perform.
                                     Options: 'NMT', None
                                     Default: 'NMT'
            interp (str, optional): Type of interpolation to perform
                                    Options: 'struc_lin', 'struc_cub'
                                    Default: 'struc_cub'
            part_detect (str, optional): The type of particle detection to use
            sd_P_target (int, optional): The number of particles to target per
                                         kernel when estimating the seeding
                                         density.
                                         default = 20
                                         Refer to piv_image.calc_seeding_density
                                         for more information
            target_init_NI (int, optional): The number of particles to target
                                            per correlation window in the first
                                            iteration.
                                            Considering AIW, it is possible the
                                            resulting window will be
                                            significantly larger depending on
                                            the underlying displacement.
                                            default = 20.
            target_fin_NI (int, optional): The number of particles to target
                                           per correlation window in the last
                                           iteration.
                                           Unlike the initial target, the final
                                           WS should contain approximately this
                                           many particles, depending on the
                                           accuracy of particle detection and
                                           seeding density estimation
                                           default = 8.
        """

        self.init_WS = init_WS
        self.final_WS = final_WS
        self.init_spacing = init_spacing
        self.final_spacing = final_spacing
        self.n_iter_main = n_iter_main
        self.n_iter_ref = n_iter_ref
        self.vec_val = vec_val
        self.interp = interp
        self.part_detect = part_detect
        self.sd_P_target = sd_P_target
        self.target_init_NI = target_init_NI
        self.target_fin_NI = target_fin_NI
        self.verbosity = verbosity

    def __eq__(self, other):
        """
        Allow for comparing equality between settings classes

        Args:
            other (WidimSettings): The other WidimSettings to be compared to

        Returns:
            Bool: Whether the two WidimSettings match
        """

        if not isinstance(other, AdaptStructSettings):
            return NotImplemented

        for s, o in zip(self.__dict__.values(), other.__dict__.values()):
            if s != o:
                if not np.all(np.isnan((s, o))):
                    return False

        return True

    def __repr__(self):
        output = f" init_WS: {self.init_WS}\n"
        output += f" final_WS: {self.final_WS}\n"
        output += f" vec_val: {self.vec_val}\n"
        output += f" interp: {self.interp}\n"
        return output

    @property
    def init_WS(self):
        return self._init_ws

    @init_WS.setter
    def init_WS(self, value):
        """Sets the value of initial window size checking its validity

        Args:
            value (int/string): Initial window size,
                                if numeric, must be odd
                                5 <= init_WS <= 245
                                if string, must be 'auto'
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

        Args:
            value (int): Final window size,
                         if numeric, must be odd
                         5 <= final_WS <= 245
                         otherwise 'auto'
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
    def init_spacing(self):
        return self._init_spacing

    @init_spacing.setter
    def init_spacing(self, value):
        """Sets the value of initial vector spacing checking its validity

        Args:
            value (int/string): Initial vector spacing in px,
                                if numeric, must be >=2
                                if string, must be 'auto'
        """

        if value is None or value == 'auto':
            self._init_spacing = 'auto'
        elif type(value) is str and value != 'auto':
            raise ValueError("If non-numeric input, must be 'auto'")
        elif int(value) != value:
            raise ValueError("Initial spacing must be integer")
        elif (value <= 2):
            raise ValueError("Initial spacing must be 2 <= spacing")
        else:
            self._init_spacing = int(value)

    @property
    def final_spacing(self):
        return self._final_spacing

    @final_spacing.setter
    def final_spacing(self, value):
        """Sets the value of the final vector spacing, checking validity

        Args:
            value (int/str): Final vector spacing, px
                             if numeric, must be >=2
                             otherwise 'auto'
        """

        if value is None or value == 'auto':
            self._final_spacing = 'auto'
        elif type(value) is str and value != 'auto':
            raise ValueError("If non-numeric input, must be 'auto'")
        elif int(value) != value:
            raise ValueError("Final spacing must be integer")
        elif (value <= 2):
            raise ValueError("Final spacing must be 2 <= spacing")
        else:
            self._final_spacing = value

    @property
    def n_iter_main(self):
        return self._n_iter_main

    @n_iter_main.setter
    def n_iter_main(self, value):
        """Sets the number of main iterations, checking validity

        Args:
            value (float): Number of main iterations, wherein the WS and
                           spacing will reduce from init_WS to final_WS
                           1 <= n_iter_main <= 10
                           If the number of main iterations is 1
                           then the final_WS is ignored
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

        Args:
            value (float): Number of refinement iterations, where the
                           WS and locations remain fixed, however,
                           subsequent iterations are performed to
                           improve the solution
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

        Args:
            value (float): Type of vector validation to perform.
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

        Args:
            value (float): Type of interpolation to perform
                            Options: 'struc_lin', 'struc_cub'
        """

        options = ['struc_lin', 'struc_cub']
        if value not in options:
            raise ValueError("Interpolation method not handled")

        self._interp = value


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
    dp = dense_predictor.DensePredictor(
        np.zeros(img.dim), np.zeros(img.dim), img.mask)

    # if one of init/final WS are auto, we will need the seeding info
    if 'auto' in [settings.init_WS, settings.final_WS]:
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

        n_windows = (settings.init_N_windows +
                     np.floor(settings.final_N_windows *
                              (_iter-1)/settings.n_iter_main))

        vprint(BASIC, "Creating sampling distribution")
        if settings.distribution_method == "AIS":
            xy_dist = distribution.AIS(phi, img.mask, n_windows)
            xx, yy = xy_dist[:, 0], xy_dist[:, 1]
        else:
            raise NotImplementedError("Distribution method not implemented")

        vprint(BASIC, "{} windows".format(len(xx)))

        # correlate using adaptive initial window size.
        if _iter == 1:
            # if the initial WS is auto, calculate this
            if settings.init_WS == 'auto':
                # get WS based on seeding only.
                ws_seed_init = utilities.round_to_odd(
                    np.sqrt(settings.target_init_NI / min_sd))
                ws_seed_init[np.isnan(ws_seed_init)] = 97

                # create correlation windows
                dist = distribution.Distribution.from_locations(xx, yy,
                                                                ws_seed_init[yy, xx])

                # analyse the windows
                vprint(BASIC, "Analysing first iteration with AIW")
                dist.AIW(img_def, dp)

                # need to store the actual initial WS for subsequent iterations
                ws_first_iter = dist.interp_WS_unstructured(img_def.mask)
            else:
                # just create and correlate the windows
                dist = distribution.Distribution.from_locations(xx,
                                                                yy,
                                                                settings.init_WS)

                ws_first_iter = np.ones(img_def.dim) * settings.init_WS
                vprint(BASIC, "Analysing first iteration with uniform window size")
                dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                vprint(BASIC, "Validate vectors")
                dist.validation_NMT_8NN()

            vprint(BASIC, "Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim,
                                            inter_h=4)
            dp = dense_predictor.DensePredictor(u, v, img_def.mask)
            # dp.plot_displacement_field()

            vprint(BASIC, "Deforming image")
            img_def = img.deform_image(dp)

        else:
            # get WS for current iteration,
            ws = ws_first_iter + ((_iter-1) / (settings.n_iter_main - 1)) * \
                (ws_final - ws_first_iter)
            ws = utilities.round_to_odd(ws)
            ws[np.isnan(ws)] = 5
            ws = ws[yy, xx]

            # create correlation windows
            dist = distribution.Distribution.from_locations(xx, yy, ws)

            # correlate windows
            dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                vprint(BASIC, "Validate vectors")
                dist.validation_NMT_8NN()

            vprint(BASIC, "Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim,
                                            inter_h=4)
            dp = dense_predictor.DensePredictor(u, v, img_def.mask)

            vprint(BASIC, "Deforming image")
            img_def = img.deform_image(dp)

    vprint(BASIC, "Refinement iterations")
    for _iter in range(1, settings.n_iter_ref + 1):

        vprint(BASIC, "Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            vprint(BASIC, "validate vectors")
            dist.validation_NMT_8NN()

        vprint(BASIC, "Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim,
                                        inter_h=4)
        dp = PIV.DensePredictor(u, v, img_def.mask)

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
                 vec_val='NMT', interp='unstruc_cub',
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


if __name__ == '__main__':
    img = PIV.PIVImage.from_flowtype(22, 1)
    sol = widim(img, WidimSettings())
    # multi_grid_analysis(img)
