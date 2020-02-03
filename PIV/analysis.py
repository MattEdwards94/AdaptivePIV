import numpy as np
import PIV.distribution as distribution
import PIV.utilities as utilities
import math
import PIV.corr_window as corr_window
import PIV.dense_predictor as dense_predictor
import PIV.ensemble_solution as es
import PIV.multiGrid as mg
import PIV


def ensemble_widim(flowtype, im_start, im_stop, settings):
    """Analyses an ensemble of images and returns an EnsembleSolution object

    Args:
        flowtype (int): The flow type id. See image_info.all_flow_types()
        im_start (int): The number of the first image in the series to analyse
        im_stop (int): The number of the last image in the series to analyse
                       Inclusive. i.e. 1-3 will analyse 1, 2, and 3
        settings (dict): Settings to analyse the images with

    Returns:
        EnsembleSolution: An EnsembleSolution object. See ensemble_solution.py
    """
    ensR = es.EnsembleSolution(settings, flowtype)

    for i in range(im_start, im_stop + 1):
        print("Analysing image {}".format(i))
        dp = widim(PIV.PIVImage.from_flowtype(flowtype, i),
                   settings)
        ensR.add_displacement_field(dp)

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

    img_def = img
    dp = PIV.DensePredictor(
        np.zeros(img.dim), np.zeros(img.dim), img.mask)

    # main iterations
    for _iter in range(1, settings.n_iter_main + 1):
        print("Starting main iteration, {}".format(_iter))

        # calculate spacing and create sample grid
        print("Calculating WS and spacing")
        WS = WS_for_iter(_iter, settings)
        print("WS: {}".format(WS))
        h = max(1, math.floor((1 - settings.WOR) * WS))
        print(h)

        print("Creating grid and windows")
        xv, yv = (np.arange(0, img.n_cols, h),
                  np.arange(0, img.n_rows, h))
        xx, yy = np.meshgrid(xv, yv)
        ws_grid = np.ones_like(xx) * WS
        print("{} windows".format(len(xx.ravel())))

        # create distribution of correlation windows
        dist = PIV.Distribution.from_locations(xx, yy, ws_grid)

        print("Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            print("Validate vectors")
            dist.validation_NMT_8NN()

        print("Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
        dp = PIV.DensePredictor(u, v, img_def.mask)

        print("Deforming image")
        img_def = img.deform_image(dp)

    print("Starting refinement iterations")

    for _iter in range(1, settings.n_iter_ref + 1):

        print("Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            print("validate vectors")
            dist.validation_NMT_8NN()

        print("Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
        dp = PIV.DensePredictor(u, v, img_def.mask)

        print("Deforming image")
        img_def = img.deform_image(dp)

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
                 vec_val='NMT', interp='struc_cub'):
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

def structured_adaptive_analysis(img, settings):
    """
    Analyses the PIV image using a structured grid, but allowing for 
    adaptively sized windows

    Args:
        img (PIVImage): PIVImage object containing the images to be analysed
        settings (dict): dictionary of settings obtained by using
                        'AdaptStructSettings()'

    """
    # For now, lets forget about auto spacing - we will consider this later
    if 'auto' in [settings.init_spacing, settings.final_spacing]:
        raise NotImplementedError("Auto spacing is not implemented yet")

    # if one of init/final WS/spacing are auto, we will need the seeding info
    if 'auto' in [settings.init_WS, settings.final_WS,
                  settings.init_spacing, settings.final_spacing]:
        img.calc_seeding_density(method=settings.part_detect,
                                 P_target=settings.sd_P_target)

    for _iter in range(1, settings.n_iter_main+1):
        print("Starting main iteration, {}".format(iter_))

        print("Creating sampling grid")
        init_h, fin_h = settings.init_spacing, settings.final_spacing
        h = fin_h + ((_iter-1) / (settings.n_iter_main - 1)) * (fin_h - init_h)
        print(f"  sample spacing: {h}")
        xv, yv = (np.arange(0, img.n_cols, h),
                  np.arange(0, img.n_rows, h))
        xx, yy = np.meshgrid(xv, yv)
        print("{} windows".format(len(xx.ravel())))

        # correlate using adaptive initial window size.
        if _iter == 1:
            # if the initial WS is auto, calculate this
            if settings.init_WS == 'auto':
                # get WS based on seeding only.
                ws_seed_init = utilities.round_to_odd(
                    np.sqrt(settings.target_init_NI / min_sd))
                ws_seed_init[np.isnan(ws_seed_init)] = 97

                # create correlation windows
                cw_list = corr_window.corrWindow_list(xx,
                                                      yy,
                                                      ws_seed_init)
                dist = distribution.Distribution(cw_list)

                # analyse the windows
                print("Analysing first iteration with AIW")
                dist.AIW(img_def, dp)
                

                # need to store the actual initial WS for subsequent iterations
                ws_first_iter = dist.interp_WS(img_def.mask)
                # print(ws_first_iter)
                fig = plt.figure(figsize=(20, 10))
            else:
                # just create and correlate the windows
                cw_list = corr_window.corrWindow_list(xx.ravel(),
                                                      yy.ravel(),
                                                      settings.init_WS)
                dist = distribution.Distribution(cw_list)

                ws_first_iter = np.ones(img_def.dim) * settings.init_WS
                print("Analysing first iteration with uniform window size")
                dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                print("Validate vectors")
                dist.validation_NMT_8NN()

            print("Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
            dp = dense_predictor.DensePredictor(u, v, img_def.mask)
            # dp.plot_displacement_field()

            print("Deforming image")
            img_def = img.deform_image(dp)

        else:
            # get WS for current iteration,
            ws = ws_first_iter + ((_iter-1) / (settings.n_iter_main - 1)) * \
                (ws_final - ws_first_iter)
            ws = utilities.round_to_odd(ws)
            ws[np.isnan(ws)] = 5

            # create correlation windows
            cw_list = corr_window.corrWindow_list(xx.ravel(),
                                                  yy.ravel(),
                                                  ws)
            dist = distribution.Distribution(cw_list)

            # correlate windows
            dist.correlate_all_windows(img_def, dp)

            if settings.vec_val is not None:
                print("Validate vectors")
                dist.validation_NMT_8NN()

            print("Interpolating")
            u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
            dp = dense_predictor.DensePredictor(u, v, img_def.mask)

            print("Deforming image")
            img_def = img.deform_image(dp)

    print("Refinement iterations")
    for _iter in range(1, settings.n_iter_ref + 1):

        # create correlation windows
        cw_list = corr_window.corrWindow_list(xx.ravel(), yy.ravel(), ws_seed)
        dist = distribution.Distribution(cw_list)

        # correlate using adaptive initial window size.
        dist.AIW(img)

        # if

        # ~~~~ Create sampling grid ~~~~ #


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
                 sd_P_target=20):
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


if __name__ == '__main__':
    img = PIV.PIVImage.from_flowtype(22, 1)

    multi_grid_analysis(img)
