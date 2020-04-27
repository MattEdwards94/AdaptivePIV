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
import pdb

ESSENTIAL, BASIC, TERSE = 1, 2, 3


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

    return dp, dist


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
