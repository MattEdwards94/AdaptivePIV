import numpy as np
import PIV.distribution as distribution
import PIV.utilities as utilities
import math
import PIV.corr_window as corr_window
import PIV.dense_predictor as dense_predictor
import PIV.piv_image as piv_image
import PIV.ensemble_solution as es
import PIV.multiGrid as mg


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
        dp = widim(piv_image.load_PIVImage(flowtype, i),
                   settings)
        ensR.add_displacement_field(dp)

    return ensR


def multi_grid_analysis(img):
    """Analyses an image using the multi_grid approach
    """

    init_WS = 129
    final_WS = 65

    dp = dense_predictor.DensePredictor(
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
    dp = dense_predictor.DensePredictor(
        np.zeros(img.dim), np.zeros(img.dim), img.mask)

    # main iterations
    for iter_ in range(1, settings.n_iter_main + 1):
        print("Starting main iteration, {}".format(iter_))

        # calculate spacing and create sample grid
        print("Calculating WS and spacing")
        WS = WS_for_iter(iter_, settings)
        print("WS: {}".format(WS))
        h = max(1, math.floor((1 - settings.WOR) * WS))
        print(h)

        print("Creating grid and windows")
        xv, yv = (np.arange(0, img.n_cols, h),
                  np.arange(0, img.n_rows, h))
        xx, yy = np.meshgrid(xv, yv)
        print("{} windows".format(len(xx.ravel())))

        # create distribution of correlation windows
        cwList = corr_window.corrWindow_list(xx.ravel(), yy.ravel(), WS)
        dist = distribution.Distribution(cwList)

        print("Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            print("Validate vectors")
            dist.validation_NMT_8NN()

        print("Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
        dp = dense_predictor.DensePredictor(u, v, img_def.mask)

        print("Deforming image")
        img_def = img.deform_image(dp)

    print("Starting refinement iterations")

    for iter_ in range(1, settings.n_iter_ref + 1):

        print("Correlating all windows")
        dist.correlate_all_windows(img_def, dp)

        if settings.vec_val is not None:
            print("validate vectors")
            dist.validation_NMT_8NN()

        print("Interpolating")
        u, v = dist.interp_to_densepred(settings.interp, img_def.dim)
        dp = dense_predictor.DensePredictor(u, v, img_def.mask)

        print("Deforming image")
        img_def = img.deform_image(dp)

    return dp


def WS_for_iter(iter_, settings):
    """
    Returns the WS to be used for iteration iter_ for the current settings

    The window size is calculated by finding what reduction factor (RF) would
    need to be applied to init_WS, n_iter_main times
    such that at the end

    WS = init_WS*(RF^n_iter_main) = final_WS

    iter_ = 1 returns init_WS
        UNLESS
        iter_ = 1 and n_iter_main == 1, which returns final_WS
    iter_ >= n_iter_main returns final_WS

    Args:
        iter_ (int): The iteration to calculate the WS for.
                     Must be 1 <= iter_ <= n_iter_main + n_iter_ref
        settings (dict): Settings to be used, see 'widim_settings()'

    Returns:
        WS: Returns the WS rounded to the nearest odd integer
    """

    # check inputs for special cases
    if iter_ == 1:
        if settings.n_iter_main == 1:
            return settings.final_WS
        else:
            return settings.init_WS

    if iter_ >= settings.n_iter_main:
        return settings.final_WS

    # now calculate intermediate WS value
    reduction_fact = np.exp(
        np.log(settings.final_WS / settings.init_WS)
        / (settings.n_iter_main - 1)
    )
    WS = settings.init_WS * (reduction_fact ** (iter_ - 1))

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


def run_script():
    IA, IB, mask = piv_image.load_image_from_flow_type(22, 1)
    img = piv_image.PIVImage(IA, IB, mask)
    print("here")
    settings = WidimSettings(final_WS=15, n_iter_ref=0)

    widim(img, settings)


if __name__ == '__main__':
    # # load the image
    # flowtype, im_number = 1, 1
    # img = piv_image.load_PIVImage(flowtype, im_number)
    # # img.plot_images()
    # settings = WidimSettings(init_WS=129,
    #                          final_WS=65,
    #                          n_iter_main=2)

    # # analyse the image
    # dp = widim(img, settings)

    # # print(dp.u[200, 100])
    # # dp.plot_displacement_field(width=0.001,
    # #                            headlength=2.5,
    # #                            headwidth=2,
    # #                            headaxislength=6)

    # # ensR = ensemble_widim(22, 1, 2, settings)
    # # ensR.save_to_file('test_file.mat')

    img = piv_image.load_PIVImage(22, 1)

    multi_grid_analysis(img)