import numpy as np
import distribution
import utilities
import math
import corr_window
import dense_predictor
import matplotlib.pyplot as plt
import piv_image
import pdb


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
    for iter_ in range(1, settings['n_iter_main'] + 1):
        print("Starting main iteration, {}".format(iter_))

        # calculate spacing and create sample grid
        print("Calculating WS and spacing")
        WS = WS_for_iter(iter_, settings)
        print("WS: {}".format(WS))
        h = max(1, round((1 - settings['WOR']) * WS))

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
        # if iter_ == 2:
        #     pdb.set_trace()
        #     u, v = dist.interp_to_densepred(settings['interp'], img_def.dim)
        #     dp = dense_predictor.DensePredictor(u, v, img_def.mask)
        #     dp.plot_displacement_field()

        if settings['vec_val'] is not None:
            print("validate vectors")
            dist.validation_NMT_8NN()

        print("interpolating")
        u, v = dist.interp_to_densepred(settings['interp'], img_def.dim)
        dp = dense_predictor.DensePredictor(u, v, img_def.mask)

        print("deforming image")
        img_def = img.deform_image(dp)

    # dp.plot_displacement_field()


def WS_for_iter(iter_, settings):
    """
    Returns the WS to be used for iteration iter_ for the current settings

    The window size is calculated by finding what reduction factor (RF) would
    need to be applied to settings['init_WS'], ['n_iter_main'] times
    such that at the end

    WS = ['init_WS']*(RF^['n_iter_main']) = ['final_WS']

    iter_ = 1 returns ['init_WS']
        UNLESS
        iter_ = 1 and ['n_iter_main'] == 1, which returns ['final_WS']
    iter_ >= ['n_iter_main'] returns ['final_WS']

    Args:
        iter_ (int): The iteration to calculate the WS for.
                     Must be 1 <= iter_ <= n_iter_main + n_iter_ref
        settings (dict): Settings to be used, see 'widim_settings()'

    Returns:
        WS: Returns the WS rounded to the nearest odd integer
    """

    # check inputs for special cases
    if iter_ == 1:
        if settings['n_iter_main'] == 1:
            return settings['final_WS']
        else:
            return settings['init_WS']

    if iter_ >= settings['n_iter_main']:
        return settings['final_WS']

    # now calculate intermediate WS value
    reduction_fact = np.exp(
        np.log(settings['final_WS'] / settings['init_WS'])
        / (settings['n_iter_main'] - 1)
    )
    WS = settings['init_WS'] * (reduction_fact ** (iter_ - 1))

    # return the nearest odd integer
    return utilities.round_to_odd(WS)


def widim_settings(init_WS=97, final_WS=33, WOR=0.5,
                   n_iter_main=3, n_iter_ref=2,
                   vec_val='NMT', interp='struc_cub'):
    """
    Returns a dict with interrogation settings for a widim analysis

    Args:
        init_WS (int, optional): Initial window size, must be odd and
                                 5 <= init_WS <= 245
        final_WS (int, optional): Final window size, must be odd and
                                  5 <= final_WS <= 245
        WOR (float, optional): Window overlap ratio, must be 0 <= WOR < 1
        n_iter_main (int, optional): Number of main iterations, wherein the WS
                                     and spacing will reduce from init_WS to
                                     final_WS
                                     Must be 1 <= n_iter_main <= 10
                                     If the number of main iterations is 1
                                     then the final_WS is ignored
        n_iter_ref (int, optional): Number of refinement iterations, where the
                                    WS and locations remain fixed, however,
                                    subsequent iterations are performed to
                                    improve the solution
                                    Must be 0 <= n_iter_ref <= 10
        vec_val (str, optional): Type of vector validation to perform.
                                 Options: 'NMT'
                                 Default: 'NMT'
        interp (str, optional): Type of interpolation to perform
                                Options: 'struc_lin', 'struc_cub'
                                Default: 'struc_cub'
    """

    # check all the inputs are valid
    # ====== init_WS ====== #
    if int(init_WS) != init_WS:
        raise ValueError("Initial WS must be integer")
    if (init_WS < 5) or (init_WS > 245):
        raise ValueError("Initial WS must be 5 <= WS <= 245")
    if init_WS % 2 != 1:
        raise ValueError("Initial WS must be odd")
    if init_WS < final_WS:
        raise ValueError("Initial WS must be at least as big as final_WS")

    # ====== final_WS ====== #
    if int(final_WS) != final_WS:
        raise ValueError("Final WS must be integer")
    if (final_WS < 5) or (final_WS > 245):
        raise ValueError("Final WS must be 5 <= WS <= 245")
    if final_WS % 2 != 1:
        raise ValueError("Final WS must be odd")

    # ====== WOR ====== #
    if WOR < 0:
        raise ValueError("WOR must be greater than 0")
    if WOR >= 1:
        raise ValueError("WOR must be strictly less than 1")

    # ====== n_iter_main ====== #
    if int(n_iter_main) != n_iter_main:
        raise ValueError("Number of iterations must be integer")
    if n_iter_main < 1:
        raise ValueError("Number of iterations must be at least 1")
    if n_iter_main > 10:
        raise ValueError("Number of main iterations must be at most 10")

    # ====== n_iter_ref ====== #
    if int(n_iter_ref) != n_iter_ref:
        raise ValueError("Number of refinement iterations must be integer")
    if n_iter_ref < 0:
        raise ValueError("Number of refinement iterations must be at least 0")
    if n_iter_ref > 10:
        raise ValueError("Number of refinement iterations must be at most 10")

    # ====== vector validation ====== #
    options_vec_val = ['NMT']
    if not vec_val in options_vec_val:
        raise ValueError("Vector validation method not handled")

    # ====== Interpolation ====== #
    options_interp = ['struc_lin', 'struc_cub']
    if not interp in options_interp:
        raise ValueError("Interpolation method not handled")

    # all settings are now considered valid
    settings = {
        "init_WS": init_WS,
        "final_WS": final_WS,
        "WOR": WOR,
        "n_iter_main": n_iter_main,
        "n_iter_ref": n_iter_ref,
        "vec_val": vec_val,
        "interp": interp,
    }

    return settings


def run_script():
    IA, IB, mask = piv_image.load_image_from_flow_type(22, 1)
    img = piv_image.PIVImage(IA, IB, mask)
    print("here")
    settings = widim_settings(final_WS=15)

    widim(img, settings)


if __name__ == '__main__':
    # load the image
    IA, IB, mask = piv_image.load_image_from_flow_type(22, 1)
    img = piv_image.PIVImage(IA, IB, mask)
    print("here")
    settings = widim_settings(final_WS=15)

    widim(img, settings)
