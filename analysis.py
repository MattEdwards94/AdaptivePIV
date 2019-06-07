import numpy as np
import distribution


def widim():
    """

    """

    pass


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


if __name__ == '__main__':
    pass
