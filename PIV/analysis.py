import numpy as np
import PIV.ensemble_solution as es
import PIV
import matplotlib.pyplot as plt
from PIV.utilities import vprint, WS_for_iter
from .adaptive_unstruc import adaptive_analysis, AdaptSettings
from .widim_algorithm import widim, widim_AIW, WidimSettings
from .adaptive_struc import structured_adaptive_analysis, AdaptStructSettings
from .adaptive_multigrid import amg_refinement, adapt_multi_grid, MultiGridSettings

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


# if __name__ == '__main__':
    # img = PIV.PIVImage.from_flowtype(22, 1)
    # sol = widim(img, WidimSettings())
    # multi_grid_analysis(img)
