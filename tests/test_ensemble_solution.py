import ensemble_solution as ens
from analysis import WidimSettings
import numpy as np
import dense_predictor
import utilities


def test_ensemble_solution_init():
    """
    Checks that the settings and flow type are stored, and that the rest of
    the properties are initialised to None
    """

    settings = WidimSettings()
    flowtype = 1
    ensR = ens.EnsembleSolution(settings, flowtype)
    assert ensR.settings == settings
    assert ensR.flowtype == flowtype
    assert ensR.u is ensR.v is ensR.n_images is None


def test_add_displacement_field():
    """Check that the mean solution is correctly updated
    This is effectively double checking the mean and var calculator, or at
    least it is checking that the mean and var is correctly used
    """

    # create ensemble solution and dummy displacement field
    settings = WidimSettings()
    flowtype = 1
    ensR = ens.EnsembleSolution(settings, flowtype)
    u, v = 2 * np.ones((100, 200)), 3 * np.ones((100, 200))
    dp = dense_predictor.DensePredictor(u, v)

    # add the displacement field to the solution and check it is as expected
    ensR.add_displacement_field(dp)
    u_mean_var = utilities.MeanAndVarCalculator(u)
    v_mean_var = utilities.MeanAndVarCalculator(v)

    assert ensR.u == u_mean_var
    assert ensR.v == v_mean_var
    assert ensR.n_images == 1

    # now add another and check it is correct
    dp = dense_predictor.DensePredictor(2 * u, 2 * v)
    ensR.add_displacement_field(dp)
    u_mean_var.add_values(2 * u)
    v_mean_var.add_values(2 * v)

    assert ensR.u == u_mean_var
    assert ensR.v == v_mean_var
    assert ensR.n_images == 2
