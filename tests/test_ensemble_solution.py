import PIV.ensemble_solution as ens
from PIV.analysis import WidimSettings
import numpy as np
import PIV.dense_predictor as dense_predictor
import PIV.utilities as utilities
import pytest

# ensure that we always look in the right location for data
@pytest.fixture(autouse=True)
def test_data_location(monkeypatch):
    def location():
        return "./PIV/data/"
    monkeypatch.setattr(utilities, "root_path", location)


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
    assert ensR.dp_true is None


def test_ensemble_solution_init_known_disp_field():
    """
    Checks that the settings and flow type are stored, and that the rest of
    the properties are initialised to None

    In this case though, the displacement field is known, so this should 
    be loaded
    """

    settings = WidimSettings()
    flowtype = 39

    ensR = ens.EnsembleSolution(settings, flowtype=39)
    assert ensR.settings == settings
    assert ensR.flowtype == flowtype
    assert ensR.u is ensR.v is ensR.n_images is None
    exp = dense_predictor.DensePredictor.from_dimensions((500, 500), (5, 0))
    assert ensR.dp_true == exp


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


def test_bias():
    """Test the bias returns the mean values minus the true values
    """

    # create an ensemble solution with dummy values
    # using flowtype 39 to simplify here since I know the domain size is 500x500
    ensR = ens.EnsembleSolution(WidimSettings, flowtype=39)
    print(ensR.dim)

    # create some displacement fields to add to the ensemble
    u1, u2 = np.ones(ensR.dim)*1, np.ones(ensR.dim)*2
    u3, u4 = np.ones(ensR.dim)*3, np.ones(ensR.dim)*4
    v1, v2 = np.ones(ensR.dim)*0.5, np.ones(ensR.dim)*1.5
    v3, v4 = np.ones(ensR.dim)*2.5, np.ones(ensR.dim)*4

    exp_u_mean = np.ones(ensR.dim)*2.5
    exp_v_mean = np.ones(ensR.dim)*2.125

    # declare the true displacement field to be 2, 1.5
    ensR.dp_true = dense_predictor.DensePredictor.from_dimensions(ensR.dim,
                                                                  (2, 2.5))

    # add the displacement fields
    dp1 = dense_predictor.DensePredictor(u1, v1)
    dp2 = dense_predictor.DensePredictor(u2, v2)
    dp3 = dense_predictor.DensePredictor(u3, v3)
    dp4 = dense_predictor.DensePredictor(u4, v4)
    ensR.add_displacement_field(dp1)
    ensR.add_displacement_field(dp2)
    ensR.add_displacement_field(dp3)
    ensR.add_displacement_field(dp4)

    exp_u_bias = np.ones(ensR.dim)*0.5
    exp_v_bias = np.ones(ensR.dim)*-0.375
    exp_dp_bias = dense_predictor.DensePredictor(exp_u_bias, exp_v_bias)

    assert exp_dp_bias == ensR.bias


def test_total_err():
    """Check that the total error is equal to the sqrt of the 
    sum of bias and std
    """

    ensR = ens.EnsembleSolution(WidimSettings, flowtype=39)
    print(ensR.dim)

    # create some displacement fields to add to the ensemble
    u1, u2 = np.ones(ensR.dim)*1, np.ones(ensR.dim)*2
    u3, u4 = np.ones(ensR.dim)*3, np.ones(ensR.dim)*4
    v1, v2 = np.ones(ensR.dim)*0.5, np.ones(ensR.dim)*1.5
    v3, v4 = np.ones(ensR.dim)*2.5, np.ones(ensR.dim)*4

    # declare the true displacement field to be 2, 1.5
    ensR.dp_true = dense_predictor.DensePredictor.from_dimensions(ensR.dim,
                                                                  (2, 2.5))
    # add the displacement fields
    ensR.add_displacement_field(dense_predictor.DensePredictor(u1, v1))
    ensR.add_displacement_field(dense_predictor.DensePredictor(u2, v2))
    ensR.add_displacement_field(dense_predictor.DensePredictor(u3, v3))
    ensR.add_displacement_field(dense_predictor.DensePredictor(u4, v4))

    tot_sq_exp = ensR.bias*ensR.bias + ensR.std*ensR.std
    print(tot_sq_exp.v)
    print((ensR.tot_err*ensR.tot_err).v)
    assert np.allclose(tot_sq_exp.u, (ensR.tot_err*ensR.tot_err).u)
    assert np.allclose(tot_sq_exp.v, (ensR.tot_err*ensR.tot_err).v)
