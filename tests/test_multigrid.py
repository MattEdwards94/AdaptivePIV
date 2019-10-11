import pytest
import numpy as np
import PIV.multiGrid as mg


def test_init_multigrid():
    """Tests the initialisation of an AMG

    This should create a grid with regular spacing, and each four points of a
    square should represent a cell
    """

    img_dim = (500, 750)
    h = 64

    # expected grid
    xv, yv = (np.arange(0, img_dim[1], h),
              np.arange(0, img_dim[0], h))
    x_exp, y_exp = np.meshgrid(xv, yv)

    # actual
    amg = mg.MultiGrid(img_dim, h, WS=127)
    x_act, y_act = amg.x, amg.y

    assert np.allclose(x_act, x_exp.ravel())
    assert np.allclose(y_act, y_exp.ravel())
