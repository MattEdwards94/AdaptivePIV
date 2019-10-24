import pytest
import numpy as np
import PIV.multiGrid as mg
import PIV.corr_window as corr_window


@pytest.fixture
def mock_amg():
    """Creates a grid of correlation windows and returns them as a list
    """
    h = 64
    img_dim = (3 * h + 1, 3 * h + 1)
    amg = mg.MultiGrid(img_dim, h, WS=127)
    return amg


def test_init_multigrid():
    """Tests the initialisation of a multiGrid object

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


def test_multigrid_single_cell_has_no_neighbours():
    """Test that if we create a multigrid with only 1 cell that it has
    no neighbours
    """

    # only enough room for 1 cell
    img_dim = (65, 65)
    h = 64
    amg = mg.MultiGrid(img_dim, h, WS=127)

    assert amg.cells[0].neighbours == {"north": None,
                                       "east": None,
                                       "south": None,
                                       "west": None,
                                       }


def test_multigrid_calculates_neighbours_correctly():
    """ Check that each cell has it's neighbours correctly added
    """

    # create a grid which will result in 9 cells
    h = 64
    img_dim = (3 * h + 1, 3 * h + 1)
    amg = mg.MultiGrid(img_dim, h, WS=127)

    # check that each cell has the expected neighbours

    # expected neieghbours left to right, bottom to top
    cells = [{"north": 3, "east": 1, "south": None, "west": None},  # bl
             {"north": 4, "east": 2, "south": None, "west": 0},  # bm
             {"north": 5, "east": None, "south": None, "west": 1},  # br
             {"north": 6, "east": 4, "south": 0, "west": None},  # ml
             {"north": 7, "east": 5, "south": 1, "west": 3},  # mm
             {"north": 8, "east": None, "south": 2, "west": 4},  # mr
             {"north": None, "east": 7, "south": 3, "west": None},  # tl
             {"north": None, "east": 8, "south": 4, "west": 6},  # tm
             {"north": None, "east": None, "south": 5, "west": 7},  # tr
             ]

    for gc, cell in zip(amg.cells, cells):

        assert gc.neighbours == cell


def test_grid_cell_init_stores_cwlist_and_multigrid(mock_amg):
    """Check that the GridCell stores a 'pointer' to the parent multigrid and
    to the correlation window list.
    This allows cells to be split and insert the new windows and cells into the
    parent multigrid without difficulty
    """

    # check that the corr window list in the cell IS the list in the multigrid
    assert mock_amg.cells[0].cw_list is mock_amg.windows

    # check that the multigrid object in the cell IS the parent multigrid
    assert mock_amg.cells[0].multigrid is mock_amg


def test_grid_cell_init_sets_tier_to_0(mock_amg):
    """The tier tells us how many times a cell has been refined. Check that
    it starts of at 0
    """

    for cell in mock_amg.cells:
        assert cell.tier == 0


def test_grid_cell_init_stores_corrWindows():
    """Check that each corrWindow relating to each corner is stored as a
    reference to a single corr window object for each location
    """

    amg = mg.MultiGrid([65, 65], 64, WS=127)

    # get the grid cell
    gc = amg.cells[0]

    # check the locations of the corr windows, and check that it hasn't
    # created a copy
    assert gc.bl is amg.windows[0]
    assert gc.bl.x == 0
    assert gc.bl.y == 0
    assert gc.br is amg.windows[1]
    assert gc.br.x == 64
    assert gc.br.y == 0
    assert gc.tl is amg.windows[2]
    assert gc.tl.x == 0
    assert gc.tl.y == 64
    assert gc.tr is amg.windows[3]
    assert gc.tr.x == 64
    assert gc.tr.y == 64


def test_n_windows(mock_amg):
    """Check the property number of windows

    mock amg should produce a 4x4 grid of points, hence 16 windows
    """
    assert mock_amg.n_windows == 16


def test_n_cells(mock_amg):
    """Check the property number of cells

    mock amg should produce a 4x4 grid of points, hence 9 cells
    """
    assert mock_amg.n_cells == 9


def test_split_cell_creates_four_more_cells(mock_amg):
    """Check that the multigrid has 5 more windows, and that there are 4 more
    cells
    """

    init_n_windows = mock_amg.n_windows
    init_n_cells = mock_amg.n_cells

    mock_amg.cells[0].split()

    assert init_n_windows + 5 == mock_amg.n_windows
    assert init_n_cells + 4 == mock_amg.n_cells


def test_split_cell_throws_error_for_non_int_location():
    """We can only have windows at integer locations so check that if a split
    would result in a non-integer location that an error is raised
    """

    # create grid with an odd spacing
    amg = mg.MultiGrid([64, 64], 63, WS=127)

    # try to split a cell and check it raises an error
    with pytest.raises(ValueError):
        amg.cells[0].split()


def test_split_cell_sets_new_ws_to_bl(mock_amg):
    """In the absence of anything better, set the new windows WS equal to the
    bottom left WS
    """

    # change the bottom left window size (from 127) such that we know it is
    # this window which is being referenced
    mock_amg.cells[0].bl.WS = 73
    mock_amg.cells[0].split()

    for window in mock_amg.windows[-5:]:
        assert window.WS == 73


def test_cell_coordinates(mock_amg):
    """Check that calling coordinates returns a list of tuples of coordinates
    """
    expected = [(0, 0), (64, 0), (0, 64), (64, 64)]
    assert mock_amg.cells[0].coordinates == expected


def test_split_cell_adds_new_windows_correctly(mock_amg):
    """Check that 4 new cells are added and that the locations are as expected

    for mock_amg the spacing is 64, so we expect
    bl cell [(0, 0), (32, 0), (0, 32), (32, 32)]
    br cell [(32, 0), (64, 0), (32, 32), (64, 32)]
    tl cell [(0, 32), (32, 32), (0, 64), (32, 64)]
    tr cell [(32, 32), (64, 32), (32, 64), (64, 64)]
    """

    # check the last 4 cells after splitting the bottom left and compare the
    # coordinate locations
    mock_amg.cells[0].split()

    new_bl = mock_amg.cells[-4]
    assert new_bl.multigrid is mock_amg
    assert new_bl.cw_list is mock_amg.windows
    assert new_bl.coordinates == [(0, 0), (32, 0), (0, 32), (32, 32)]

    new_br = mock_amg.cells[-3]
    assert new_br.multigrid is mock_amg
    assert new_br.cw_list is mock_amg.windows
    assert new_br.coordinates == [(32, 0), (64, 0), (32, 32), (64, 32)]

    new_tl = mock_amg.cells[-2]
    assert new_tl.multigrid is mock_amg
    assert new_tl.cw_list is mock_amg.windows
    assert new_tl.coordinates == [(0, 32), (32, 32), (0, 64), (32, 64)]

    new_tr = mock_amg.cells[-1]
    assert new_tr.multigrid is mock_amg
    assert new_tr.cw_list is mock_amg.windows
    assert new_tr.coordinates == [(32, 32), (64, 32), (32, 64), (64, 64)]
