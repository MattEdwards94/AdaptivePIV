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


@pytest.fixture
def mock_grid():
    img_dim, spacing = (193, 129), 64
    return mg.Grid(img_dim, spacing)


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

    assert amg.cells[0].north == None
    assert amg.cells[0].east == None
    assert amg.cells[0].south == None
    assert amg.cells[0].west == None


def test_multigrid_calculates_neighbours_correctly():
    """ Check that each cell has it's neighbours correctly added
    """

    # create a grid which will result in 9 cells
    h = 64
    img_dim = (3 * h + 1, 3 * h + 1)
    amg = mg.MultiGrid(img_dim, h, WS=127)

    # check that each cell has the expected neighbours
    print(amg.n_cells)

    # expected neieghbours left to right, bottom to top
    cells = [{"north": amg.cells[3], "east": amg.cells[1], "south": None, "west": None},  # bl
             {"north": amg.cells[4], "east": amg.cells[2],
                 "south": None, "west": amg.cells[0]},  # bm
             {"north": amg.cells[5], "east": None,
                 "south": None, "west": amg.cells[1]},  # br
             {"north": amg.cells[6], "east": amg.cells[4],
                 "south": amg.cells[0], "west": None},  # ml
             {"north": amg.cells[7], "east": amg.cells[5],
                 "south": amg.cells[1], "west": amg.cells[3]},  # mm
             {"north": amg.cells[8], "east": None,
                 "south": amg.cells[2], "west": amg.cells[4]},  # mr
             # tl
             {"north": None, "east": amg.cells[7],
                 "south": amg.cells[3], "west": None},
             # tm
             {"north": None,
                 "east": amg.cells[8], "south": amg.cells[4], "west": amg.cells[6]},
             {"north": None, "east": None,
                 "south": amg.cells[5], "west": amg.cells[7]},  # tr
             ]

    for ii, (gc, cell) in enumerate(zip(amg.cells, cells)):
        print(ii)
        assert gc.north == cell['north']
        assert gc.east == cell['east']
        assert gc.south == cell['south']
        assert gc.west == cell['west']


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
    assert gc.bl_win is amg.windows[0]
    assert gc.bl_win.x == 0
    assert gc.bl_win.y == 0
    assert gc.br_win is amg.windows[1]
    assert gc.br_win.x == 64
    assert gc.br_win.y == 0
    assert gc.tl_win is amg.windows[2]
    assert gc.tl_win.x == 0
    assert gc.tl_win.y == 64
    assert gc.tr_win is amg.windows[3]
    assert gc.tr_win.x == 64
    assert gc.tr_win.y == 64


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


def test_cell_north(mock_amg):
    """Tests the property 'north' that it returns the neighbour to the north
    """

    assert mock_amg.cells[0].north is mock_amg.cells[3]


def test_set_cell_north(mock_amg):
    """Test that we can set the northerly value of a cell
    """

    # change the neighbour to the north.
    # this is not the correct neighbour
    mock_amg.cells[0].north = mock_amg.cells[1]
    assert mock_amg.cells[0].north == mock_amg.cells[1]


def test_cell_east(mock_amg):
    """Tests the property 'east' that it returns the neighbour to the east
    """

    assert mock_amg.cells[0].east is mock_amg.cells[1]


def test_set_cell_east(mock_amg):
    """Test that we can set the easterly value of a cell
    """

    # change the neighbour to the east.
    # this is not the correct neighbour
    mock_amg.cells[0].east = mock_amg.cells[2]
    assert mock_amg.cells[0].east == mock_amg.cells[2]


def test_cell_south(mock_amg):
    """Tests the property 'south' that it returns the neighbour to the south
    """

    assert mock_amg.cells[4].south is mock_amg.cells[1]


def test_set_cell_south(mock_amg):
    """Test that we can set the southerly value of a cell
    """

    # change the neighbour to the south.
    # this is not the correct neighbour
    mock_amg.cells[4].south = mock_amg.cells[2]
    assert mock_amg.cells[4].south == mock_amg.cells[2]


def test_cell_west(mock_amg):
    """Tests the property 'west' that it returns the neighbour to the west
    """

    assert mock_amg.cells[4].west is mock_amg.cells[3]


def test_set_cell_west(mock_amg):
    """Test that we can set the westerly value of a cell
    """

    # change the neighbour to the west.
    # this is not the correct neighbour
    mock_amg.cells[4].west = mock_amg.cells[2]
    assert mock_amg.cells[4].west == mock_amg.cells[2]


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
    mock_amg.cells[0].bl_win.WS = 73
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


def test_split_cell_sets_new_tier_level(mock_amg):
    """Check that the tier level of the new cells is 1 more than the cell
    being split
    """

    mock_amg.cells[0].split()

    assert mock_amg.cells[-4].tier == 1
    assert mock_amg.cells[-3].tier == 1
    assert mock_amg.cells[-2].tier == 1
    assert mock_amg.cells[-1].tier == 1

    mock_amg.cells[-1].split()
    assert mock_amg.cells[-4].tier == 2
    assert mock_amg.cells[-3].tier == 2
    assert mock_amg.cells[-2].tier == 2
    assert mock_amg.cells[-1].tier == 2


def test_split_adds_children(mock_amg):
    """Check that when splitting a cell, the new cells are stored as children
    of the cell that is split
    """

    mock_amg.cells[0].split()
    assert mock_amg.cells[0].children['bl'] is mock_amg.cells[-4]
    assert mock_amg.cells[0].children['br'] is mock_amg.cells[-3]
    assert mock_amg.cells[0].children['tl'] is mock_amg.cells[-2]
    assert mock_amg.cells[0].children['tr'] is mock_amg.cells[-1]


def test_split_adds_parents(mock_amg):
    """Check that the newly spawned cells have the correct parent updated
    """

    mock_amg.cells[0].split()
    assert mock_amg.cells[-4].parent is mock_amg.cells[0]
    assert mock_amg.cells[-3].parent is mock_amg.cells[0]
    assert mock_amg.cells[-2].parent is mock_amg.cells[0]
    assert mock_amg.cells[-1].parent is mock_amg.cells[0]


def test_split_adds_known_neighbours(mock_amg):
    """test that the easy neighbours are added
    i.e. the bl has the north and east known
    the br has the nort and west known
    """

    mock_amg.cells[4].split()
    # bl
    assert mock_amg.cells[-4].north is mock_amg.cells[-2]
    assert mock_amg.cells[-4].east is mock_amg.cells[-3]

    # br
    assert mock_amg.cells[-3].north is mock_amg.cells[-1]
    assert mock_amg.cells[-3].west is mock_amg.cells[-4]

    # tl
    assert mock_amg.cells[-2].south is mock_amg.cells[-4]
    assert mock_amg.cells[-2].east is mock_amg.cells[-1]

    # tr
    assert mock_amg.cells[-1].south is mock_amg.cells[-3]
    assert mock_amg.cells[-1].west is mock_amg.cells[-2]


def test_has_children_property(mock_amg):
    """Test that has_children returns true or false accordingly
    """

    # split a cell so we can be sure it should have children
    mock_amg.cells[4].split()

    assert mock_amg.cells[4].has_children
    assert not mock_amg.cells[1].has_children
    assert not mock_amg.cells[4].children['bl'].has_children


def test_split_cell_splits_neighbours(mock_amg):
    """When a cell is split, we need the neighbouring cells to also be split
    if they would be 2 levels of fine-ness different
    """

    # split the centre cell in the mock grid
    # this will create 4 more cells at tier 1
    mock_amg.cells[4].split()

    # now split the bottom right of these cells
    # this should force the east and south cells to also be split
    mock_amg.cells[4].children['br'].split()

    assert mock_amg.cells[5].has_children
    assert mock_amg.cells[1].has_children


def test_split_cell_north_sets_neighbours(mock_amg):
    """ If we split a cell, and then split the cell to the north, check that we
    have the correct neighbour information
    """

    mock_amg.cells[4].split()  # middle cell
    mock_amg.cells[7].split()  # north cell

    north = mock_amg.cells[7]
    south = mock_amg.cells[4]

    assert south.children['tl'].north == north.children['bl']
    assert south.children['tr'].north == north.children['br']
    assert north.children['bl'].south == south.children['tl']
    assert north.children['br'].south == south.children['tr']


def test_split_cell_east_sets_neighbours(mock_amg):
    """If we split a cell, and then split the cell to the east, check that we
    have the correct neighbour information
    """

    mock_amg.cells[4].split()  # middle cell
    mock_amg.cells[5].split()  # east cell

    east = mock_amg.cells[5]
    west = mock_amg.cells[4]

    assert west.children['tr'].east == east.children['tl']
    assert west.children['br'].east == east.children['bl']
    assert east.children['tl'].west == west.children['tr']
    assert east.children['bl'].west == west.children['br']


def test_split_cell_south_sets_neighbours(mock_amg):
    """If we split a cell, and then split the cell to the south, check that we
    have the correct neighbour information
    """

    mock_amg.cells[4].split()  # middle cell
    mock_amg.cells[1].split()  # south cell

    south = mock_amg.cells[1]
    north = mock_amg.cells[4]

    assert south.children['tl'].north == north.children['bl']
    assert south.children['tr'].north == north.children['br']
    assert north.children['bl'].south == south.children['tl']
    assert north.children['br'].south == south.children['tr']


def test_split_cell_west_sets_neighbours(mock_amg):
    """If we split a cell, and then split the cell to the west, check that we
    have the correct neighbour information
    """

    mock_amg.cells[4].split()  # middle cell
    mock_amg.cells[3].split()  # west cell

    west = mock_amg.cells[3]
    east = mock_amg.cells[4]

    assert west.children['tr'].east == east.children['tl']
    assert west.children['br'].east == east.children['bl']
    assert east.children['tl'].west == west.children['tr']
    assert east.children['bl'].west == west.children['br']


def test_n_tiers_setting(mock_amg):
    """Checks that the current tier setting is stored in the multigrid
    """

    # check the largest tier when the multigrid is just created, should be 0
    assert mock_amg.max_tier == 0

    # now split a cell and check the max tier
    mock_amg.cells[4].split()
    assert mock_amg.max_tier == 1

    # and same again
    mock_amg.cells[-1].split()
    assert mock_amg.max_tier == 2


def test_Grid_creates_array_space():
    """A grid stores references to the relavent CorrWindows in a grid. 
    For a given image dimension and spacing there can only be a certain number
    of windows fitting in
    """

    # create dummy meshgrid
    img_dim, spacing = (193, 193), 64
    x_vec = np.arange(0, img_dim[1], spacing)
    y_vec = np.arange(0, img_dim[0], spacing)
    xx, yy = np.meshgrid(x_vec, y_vec)

    # create Grid
    g = mg.Grid(img_dim, spacing)

    assert g.ny == len(y_vec)
    assert g.nx == len(x_vec)


def test_Grid_creates_list_of_lists_of_None(mock_grid):
    """Checks that a private array of size [ny, nx] is created
    """
    # note: img_dim, spacing = (193, 129), 64
    # hence we get 4 points in y and 3 points in x
    exp = [[None, None, None], [None, None, None],
           [None, None, None], [None, None, None]]
    assert mock_grid._array == exp


def test_Grid_get_x_vec(mock_grid):
    """we want to be able to get Grid.x_vec to return a vector of x locations
    """

    # we will get the x locations from the first row, so we only need to create
    # 3 windows to test this
    windows = [corr_window.CorrWindow(j, 0, WS=127) for j in range(0, 129, 64)]
    mock_grid._array[0] = windows

    exp = [0, 64, 128]
    assert exp == mock_grid.x_vec


def test_Grid_get_y_vec(mock_grid):
    """we want to be able to get Grid.y_vec to return a vector of x locations
    """

    # we will get the y locations from the first column, so we only need to
    # create 4 windows to test this
    windows = [corr_window.CorrWindow(0, i, WS=127) for i in range(0, 193, 64)]
    for ii in range(4):
        mock_grid._array[ii][0] = windows[ii]

    exp = [0, 64, 128, 192]
    assert exp == mock_grid.y_vec


def test_get_meshgrid(mock_grid):
    """Check that the correct meshgrid is returned

    x_vec should be [0, 64, 128]
    y_vec should be [0, 64, 128, 193]    
    """

    windows = [corr_window.CorrWindow(j, 0, WS=127) for j in range(0, 129, 64)]
    mock_grid._array[0] = windows
    windows = [corr_window.CorrWindow(0, i, WS=127) for i in range(0, 193, 64)]
    for ii in range(1, 4):
        mock_grid._array[ii][0] = windows[ii]

    expx, expy = np.meshgrid([0, 64, 128], [0, 64, 128, 192])
    actx, acty = mock_grid.get_meshgrid()

    assert np.all(actx == expx)
    assert np.all(acty == expy)


def test_get_values_returns_zeros_with_no_windows(mock_grid):
    """Since this is to be used in a multilevel interpolation, we want the 
    default value to be 0, since if there is no window we want the error 
    function to be 0
    """

    exp = np.zeros((mock_grid.ny, mock_grid.nx))
    u, v = mock_grid.get_values()

    assert np.allclose(u, exp)
    assert np.allclose(v, exp)


def test_get_values_with_sample_window_values(mock_grid):
    """Given CorrWindows with non-zero displacement, we want get_values() to 
    return the u and v displacements as part of a grid
    """

    # create an array of random displacements
    randu = np.random.rand(4, 3)
    randv = np.random.rand(4, 3)

    # create the right amount of CorrWindows
    x_vec, y_vec = [0, 64, 128], [0, 64, 128, 192]
    for i in range(mock_grid.ny):
        for j in range(mock_grid.nx):
            tmp = corr_window.CorrWindow(x_vec[j], y_vec[i], WS=127)
            mock_grid._array[i][j] = tmp
            mock_grid._array[i][j].u = randu[i][j]
            mock_grid._array[i][j].v = randv[i][j]

    u, v = mock_grid.get_values()

    assert np.allclose(u, randu)
    assert np.allclose(v, randv)
