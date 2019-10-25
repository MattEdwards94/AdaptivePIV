import numpy as np
import PIV.distribution as distribution
import PIV.corr_window as corr_window


class MultiGrid(distribution.Distribution):

    def __init__(self, img_dim, spacing, WS):
        """Defines a multigrid object with an initial grid spacing

        Args:
            img_dim (tuple): The dimensions of the image to be sampled.
                             (n_rows, n_cols)
            spacing (int): The initial spacing between samples
        """

        # create the coordinate grid
        x_vec = np.arange(0, img_dim[1], spacing)
        y_vec = np.arange(0, img_dim[0], spacing)
        xx, yy = np.meshgrid(x_vec, y_vec)

        # turn each of the coordinates into a corrwindow object
        # note that this flattens the grid row by row
        self.windows = corr_window.corrWindow_list(xx.ravel(), yy.ravel(), WS)

        # now go through and create the cells, identifying the coordinates for
        # each corner
        self.cells = []
        # this refers to the number of grid points, there will be 1 less cell
        # in each direction
        n_rows, n_cols = np.shape(xx)
        n_rc, n_cc = n_rows - 1, n_cols - 1  # number of cells each dir

        # loop over each row and column of 'cells'
        # note that cells are 0 index and range does not include the last value
        for rr in range(n_rc):
            for cc in range(n_cc):
                # as above, the grid has been flattened row by row
                # bl, br, tl, tr correspond to the index of the grid point
                # (i.e. corr window) within the list self.windows
                bl = rr * n_cols + cc
                br = bl + 1
                tl = bl + n_cols
                tr = tl + 1
                gc = GridCell(self, bl, br, tl, tr)
                # define this as the first tier
                gc.tier = 0
                self.cells.append(gc)

        # loop over and set the neighbours of each of the cells,
        # we need to do this after, since not all of the cells will exist if
        # we run during the loop
        for rr in range(n_rc):
            for cc in range(n_cc):
                # define the cells neighbours
                # north is cell number + num cells per row, unless on top row
                # east is cell number + 1, unless on right most col
                # south is cell number - num cells per row, unless on bottom row
                # west is cell number - 1 , unless on left most col
                # cell num
                cn = rr * n_cc + cc
                self.cells[cn].north = self.cells[cn +
                                        n_cc] if rr != (n_rc - 1) else None
                self.cells[cn].east = self.cells[cn +
                                         1] if cc != (n_cc - 1) else None
                self.cells[cn].south = self.cells[cn -
                                           n_cc] if rr != 0 else None
                self.cells[cn].west = self.cells[cn -
                                         1] if cc != 0 else None

    @property
    def n_windows(self):
        return len(self.windows)

    @property
    def n_cells(self):
        return len(self.cells)

    def split_all_cells(self):
        """Split all cells, each into 4 new cells
        """

        n = self.n_cells
        for i in range(n):
            self.cells[i].split()

    def validation_NMT_8NN(self):
        raise ValueError("Not defined for MultiGrid")

    def interp_to_densepred(self):
        raise ValueError("Not defined for MultiGrid")


class GridCell():
    def __init__(self, multigrid, id_bl, id_br, id_tl, id_tr):
        # cwList will act as a pointer to the list of corr windows and hence
        # shouldn't impose too much of a memory overhead.
        self.multigrid = multigrid
        self.cw_list = multigrid.windows

        self.id_bl = id_bl
        self.id_br = id_br
        self.id_tl = id_tl
        self.id_tr = id_tr
        self.bl = self.cw_list[id_bl]
        self.br = self.cw_list[id_br]
        self.tl = self.cw_list[id_tl]
        self.tr = self.cw_list[id_tr]

        self.tier = 0
        self.children, self.parent = None, None
        self._north = None
        self._east = None
        self._south = None
        self._west = None

    @property
    def has_children(self):
        return False if self.children is None else True

    @property
    def north(self):
        """Get the neighbour to the north, if there is one

        Returns:
            GridCell: The grid cell immediately to the north, if there is one
        """
        return self._north

    @north.setter
    def north(self, value):
        self._north = value

    @property
    def east(self):
        return self._east

    @east.setter
    def east(self, value):
        self._east = value

    @property
    def south(self):
        return self._south

    @south.setter
    def south(self, value):
        self._south = value

    @property
    def west(self):
        return self._west

    @west.setter
    def west(self, value):
        self._west = value

    def split_neighbs_if_needed(self):
        """This function is intended to be called when a cell is requested to
        be split.

        It will check in the neighbours of the parent cell and
        split them if needed.
        """

        # if the neighbour exists at this level then we don't need to split
        # anything
        if self.north is None:
            # test if the parent's north neighbour exists
            # it might not if it is at a border
            if self.parent.north is not None:
                self.parent.north.split()

        if self.east is None:
            # test if the parent's east neighbour exists
            # it might not if it is at a border
            if self.parent.east is not None:
                self.parent.east.split()

        if self.south is None:
            # test if the parent's south neighbour exists
            # it might not if it is at a border
            if self.parent.south is not None:
                self.parent.south.split()

        if self.west is None:
            # test if the parent's west neighbour exists
            # it might not if it is at a border
            if self.parent.west is not None:
                self.parent.west.split()

    def split(self):
        """Split a cell into 4 child cells. At the same time, update the
        neighbour list of surrounding cells
        """
        # if the cell is tier 0 then we definitely don't need to split neighbs
        if self.tier > 0:
            self.split_neighbs_if_needed()

        # create the new windows at mid-points.
        # CorrWindow will throw error is non-int is passed
        ctr_x, ctr_y = (self.bl.x + self.br.x) / 2, (self.bl.y + self.tl.y) / 2
        left_mid = corr_window.CorrWindow(self.bl.x, ctr_y, self.bl.WS)
        ctr_btm = corr_window.CorrWindow(ctr_x, self.bl.y, self.bl.WS)
        ctr_mid = corr_window.CorrWindow(ctr_x, ctr_y, self.bl.WS)
        ctr_top = corr_window.CorrWindow(ctr_x, self.tl.y, self.bl.WS)
        right_mid = corr_window.CorrWindow(self.br.x, ctr_y, self.bl.WS)
        self.cw_list.append(left_mid)
        self.cw_list.append(ctr_btm)
        self.cw_list.append(ctr_mid)
        self.cw_list.append(ctr_top)
        self.cw_list.append(right_mid)

        # get the ids of the new cw's for adding in the new cells
        lm = self.multigrid.n_windows - 5
        cb = lm + 1
        cm = lm + 2
        ct = lm + 3
        rm = lm + 4

        # now create the cells and add them into the multigrid object
        bl = GridCell(self.multigrid, self.id_bl, cb, lm, cm)
        bl.tier = self.tier + 1
        bl.parent = self
        br = GridCell(self.multigrid, cb, self.id_br, cm, rm)
        br.tier = self.tier + 1
        br.parent = self
        tl = GridCell(self.multigrid, lm, cm, self.id_tl, ct)
        tl.tier = self.tier + 1
        tl.parent = self
        tr = GridCell(self.multigrid, cm, rm, ct, self.id_tr)
        tr.tier = self.tier + 1
        tr.parent = self

        # set known neigbours
        bl.north = tl
        bl.east = br
        br.north = tr
        br.west = bl
        tl.south = bl
        tl.east = tr
        tr.south = br
        tr.west = tl

        # check for neighbours at the same tier level as the current cell
        # check north
        if self.north is not None:
            ney_child = self.north.children
            if ney_child is not None:
                # There are neighbouring cells at the current tier to the north
                # set these cells as the northerly neighbours of the new child
                # cells.
                # also set the southerly neighbours of the northerly cells
                tl.north = ney_child['bl']
                ney_child['bl'].south = tl
                tr.north = ney_child['br']
                ney_child['br'].south = tr

        # check east
        if self.east is not None:
            ney_child = self.east.children
            if ney_child is not None:
                # There are neighbouring cells at the current tier to the east
                # set these cells as the easterly neighbours of the new child
                # cells.
                # also set the westerly neighbours of the easterly cells
                br.east = ney_child['bl']
                ney_child['bl'].west = br
                tr.east = ney_child['tl']
                ney_child['tl'].west = tr

        # check south
        if self.south is not None:
            ney_child = self.south.children
            if ney_child is not None:
                # There are neighbouring cells at the current tier to the south
                # set these cells as the southerly neighbours of the new child
                # cells.
                # also set the northerly neighbours of the southerly cells
                bl.south = ney_child['tl']
                ney_child['tl'].north = bl
                br.south = ney_child['tr']
                ney_child['tr'].north = br

        # check west
        if self.west is not None:
            ney_child = self.west.children
            if ney_child is not None:
                # There are neighbouring cells at the current tier to the west
                # set these cells as the westerly neighbours of the new child
                # cells.
                # also set the easterly neighbours of the westerly cells
                bl.west = ney_child['br']
                ney_child['br'].east = bl
                tl.west = ney_child['tr']
                ney_child['tr'].east = tl

        self.multigrid.cells.append(bl)
        self.multigrid.cells.append(br)
        self.multigrid.cells.append(tl)
        self.multigrid.cells.append(tr)

        self.children = {'bl': bl, 'br': br, 'tl': tl, 'tr': tr}

        # TODO: Check that neighbours are split accordingly

    @property
    def coordinates(self):
        """Return the coordinates of the current cell going anti clockwise
        from the bottom left
        """
        return [(self.bl.x, self.bl.y),
                (self.br.x, self.br.y),
                (self.tl.x, self.tl.y),
                (self.tr.x, self.tr.y), ]

    def print_locations(self):
        print("bl", self.bl)
        print("br", self.br)
        print("tl", self.tl)
        print("tr", self.tr)


if __name__ == "__main__":
    imgDim = [65, 65]
    h = 64
    mg = MultiGrid(imgDim, h, WS=127)

    mg.cells[0].print_locations()

    mg.cwList[1].x = 75

    mg.cells[0].print_locations()

    print(mg.cwList is mg.cells[0].cwList)

    print(mg.cwList[2] is mg.cells[0].tl)
