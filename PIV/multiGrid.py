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
        self.windows = corr_window.corrWindow_list(xx.ravel(), yy.ravel(), WS)

        # now go through and create the cells, identifying the coordinates for
        # each corner
        self.cells = []
        n_rows, n_cols = np.shape(xx)

        for rr in range(n_rows - 1):
            for cc in range(n_cols - 1):
                bl = rr * n_cols + cc
                br = bl + 1
                tl = bl + n_cols
                tr = tl + 1
                gc = GridCell(self.windows, bl, br, tl, tr)
                gc.tier = 1
                self.cells.append(gc)

    def validation_NMT_8NN(self):
        raise ValueError("Not defined for MultiGrid")

    def interp_to_densepred(self):
        raise ValueError("Not defined for MultiGrid")


class GridCell():
    def __init__(self, cwList, id_bl, id_br, id_tl, id_tr):
        # cwList will act as a pointer to the list of corr windows and hence
        # shouldn't impose too much of a memory overhead.
        self.cwList = cwList
        # print(id_bl, id_br, id_tl, id_tr)
        self.bl = self.cwList[id_bl]
        self.br = self.cwList[id_br]
        self.tl = self.cwList[id_tl]
        self.tr = self.cwList[id_tr]

        self.tier = 0
        self.children, self.parent = None, None
        self.neighbours = {"north": None,
                           "east": None,
                           "south": None,
                           "west": None,
                           }

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
