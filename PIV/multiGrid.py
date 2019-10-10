import numpy as np


class GridCell:

    def __init__(self, xIn, yIn):
        """
        Initialises a grid cell based upon the four input coordinates
        defined in xIn and yIn

        Args:
            xIn (TYPE): Description
            yIn (TYPE): Description

        Deleted Parameters:
            xIn(TYPE): Description
            yIn(TYPE): Description
        """

        # sort according to x and then y
        self.x, self.y = (list(x) for x in zip(*sorted(zip(xIn, yIn))))
        self.tier = 0

        # initialise other variables
        self.children = None
        self.parent = None
        self.neighbours = {"north": None,
                           "east": None,
                           "south": None,
                           "west": None,
                           }

    def bl(self):
        """gets the bottom left coordinate of the current cell
        """
        return self.x[0], self.y[0]

    def tr(self):
        """gets the top right coordinates of the corrent cell
        """
        return self.x[2], self.y[2]

    def cellCentre(self):
        """Returns the central pixel of the cell
        """
        xCtr = sum(self.x) / 4
        yCtr = sum(self.y) / 4
        return xCtr, yCtr

    def split(self):
        """
        splits the current cell into 4 new ones.
        returns the new vertices and cells which are created as a result
        each new cell has the same MultiGrid handle as the current cells, and
        the tier level is incremented by 1.
        Throws an error if the split results in non-integer locations

        This does not actually save the cell into a multigrid distribution

        USAGE
           [newVertices, newCells] = obj.split(self)

        """

        xl, yb = self.bl()
        xr, yt = self.tr()
        xc, yc = self.cellCentre()

        newVert = np.array([[xl, xc],
                            [xr, yc],
                            [xc, yb],
                            [xc, yt],
                            [xc, yc]])

        # define the new cells
        nCell = []
        # bottom left
        nCell.append(GridCell([xl, xl, xc, xc], [yb, yc, yb, yc]))
        # bottom right
        nCell.append(GridCell([xc, xc, xr, xr], [yb, yc, yb, yc]))
        # top left
        nCell.append(GridCell([xl, xl, xc, xc], [yc, yt, yc, yt]))
        # top right
        nCell.append(GridCell([xc, xc, xr, xr], [yc, yt, yc, yt]))

        if self.tier == 0:
            self.tier = 1

        for ii, gc in enumerate(nCell):
            gc.tier = self.tier + 1
            gc.parent = self
            gc.position = ii

        self.children = nCell

        return nCell, newVert


class MultiGrid:

    def __init__(self, imgDim, h):
        """
        intialises the multi grid object by creating a grid as
        [x,y] = meshgrid(1:h:imgDim(2), 1:h:imgDim(1));
        Each 2x2 cell is then identified and constructed as a gridCell which
        stores information about its vertices and which cells it is connected
        to.

        Args:
            imgDim (TYPE): Description
            h (TYPE): Description
        """
        xVec = np.arange(0, imgDim[1], h)
        yVec = np.arange(0, imgDim[0], h)
        x, y = np.meshgrid(xVec, yVec)

        nRows, nCols = np.shape(x)
        nRows -= 1
        nCols -= 1
        self.gridCellList = []
        # loop over the x and y coordinates
        for cc, col in enumerate(xVec[1:]):
            for rr, row in enumerate(yVec[1:]):
                xIn = [xVec[cc - 1], xVec[cc - 1], col, col]
                yIn = [yVec[rr - 1], yVec[rr - 1], row, row]
                gc = GridCell(xIn, yIn)
                gc.tier = 1
                self.gridCellList.append(gc)

        # loop through all cells and declare the neighbours
        for cc in range(0, nCols - 1):
            for rr in range(0, nRows - 1):
                id = cc * nRows + rr
                if rr != nRows:
                    # north
                    self.gridCellList[id].neighbours["north"] = \
                        self.gridCellList[id + 1]
                if cc != nCols:
                    # east
                    self.gridCellList[id].neighbours["east"] = \
                        self.gridCellList[id + nRows]
                if rr != 0:
                    # south
                    self.gridCellList[id].neighbours["south"] = \
                        self.gridCellList[id - 1]
                if cc != 0:
                    self.gridCellList[id].neighbours["west"] = \
                        self.gridCellList[id - nRows]

    def addCell(self, gc):
        self.gridCellList.append(gc)

    def splitCell(self, id):
        """Summary

        Args:
            id (TYPE): Description
        """
        newCells, newVertices = self.gridCellList[id].split()

        for nC in newCells:
            self.gridCellList.append(nC)

    def nCells(self):
        """returns the number of cells

        Returns:
            TYPE: Description
        """
        return np.size(self.gridCellList)


if __name__ == "__main__":
    imgDim = [3500, 3500]
    h = 64
    mg = MultiGrid(imgDim, h)

    for counter in range(0, mg.nCells()):
        mg.splitCell(counter)
        # if counter == 5000:
        #     break

    print(mg.nCells())
