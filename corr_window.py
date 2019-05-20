import numpy as np
import piv_image
import math


class CorrWindow:
    """
    A CorrWindow is an object which contains 'information' about a sample
    location, i.e. x, y, WS, u, v, and provides functionality to correlate
    such a location, given an image, densepredictor and settings.

    Does not store any kind of intensity or displacement data otherwise this
    would make everything slower
    """

    def __init__(self, x, y, WS):
        """
        Initialises the CorrWindow object based upon the location in space and
        a specified window size

        Args:
            x (int): x location of the window
            y (int): y location of the window
            WS (odd int): size of the correlation window
                          must be odd
                          must be integer
                          Assumes square windows
        """

        # check that WS is odd
        if not WS % 2 == 1:
            raise ValueError("Even sized windows are not allowed")

        # check that negative values are caught
        if x < 0:
            raise ValueError("x must be positive")
        if y < 0:
            raise ValueError("y must be positive")
        if WS < 0:
            raise ValueError("WS must be positive")

        self.x = int(x)
        self.y = int(y)
        self.WS = int(WS)
        self.rad = int((WS - 1) * 0.5)

    def prepare_correlation_windows(self, img):
        """
        Extracts the image intensities and mask values around self.x, self.y

        Prepares the image by subtracting the mean of the non-masked pixels
        Also sets the masked pixels to 0

        Args:
            img (PIVImage): The piv image pair to be analysed

        Returns:
            wsa: image intensities from img.IA around (self.x, self.y), with
                 the mean of wsa subtracted and masked values set to 0
            wsb: image intensities from img.IB around (self.x, self.y), with
                 the mean of wsb subtracted and masked values set to 0
            mask: mask flag values from img.mask around (self.x, self.y)
        """

        # get the raw image intensities
        ia, ib, mask = img.get_region(self.x, self.y, self.rad)

        # get index values where the image is valid
        ID = mask == 1

        # subtract the mean values from the intensities
        wsa = ia - np.mean(ia[ID])
        wsb = ib - np.mean(ib[ID])

        # set mask pixels to 0
        wsa[mask == 0] = 0
        wsb[mask == 0] = 0

        return wsa, wsb, mask

