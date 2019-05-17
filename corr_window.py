import numpy as np


class CorrWindow(object):
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

        self.x = int(x)
        self.y = int(y)
        self.WS = int(WS)
