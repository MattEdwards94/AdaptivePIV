import corr_window
import numpy as np


class Distribution:
    def __init__(self, init_locations=None):
        """
        Initialises the distribution object with 0, 1, or many correlation
        window objects

        Args:
            init_locations ([list] CorrWindow, optional): list of correlation
                                                          windows to initialise
                                                          the distribution with
        """

        # if there is nothing passed, then initialise empty list
        if init_locations is None:
            self.windows = []
        elif type(init_locations) == list:
            # if it is only a single or multiple inputs we need to add
            self.windows = init_locations.copy()
        else:
            self.windows = [init_locations]
