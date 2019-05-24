import corr_window
import numpy as np
import time


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

    def n_windows(self):
        """
        Returns the number of windows currently stored in the distribution
        """
        return len(self.windows)

    def values(self, prop):
        """
        Returns a list of property values from the list of CorrWindows
        corresponding to the requested property 'prop'

        Args:
            prop (str): The property of self.windows to retrieve

        Returns:
            list: list of properties 'prop' from self.windows

        Example:
            >>> import corr_window
            >>> x = [10, 20, 30]
            >>> y = [15, 25, 35]
            >>> WS = [31, 41, 51]
            >>> cwList = []
            >>> for i in range(3)
            >>>     cwList.append(corr_window.CorrWindow(x[i], y[i], WS[i]))
            >>> dist = Distribution(cwList)
            >>> x_vals = dist.values("x")
            >>> print(x_vals)
            ... [10, 20, 30]
        """
        return [cw.__dict__[prop] for cw in self.windows]



if __name__ == '__main__':
    # create long list of corrWindows
    cwList = []
    for i in range(10):
        cwList.append(corr_window.CorrWindow(i, 2 * i, 31))

    dist = Distribution(cwList)
    print(dist.values('x'))
    print(dist.x_locations())
    print(dist.values('d'))

    start = time.time()
    for i in range(100):
        dist.x_locations()
    print("plain list", time.time() - start)

    start = time.time()
    for i in range(100):
        dist.values('x')
    print("plain list", time.time() - start)
