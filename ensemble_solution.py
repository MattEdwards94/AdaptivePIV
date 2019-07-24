import utilities
import numpy as np


class EnsembleSolution():

    """

    """

    def __init__(self, settings, flowtype):
        """

        """

        self.setting = settings
        self.flowtype = flowtype

        self.u = None
        self.v = None

        self.n_images = None

    def add_displacement_field(self, dp):

        if self.n_images is None:
            # initialise mean and var calculators
            self.u = utilities.MeanAndVarCalculator(dp.u)
            self.v = utilities.MeanAndVarCalculator(dp.v)
            self.n_images = 1
        else:
            self.u.add_values(dp.u)
            self.v.add_values(dp.v)
            self.n_images += 1

    def load_from_file(self, filename):
        pass

    def save_to_file(self, filename):
        pass

    def plotting(self):
        pass
