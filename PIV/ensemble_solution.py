import scipy.io as sio
import PIV.utilities as utilities


class EnsembleSolution():

    def __init__(self, settings, flowtype):
        """Initialise the ensemble solution by storing the
        settings and flow type

        Args:
            settings (Settings-like): settings object - e.g. WidimSettings
            flowtype (int): Flowtype identifier, see image_info.all_flow_types()

        """

        self.settings = settings
        self.flowtype = flowtype

        self.u = None
        self.v = None

        self.n_images = None

    def add_displacement_field(self, dp):
        """Adds a displacement field to the ensemble solution, updating
        statistics such as the mean, standard deviation, and number of
        images in the ensemble

        Args:
            dp (DensePredictor): Densepredictor object of the displacement field
        """
        if self.n_images is None:
            # initialise mean and var calculators
            self.u = utilities.MeanAndVarCalculator(dp.u)
            self.v = utilities.MeanAndVarCalculator(dp.v)
            self.n_images = 1
        else:
            self.u.add_values(dp.u)
            self.v.add_values(dp.v)
            self.n_images += 1

    def save_to_file(self, filename):
        """Save the solution to a .mat file

        Args:
            filename (string): filename of the location to save the file to
                               will overwrite existing files
                               can be absolute or relative
        """

        # create dictionary
        # print(self.settings)
        out = {"settings": self.settings,
               "flowtype": self.flowtype,
               "u": self.u,
               "v": self.v,
               "n_images": self.n_images}

        # save to matfile
        sio.savemat(filename, out)

    def plotting(self):
        pass


def load_ens_results(self, filename):
    """loads the .mat file and initialises the EnsembleSolution object

    Args:
        filename (string): Filename of the .mat file to load
    """
    pass
