import scipy.io as sio
import PIV.utilities as utilities
from PIV.dense_predictor import DensePredictor
from PIV.piv_image import load_mask
import numpy as np


class EnsembleSolution():

    def __init__(self, settings, flowtype):
        """Initialise the ensemble solution by storing the
        settings and flow type

        Will attempt to load the true displacement field upon construction

        Args:
            settings (Settings-like): settings object - e.g. WidimSettings
            flowtype (int): Flowtype identifier, see image_info.all_flow_types()

        """

        self.settings = settings
        self.flowtype = flowtype
        self.mask = load_mask(flowtype)
        self.dim = np.shape(self.mask)

        self.u = None
        self.v = None

        self.n_images = None

        try:
            self.dp_true = DensePredictor.load_true(self.flowtype)
        except ValueError:
            # the file may not exist
            self.dp_true = None

    @property
    def bias(self):
        """Returns bias (m - true) for the displacement field for each pixel

        The measured value is assumed to take the form of 
            x_m = x_t + B + e

        Where B is the bias and e is the random error. 
        As the number of samples in the ensemble increases, e tends towards 0 
        by definition. 
        Therefore, the bias B is equal to the mean of x_m - x_t

        Returns:
            Densepredictor containing the bias of u and v
        """

        if self.dp_true is not None:
            u_bias = self.u.mean - self.dp_true.u
            v_bias = self.v.mean - self.dp_true.v
            return DensePredictor(u_bias, v_bias, self.mask)
        else:
            return None

    @property
    def std(self):
        """Returns standard deviation of the measured displacement values
        about the mean

        Returns:
            Densepredictor containing the standard deviation of u and v
        """

        u_std = np.sqrt(self.u.variance*(self.n_images-1)/self.n_images)
        v_std = np.sqrt(self.v.variance*(self.n_images-1)/self.n_images)
        return DensePredictor(u_std, v_std, self.mask)

    @property
    def tot_err(self):
        """Returns the total error for u and v for every pixelin the domain

        Returns:
            Densepredictor containing the total error of u and v
        """

        # performs operations on both the u and v components
        tot_square = self.bias*self.bias + self.std*self.std

        # since we haven't told numpy how to perform the sqrt for
        # densepredictor, we need to do this separately and combine
        return DensePredictor(np.sqrt(tot_square.u),
                              np.sqrt(tot_square.v),
                              self.mask)

    @property
    def mean(self):
        return dense_predictor.DensePredictor(self.u.mean,
                                              self.v.mean,
                                              self.mask)

    @staticmethod
    def from_file(filename):
        """Loads a previously computed solution from the .mat file

        Parameters
        ----------
        filename : string
            Filename for the solution to be loaded.
        """

        contents = sio.loadmat(filename)
        out = EnsembleSolution(contents["settings"], contents["flowtype"])

        out.u = utilities.MeanAndVarCalculator(contents["u"][0][0][0])
        out.u.S = contents["u"][0][0][1]
        out.u.N = contents["u"][0][0][2]

        out.v = utilities.MeanAndVarCalculator(contents["v"][0][0][0])
        out.v.S = contents["v"][0][0][1]
        out.v.N = contents["v"][0][0][2]

        out.n_images = contents["n_images"]
        out.flowtype = contents["flowtype"]

        return out

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
            self.dim = self.u.dim
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
