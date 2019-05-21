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

    def get_correlation_map(self, img):
        """
        Obtains the correlation map for the current correlation window for the
        image 'img'

        If the current location is outside of the image passed in, then a
        ValueError is raised

        If the current location is in a region with a mask,
        then NaN is returned

        Args:
            img (PIVImage): The piv image pair to be analysed

        """

        # get prepared window intensities
        wsa, wsb, mask = self.prepare_correlation_windows(img)

        # wsa needs flipping
        wsa = wsa[::-1, ::-1]

        # find the nearest power of 2 (assuming square windows)
        nPow2 = 2**(math.ceil(np.log2(self.WS + 10)))

        # perform the correlation
        corrmap = np.real(
            np.fft.ifftn(
                np.fft.fftn(wsa, [nPow2, nPow2])
                * np.fft.fftn(wsb, [nPow2, nPow2])
            )
        )

        # return the correct region
        idx = (np.arange(self.WS) + self.rad) % nPow2
        corrmap = corrmap[np.ix_(idx, idx)]

        return corrmap

    def get_displacement_from_corrmap(self, corrmap):
        """
        Finds the largest and second largest peaks in the correlation map and
        calculates the SNR ratio

        If the largest peak is on the edge of the domain then u, v, SNR = 0, 0, 1
        Since this is an invalid displacement anyway

        Then performs a subpixel fit around the largest peak to get the local
        displacement

        Args:
            corrmap (ndarray): The correlation map as a numpy ndarray

        Returns:
            TYPE: Description
        """

        # get the biggest peak
        i, j = np.unravel_index(corrmap.argmax(), corrmap.shape)
        val_peak = corrmap[i, j]

        # catch if the peak is on the edge of the domain
        if (i == 0) or (j == 0) or (i == self.WS - 1) or (j == self.WS - 1):
            u, v, SNR = 0, 0, 1
            return u, v, SNR

        # set values around peak to NaN to find the second largest peak
        bf = np.copy(corrmap)
        bf[i - 1:i + 2, j - 1:j + 2] = np.NaN

        # get the second peak and calculate SNR
        val_second_peak = np.nanmax(bf)
        SNR = val_peak / (val_second_peak + np.spacing(1))

        # Get the neighbouring values for the Gaussian fitting
        R = np.copy(corrmap[i - 1:i + 2, j - 1:j + 2])
        scale = get_corrwindow_scaling(i, j, self.WS)
        R /= scale

        if np.min(R) <= 0:
            R += 0.00001 - np.min(R)

        u = j - self.rad + 0.5 * ((np.log(R[1, 0]) - np.log(R[1, 2])) / (
            np.log(R[1, 0]) - 2 * np.log(R[1, 1]) + np.log(R[1, 2])))
        v = i - self.rad + 0.5 * ((np.log(R[0, 1]) - np.log(R[2, 1])) / (
            np.log(R[0, 1]) - 2 * np.log(R[1, 1]) + np.log(R[2, 1])))

        return u, v, SNR


def get_corrwindow_scaling(i, j, WS):
    """
    When correlating two windows, assume one is staying fixed and the other is
    moving. When the windows are half overlapped, then only half of the image
    is contributing to the correlation value at this point
    The amount that contributed to the correlation map at a given point
    therefore depends on the amount of overlap between the two correlation
    windows
    At this origin this will always be unity (1).
    As you move away from the origin, the contribution will decrease.
    Since there is less contributing to the correlation value at this point, we
    need to scale it's influence, otherwise we end up with a bias towards the
    origin
    This can be obtained by convoluting the image sampling function with itself
    This usually is a constant weight and therefore corresponds to a
    rectangular function correlated with itself, which gives a triangular
    function

    Args:
        i (int): index along the first axis (i.e. the row number)
        j (int): index along the second axis (i.e. the column number)
        WS (int): size of the window
    """

    # work out weighting factors for correcting FFT bias
    # (See Raffel pg. 162-162)
    i_adj = np.arange(i - 1, i + 2)
    j_adj = np.arange(j - 1, j + 2)
    rad = int((WS - 1) * 0.5)
    y_val = WS - np.abs(rad - i_adj)
    x_val = WS - np.abs(rad - j_adj)

    scale = x_val * y_val.reshape((3, 1)) / (WS * WS)

    return scale

