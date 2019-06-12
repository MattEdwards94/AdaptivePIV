import numpy as np
import piv_image
import dense_predictor
import math
import time
import bottleneck as bn


class CorrWindow:
    """
    A CorrWindow is an object which contains 'information' about a sample
    location, i.e. x, y, WS, u, v, and provides functionality to correlate
    such a location, given an image, densepredictor and settings.

    Does not store any kind of intensity or displacement data otherwise this
    would make everything slower

    Attributes:
        x (int): x location of the window
        y (int): y location of the window
        WS (odd int): size of the correlation window
                      must be odd
                      must be integer
                      Assumes square windows
        rad (int): (WS-1)*0.5
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

        Raises:
            ValueError: If there is odd WS or if x, y, WS are negative
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
        self.u = np.NaN
        self.v = np.NaN
        self.u_pre_validation = np.NaN
        self.v_pre_validation = np.NaN
        self.flag = np.NaN

    def __eq__(self, other):
        """
        Allow for comparing equality between windows
        """

        if not isinstance(other, CorrWindow):
            return NotImplemented

        for s, o in zip(self.__dict__, other.__dict__):
            if not np.all(s == o):
                return False

        return True

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
        wsa = ia - (np.sum(ia[ID]) / np.sum(ID))
        wsb = ib - (np.sum(ib[ID]) / np.sum(ID))

        # set mask pixels to 0
        wsa[mask == 0] = 0
        wsb[mask == 0] = 0

        return wsa, wsb, mask

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
            u: The displacement in the u direction, i.e. the horizontal
               distance from the origin of the window to the largest peak
            v: The displacement in the v direction, i.e. the vertical
               distance from the origin of the window to the largest peak
            SNR: The signal to noise ratio. This is the ratio of the largest
                 peak in the correlation map to the second largest peak

        """

        # get the biggest peak
        i, j = np.unravel_index(corrmap.argmax(), corrmap.shape)

        # catch if the peak is on the edge of the domain
        if (i == 0) or (j == 0) or (i == self.WS - 1) or (j == self.WS - 1):
            u, v, SNR = 0, 0, 1
            return u, v, SNR

        # set values around peak to NaN to find the second largest peak
        R = np.copy(corrmap[i - 1:i + 2, j - 1:j + 2])
        corrmap[i - 1:i + 2, j - 1:j + 2] = np.NaN

        # get the second peak and calculate SNR
        SNR = R[1, 1] / (bn.nanmax(corrmap) + np.spacing(1))
        corrmap[i - 1:i + 2, j - 1:j + 2] = R

        # Get the neighbouring values for the Gaussian fitting
        R *= get_corrwindow_scaling(i, j, self.WS, self.rad)

        if np.min(R) <= 0:
            R += 0.00001 - np.min(R)

        u = j - self.rad + 0.5 * ((np.log(R[1, 0]) - np.log(R[1, 2])) / (
            np.log(R[1, 0]) - 2 * np.log(R[1, 1]) + np.log(R[1, 2])))
        v = i - self.rad + 0.5 * ((np.log(R[0, 1]) - np.log(R[2, 1])) / (
            np.log(R[0, 1]) - 2 * np.log(R[1, 1]) + np.log(R[2, 1])))

        return u, v, SNR

    def correlate(self, img, dp):
        """
        Correlates the img at the location specified by self.x, self.y with a
        window of size self.WS

        Args:
            img (PIVImage): The image intensities with which to be correlated
            dp (DensePredictor): The underlying densepredictor if the image has
                                 previously been deformed

         Returns:
            u: The displacement in the u direction, i.e. the horizontal
               distance from the origin of the window to the largest peak
               combined with the underlying displacement from
               previous iterations
            v: The displacement in the v direction, i.e. the vertical
               distance from the origin of the window to the largest peak
               combined with the underlying displacement from
               previous iterations
            SNR: The signal to noise ratio. This is the ratio of the largest
                 peak in the correlation map to the second largest peak

        """

        # check if the central window location is masked
        if not img.mask[self.y, self.x]:
            self.u, self.v, self.WS = np.nan, np.nan, np.nan
            return np.nan, np.nan, np.nan

        # load the image and mask values and perform the cross correlation
        wsa, wsb, mask = self.prepare_correlation_windows(img)
        corrmap = calculate_correlation_map(wsa, wsb, self.WS, self.rad)

        # find the subpixel displacement from the correlation map
        self.u, self.v, self.SNR = self.get_displacement_from_corrmap(corrmap)

        # combine displacement with predictor
        dpx, dpy, mask = dp.get_region(self.x, self.y, self.rad)
        n_elem = np.sum(mask)
        self.u += (np.sum(dpx[mask == 1]) / n_elem)
        self.v += (np.sum(dpy[mask == 1]) / n_elem)

        return self.u, self.v, self.SNR


def calculate_correlation_map(wsa, wsb, WS, rad):
    """
    Performs cross correlation between the two windows wsa and wsb using fft's

    pads the windows wsa and wsb to a power of two, ensuring at least 10px
    of padding.
    i.e. if the WS is 61, then the correlation will be performed using a window
    padded to 128px

    Args:
        wsa (ndarray): Window intensities from image a
        wsb (ndarray): Window intensities from image b
        WS (int): The size of the window in pixels
        rad (int): (WS-1)/2

    Returns:
        corrmap (ndarry): the correlation map between wsa and wsb
    """
    # wsa needs flipping
    wsa = wsa[::-1, ::-1]

    # find the nearest power of 2 (assuming square windows)
    nPow2 = 2**(math.ceil(np.log2(WS + 10)))

    # perform the correlation
    corrmap = np.real(
        np.fft.ifftn(
            np.fft.fftn(wsa, [nPow2, nPow2])
            * np.fft.fftn(wsb, [nPow2, nPow2])
        )
    )

    # return the correct region
    idx = (np.arange(WS) + rad) % nPow2
    bf = corrmap[idx, :]
    corrmap = bf[:, idx]

    return corrmap


def get_corrwindow_scaling(i, j, WS, rad):
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
        rad (int): (WS-1)*0.5 - to avoid re-calculating

    Returns:
        scale (ndarray): The scaling term to be applied to the correlation map
                         scale has shape (3, 3) where the central value
                         corresponds to the pixel containing the peak of the
                         correlation map
    """

    # work out weighting factors for correcting FFT bias
    # (See Raffel pg. 162-162)
    y_val = WS - np.abs(np.array([rad - i + 1, rad - i, rad - i - 1]))
    x_val = WS - np.abs(np.array([rad - j + 1, rad - j, rad - j - 1]))

    return (WS * WS) / (x_val * y_val[:, np.newaxis])


def corrWindow_list(x, y, WS):
    """
    Creates a corrWindow object for each location in x, y, with window size WS

    If WS is a scalar int, then all windows will be given the same size
    If not, WS must be the same length as the input

    Args:
        x (list, int): The x location of the windows
        y (list, int): The y location of the windows
        WS (list, odd int): The window sizes
    """

    if isinstance(WS, int):
        WS = [WS] * len(x)

    cwList = list(map(CorrWindow, x, y, WS))

    return cwList


if __name__ == '__main__':
    IA, IB, mask = piv_image.load_image_from_flow_type(22, 1)
    img = piv_image.PIVImage(IA, IB, mask)
    dp = dense_predictor.DensePredictor(
        np.zeros(img.img_dim), np.zeros(img.img_dim))

    x, y, WS = 50, 50, 33
    arr = []
    for j in range(10):
        start = time.time()
        for i in range(750):
            u, v, snr = CorrWindow(x, y, WS).correlate(img, dp)
        end = time.time()
        arr.append(end - start)
    print(arr)
    print(np.mean(arr))

    print(u, v, snr)
    print("-3.612, 3.4152, 1.3691")
