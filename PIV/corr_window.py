import matplotlib.pyplot as plt
import numpy as np
import PIV.piv_image as piv_image
import PIV.dense_predictor as dense_predictor
import math
import time
import cyth_corr_window
import numpy.fft._pocketfft_internal as pfi
from numpy.core import zeros, swapaxes


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
        u (double): The horizontal displacement of the window
        v (double): The vertical displacement of the window
        u_pre_validation (double): The horizontal displacement of the window
                                   before validation took place
        v_pre_validation (double): The vertical displacement of the window
                                   before validation took place
        flag (bool): Whether the vector has been validated
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

        self.x = x
        self.y = y
        self.WS = WS
        self.u = np.NaN
        self.v = np.NaN
        self.u_pre_validation = np.NaN
        self.v_pre_validation = np.NaN
        self.flag = np.NaN
        self.is_masked = None

    def __eq__(self, other):
        """
        Allow for comparing equality between windows

        Args:
            other (CorrWindow): The other CorrWindow to be compared to

        Returns:
            Bool: Whether the two CorrWindows match
        """

        if not isinstance(other, CorrWindow):
            return NotImplemented

        for s, o in zip(self.__dict__.values(), other.__dict__.values()):
            if s != o:
                if not np.all(np.isnan((s, o))):
                    return False

        return True

    def __str__(self):
        """Prints the contents of the corrwindow
        """
        return "location: ({},{}), displacement: ({}, {}), WS: {}".format(
            self.x, self.y, self.u, self.v, self.WS)

    @property
    def x(self):
        """
        Returns:
            int: horizontal position of the corr window
        """
        return self._x

    @x.setter
    def x(self, value):
        """Sets the value of x checking it's validity

        Must be integer and >= 0

        Args:
            value (int): x location of the corr window
        """

        if value < 0:
            raise ValueError("x must be positive")
        if int(value) != value:
            raise ValueError("x must be integer")

        self._x = int(value)

    @property
    def y(self):
        """
        Returns:
            int: vertical position of the corr window
        """
        return self._y

    @y.setter
    def y(self, value):
        """Sets the value of y checking it's validity

        Must be integer and >= 0

        Args:
            value (int): y location of the corr window
        """

        if value < 0:
            raise ValueError("y must be positive")
        if int(value) != value:
            raise ValueError("y must be integer")

        self._y = int(value)

    @property
    def WS(self):
        return self._WS

    @WS.setter
    def WS(self, value):
        """Sets the value of WS checking its validity

        Must be positive odd integer

        Args:
            value (int): Size of the correlation window in pixels
        """

        # check that WS is odd - also implicitly checks for integer
        if not value % 2 == 1:
            raise ValueError("Even sized windows are not allowed")

        # check that negative values are caught
        if value < 0:
            raise ValueError("WS must be positive")

        self._WS = int(value)

    @property
    def rad(self):
        """
        Returns:
            int: Radius of the correlation window as (WS-1) * 0.5
        """
        return int((self.WS - 1) * 0.5)

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
        wsa = ia - (np.add.reduce(ia[ID], axis=None) /
                    np.add.reduce(mask, axis=None))
        wsb = ib - (np.add.reduce(ib[ID], axis=None) /
                    np.add.reduce(mask, axis=None))

        # set mask pixels to 0
        wsa[mask == 0] = 0
        wsb[mask == 0] = 0

        return wsa, wsb, mask

    def get_displacement_from_corrmap(self, corrmap):
        """Wrapper for the cython code

        Args:
            corrmap (ndarray): Correlation map
        """
        u, v, SNR = cyth_corr_window.get_disp_from_corrmap(
            corrmap, self.WS, self.rad)

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
            self.is_masked = True
            self.u, self.v, self.SNR = 0, 0, 0
            return self.u, self.v, self.SNR

        self.is_masked = False

        # load the image and mask values and perform the cross correlation
        wsa, wsb, mask = self.prepare_correlation_windows(img)

        corrmap = calculate_correlation_map(wsa, wsb, self.WS, self.rad)

        # find the subpixel displacement from the correlation map
        self.u, self.v, self.SNR = cyth_corr_window.get_disp_from_corrmap(
            corrmap, self.WS, self.rad)
        # print(f"u: {self.u}, v: {self.v}, SNR: {self.SNR}")

        # combine displacement with predictor
        u_avg, v_avg = dp.get_local_avg_disp(self.x, self.y, self.rad)
        self.u += u_avg
        self.v += v_avg

        return self.u, self.v, self.SNR


def plot_regions(wsa, wsb, corrmap):
    plt.figure(1)
    plt.imshow(wsa)
    plt.title("IA")
    plt.figure(2)
    plt.imshow(wsb)
    plt.title("IB")
    plt.figure(3)
    plt.imshow(corrmap)
    plt.title("corrmap")
    plt.show()


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
        ifft2(
            fft2(wsa, [nPow2, nPow2])
            * fft2(wsb, [nPow2, nPow2]), [nPow2, nPow2]
        )
    )

    # return the correct region
    idx = (np.arange(WS) + rad) % nPow2
    bf = corrmap[idx, :]
    corrmap = bf[:, idx]

    return corrmap


def fft2(a, s):
    """Helper function for fft in 2 dimensions

    Is equivalent to  np.fft.fft2 (or np.fft.fftn), but accepts limited 
    arguments allowing for less input checking

    Arguments:
        a {ndarray} -- Array to perform fft on
        s {list, int} -- 1x2 list indicating the dimensions of the calculation
                         of the fft

    Returns:
        ndarray -- The frequency domain version of a
    """
    a = _raw_fft(a, n=s[1], axis=1, is_real=False, is_forward=True, inv_norm=1)
    a = _raw_fft(a, n=s[0], axis=0, is_real=False, is_forward=True, inv_norm=1)
    return a


def ifft2(a, s):
    """Helper function for ifft in 2 dimensions

    Is equivalent to  np.fft.ifft2 (or np.fft.ifftn), but accepts limited 
    arguments allowing for less input checking

    a ~ ifft2(fft2(a))

    Arguments:
        a {ndarray} -- Array to perform ifft on
        s {list, int} -- 1x2 list indicating the dimensions of the calculation
                         of the ifft

    Returns:
        ndarray -- The spatial domain version of a
    """
    a = _raw_fft(a, n=s[1], axis=1, is_real=False,
                 is_forward=False, inv_norm=1/s[1])
    a = _raw_fft(a, n=s[0], axis=0, is_real=False,
                 is_forward=False, inv_norm=1/s[0])
    return a


def _raw_fft(a, n, axis, is_real, is_forward, inv_norm):
    """Caller function for both fft and ifft

    see np.fft._pocketfft.py

    Arguments:
        a {ndarray} -- Array to be calculated upon
        n {int} -- Size of the dimension to calculate fft on (i.e. next pow 2)
        axis {int} -- Axis upon which to calculate fft
        is_forward {bool} -- Whether to calculate fft (True) or ifft (False)
        inv_norm {double} -- normalising value
    """
    if is_forward:
        # pad with 0's
        s = list(a.shape)
        index = [slice(None)]*len(s)
        index[axis] = slice(0, s[axis])
        s[axis] = n
        z = np.zeros(s, a.dtype.char)
        z[tuple(index)] = a
        a = z

    if axis == a.ndim-1:
        r = pfi.execute(a, is_real, is_forward, inv_norm)
    else:
        a = swapaxes(a, axis, -1)
        r = pfi.execute(a, is_real, is_forward, inv_norm)
        r = swapaxes(r, axis, -1)
    return r


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
