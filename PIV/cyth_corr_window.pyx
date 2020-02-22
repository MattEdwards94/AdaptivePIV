# cython: language_level=3
import numpy as np
from libc.math cimport log, abs
cimport numpy as np
import math
import cython
from cpython cimport array
import bottleneck as bn
import matplotlib.pyplot as plt


@cython.boundscheck(False)
cdef get_corrwindow_scaling(int i, int j, int WS, int rad):
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

    cdef int ii, jj
    cdef double WS2 = WS * WS
    cdef double scale[3][3]
    for ii in range(3):
        for jj in range(3):
            scale[ii][jj] = WS2 / ((WS - abs(rad - (j - 1 + jj)))
                                   * (WS - abs(rad - (i - 1 + ii))))

    return scale


@cython.boundscheck(False)
def get_displacement_from_corrmap(double[:, :] corrmap, int WS, int rad):
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

    cdef int i, j
    cdef double u, v, SNR
    cdef double R[3][3]

    # get the biggest peak
    cdef int ii, jj
    cdef double max_val = 0
    for ii in range(WS):
        for jj in range(WS):
            if corrmap[ii, jj] > max_val:
                max_val = corrmap[ii,jj]
                i, j = ii, jj

    # catch if the peak is on the edge of the domain
    if (i == 0) or (j == 0) or (i == WS - 1) or (j == WS - 1):
        u, v, SNR = 0, 0, 1
        return u, v, SNR

    # get the second largest value in the domain
    cdef double max_val2 = 0
    for ii in range(WS):
        for jj in range(WS):
            if ii in [i-1, i, i+1] and jj in [j-1, j, j+1]:
                continue
            if corrmap[ii, jj] > max_val2:
                max_val2 = corrmap[ii,jj]

    SNR = max_val / (max_val2 + 1e-15)

    # Get the neighbouring values for the Gaussian fitting
    scale = get_corrwindow_scaling(i, j, WS, rad)
    cdef int flag
    cdef double min_R = 0
    for ii in range(3):
        for jj in range(3):
            R[ii][jj] = corrmap[i+ii-1, j+jj-1]*scale[ii][jj]
            if R[ii][jj] <= min_R:
                flag = 1
                min_R = R[ii][jj]

    if flag == 1:
        for ii in range(3):
            for jj in range(3):
                R[ii][jj] += 0.00001 - min_R

    for ii in range(3):
        for jj in range(3):
            R[ii][jj] = log(R[ii][jj])

    u = j - rad + 0.5 * ((R[1][0] - R[1][2]) /
                         (R[1][0] - 2 * R[1][1] + R[1][2]))
    v = i - rad + 0.5 * ((R[0][1] - R[2][1]) /
                         (R[0][1] - 2 * R[1][1] + R[2][1]))
    return u, v, SNR
