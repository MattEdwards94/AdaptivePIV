import numpy as np
# from numpy.fft import rfft2
# from numpy.random import rand
# import timeit
import time
import math


def correlate(wsa, wsb):
    wsa = wsa - np.mean(wsa)
    wsb = wsb - np.mean(wsb)

    WS = int(np.shape(wsa)[0])
    rad = int((WS - 1) / 2)

    # flip wsa vertically and horizontally
    wsa = wsa[::-1, ::-1]

    # find the nearest power of 2
    nPow2 = 2**(math.ceil(np.log2(WS + 10)))
    corrmap = np.real(
        np.fft.ifftn(
            np.fft.fftn(wsa, [nPow2, nPow2])
            * np.fft.fftn(wsb, [nPow2, nPow2])))

    idx = (np.arange(WS) + rad) % nPow2

    corrmap = corrmap[np.ix_(idx, idx)]

    return corrmap


if __name__ == "__main__":
    # a = np.array([[0.4571, 0.6287, 0.3928, 0.9002, 0.0667],
    #               [0.9714, 0.7774, 0.5631, 0.3795, 0.8043],
    #               [0.0791, 0.4641, 0.7575, 0.6882, 0.0541],
    #               [0.9391, 0.8940, 0.2530, 0.9154, 0.8031],
    #               [0.2625, 0.4499, 0.8356, 0.5537, 0.9727]])
    print("Working")
    a = np.random.rand(55, 55)
    # print(a)
    b = a[:, ::-1]
    # print(b)

    # check time for various function
    corrmap_base = correlate(a, b)
    s = time.time()
    for i in range(5000):
        corrmap = correlate(a, b)
    print("time reg: {}".format(time.time() - s))

    # if np.allclose(corrmap_base, corrmap_inplace):
    #     s = time.time()
    #     for i in range(1000):
    #         corrmap = correlate_inplace(a, b)
    #     print("time correlate_inplace: {}".format(time.time() - s))
    # else:
    #     print("correlations not the same")
