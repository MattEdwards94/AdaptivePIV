import numpy as np
# from numpy.fft import rfft2
# from numpy.random import rand
# import timeit
import time
import math
import pyfftw
import multiprocessing


def correlate(wsa, wsb):
    wsa = wsa - np.mean(wsa)
    # print(wsa)
    wsb = wsb - np.mean(wsb)
    # print(wsb)

    WS = int(np.shape(wsa)[0])
    # print(type(WS))
    rad = int((WS - 1) / 2)
    # print(type(rad))

    # flip wsa vertically and horizontally
    wsa = wsa[::-1, ::-1]
    # print(wsa)

    # find the nearest power of 2
    Nfftx = 2**(math.ceil(np.log2(WS + 10)))
    Nffty = 2**(math.ceil(np.log2(WS + 10)))
    # print(Nfftx)

    ffta = np.fft.fftn(wsa, [Nffty, Nfftx])
    # print(ffta)
    fftb = np.fft.fftn(wsb, [Nffty, Nfftx])
    # print(fftb)

    corrmap = np.real(np.fft.ifftn(ffta * fftb))
    # print(corrmap)
    idx = np.arange(WS) + rad
    corrmap = corrmap[np.ix_(idx, idx)]

    # idx1 = (np.arange(np.shape(corrmap)[0]) + rad) % np.shape(corrmap)[0]
    # idx2 = (np.arange(np.shape(corrmap)[1]) + rad) % np.shape(corrmap)[1]
    # # print(type(idx1))
    # # print(idx1, idx2, sep="\n")

    # corrmap = corrmap[np.ix_(idx1, idx2)]
    # # print(corrmap)
    # corrmap = corrmap[:WS, :WS]

    return corrmap


def correlate_reg_interface(wsa, wsb):
    wsa = wsa - np.mean(wsa)
    wsb = wsb - np.mean(wsb)

    WS = int(np.shape(wsa)[0])
    rad = int((WS - 1) / 2)

    # flip wsa vertically and horizontally
    wsa = wsa[::-1, ::-1]
    # print(wsa)

    # find the nearest power of 2
    wsbf = WS + 10
    Nfftx = 2**(math.ceil(np.log2(wsbf)))

    ffta = pyfftw.interfaces.numpy_fft.fftn(wsa, [Nfftx, Nfftx])
    fftb = pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx])

    corrmap = np.real(np.fft.ifftn(ffta * fftb))
    # select the correct region from the map
    idx = np.arange(WS) + rad
    corrmap = corrmap[np.ix_(idx, idx)]

    return corrmap


def correlate_reg_interface_no_intermediates(wsa, wsb):
    wsa = wsa - np.mean(wsa)
    wsb = wsb - np.mean(wsb)

    WS = int(np.shape(wsa)[0])
    rad = int((WS - 1) / 2)

    # flip wsa vertically and horizontally
    wsa = wsa[::-1, ::-1]
    # print(wsa)

    # find the nearest power of 2
    wsbf = WS + 10
    Nfftx = 2**(math.ceil(np.log2(wsbf)))

    # ffta = pyfftw.interfaces.numpy_fft.fftn(wsa, [Nfftx, Nfftx])
    # fftb = pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx])

    corrmap = np.real(
        np.fft.ifftn(
            pyfftw.interfaces.numpy_fft.fftn(wsa, [Nfftx, Nfftx])
            * pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx])))
    # select the correct region from the map
    idx = np.arange(WS) + rad
    corrmap = corrmap[np.ix_(idx, idx)]

    return corrmap


def correlate_interface_multithread(wsa, wsb, nThreads):
    wsa = wsa - np.mean(wsa)
    wsb = wsb - np.mean(wsb)

    WS = int(np.shape(wsa)[0])
    rad = int((WS - 1) / 2)

    # flip wsa vertically and horizontally
    wsa = wsa[::-1, ::-1]
    # print(wsa)

    # find the nearest power of 2
    wsbf = WS + 10
    Nfftx = 2**(math.ceil(np.log2(wsbf)))

    # ffta = pyfftw.interfaces.numpy_fft.fftn(wsa, [Nfftx, Nfftx])
    # fftb = pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx])

    corrmap = np.real(
        np.fft.ifftn(
            pyfftw.interfaces.numpy_fft.fftn(
                wsa, [Nfftx, Nfftx], threads=nThreads)
            * pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx], threads=nThreads)))
    # select the correct region from the map
    idx = np.arange(WS) + rad
    corrmap = corrmap[np.ix_(idx, idx)]

    return corrmap


def correlate_fast_interface(wsa, wsb):
    wsa = wsa - np.mean(wsa)
    # print(wsa)
    wsb = wsb - np.mean(wsb)
    # print(wsb)

    WS = int(np.shape(wsa)[0])
    # print(type(WS))
    rad = int((WS - 1) / 2)
    # print(type(rad))

    # flip wsa vertically and horizontally
    wsa = wsa[::-1, ::-1]
    # print(wsa)

    # find the nearest power of 2
    wsbf = WS + 10
    Nfftx = 2**(math.ceil(np.log2(wsbf)))
    # Nffty = 2**(math.ceil(np.log2(WS + 10)))
    # print(Nfftx)

    # ffta = pyfftw.interfaces.numpy_fft.fftn(wsa, [Nfftx, Nfftx])
    # fftb = pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx])

    # ffta = np.fft.fftn(wsa, [Nfftx, Nfftx])
    # print(ffta)
    # fftb = np.fft.fftn(wsb, [Nfftx, Nfftx])

    corrmap = np.real(
        np.fft.ifftn(
            pyfftw.interfaces.numpy_fft.fftn(wsa, [Nfftx, Nfftx])
            * pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx])))
    # select the correct region from the map
    idx = np.arange(WS) + rad
    corrmap = corrmap[np.ix_(idx, idx)]

    return corrmap


def correlate_fast_fftw(wsa, wsb):
    wsa = wsa - np.mean(wsa)
    # print(wsa)
    wsb = wsb - np.mean(wsb)
    # print(wsb)

    WS = int(np.shape(wsa)[0])
    # print(type(WS))
    rad = int((WS - 1) / 2)
    # print(type(rad))

    # flip wsa vertically and horizontally
    wsa = wsa[::-1, ::-1]
    # print(wsa)

    # find the nearest power of 2
    wsbf = WS + 10
    Nfftx = 2**(math.ceil(np.log2(wsbf)))
    # Nffty = 2**(math.ceil(np.log2(WS + 10)))
    # print(Nfftx)

    ffta_fftw = pyfftw.zeros_aligned((Nfftx, Nfftx), dtype='complex128')
    a_in = pyfftw.zeros_aligned((Nfftx, Nfftx), dtype='complex128')
    a_in[:WS, :WS] = wsa
    fft = pyfftw.FFTW(a_in, ffta_fftw, axes=(0, 1))
    fft(a_in)

    # set up builder object
    # a = pyfftw.zeros_aligned((Nfftx, Nfftx), dtype='complex128')
    # ffta_obj = pyfftw.builders.fftn(a, [Nfftx, Nfftx])
    b = pyfftw.zeros_aligned((Nfftx, Nfftx), dtype='complex128')
    fftb_obj = pyfftw.builders.fftn(b, [Nfftx, Nfftx])

    # correlate wsa
    # a[:WS, :WS] = wsa
    # ffta_fftw = ffta_obj(a)

    # check similarity to interfaces and to numpy
    ffta_numpy = np.fft.fftn(wsa, [Nfftx, Nfftx])
    ffta_inter = pyfftw.interfaces.numpy_fft.fftn(wsa, [Nfftx, Nfftx])
    # fftw
    print(np.allclose(ffta_fftw, ffta_numpy))
    print(sum(sum(ffta_fftw - ffta_numpy)))
    # interfaces
    print(np.allclose(ffta_inter, ffta_numpy))
    print(sum(sum(ffta_inter - ffta_numpy)))

    # correlate wsb
    b[:WS, :WS] = wsb
    fftb_fftw = fftb_obj(b)

    # # check that it's not affected previous results
    # print("test equal object")
    # print(fftb_fftw is ffta_fftw)

    # # check similarity to interfaces and to numpy
    # fftb_numpy = np.fft.fftn(wsb, [Nfftx, Nfftx])
    # fftb_inter = pyfftw.interfaces.numpy_fft.fftn(wsb, [Nfftx, Nfftx])
    # # fftw
    # print(np.allclose(fftb_fftw, fftb_numpy))
    # print(sum(sum(fftb_fftw - fftb_numpy)))
    # # interfaces
    # print(np.allclose(fftb_inter, fftb_numpy))
    # print(sum(sum(fftb_inter - fftb_numpy)))

    corrmap = np.real(np.fft.ifftn(ffta_fftw * fftb_fftw))
    # select the correct region from the map
    idx = np.arange(WS) + rad
    corrmap = corrmap[np.ix_(idx, idx)]

    return corrmap


if __name__ == "__main__":
    # a = np.array([[0.4571, 0.6287, 0.3928, 0.9002, 0.0667],
    #               [0.9714, 0.7774, 0.5631, 0.3795, 0.8043],
    #               [0.0791, 0.4641, 0.7575, 0.6882, 0.0541],
    #               [0.9391, 0.8940, 0.2530, 0.9154, 0.8031],
    #               [0.2625, 0.4499, 0.8356, 0.5537, 0.9727]])
    print("Working")
    a = np.random.rand(75, 75)
    # print(a)
    b = a[:, ::-1]
    # print(b)

    # check time for various function
    corrmap_base = correlate(a, b)
    s = time.time()
    for i in range(1000):
        corrmap = correlate(a, b)
    print("time reg: {}".format(time.time() - s))
    corrmap_reg_interface = correlate_reg_interface(a, b)
    corrmap_interface_no_inter = correlate_reg_interface_no_intermediates(a, b)

    if np.allclose(corrmap_reg_interface, corrmap_base):
        s = time.time()
        for i in range(1000):
            corrmap = correlate_reg_interface(a, b)
        print("time correlate_reg_interface: {}".format(time.time() - s))
    else:
        print("correlations not the same")

    if np.allclose(corrmap_interface_no_inter, corrmap_base):
        s = time.time()
        for i in range(1000):
            corrmap = correlate_reg_interface_no_intermediates(a, b)
        print("time correlate_reg_interface_no_intermediates: {}".format(
            time.time() - s))
    else:
        print("correlations not the same")

    nThreads = multiprocessing.cpu_count()
    corrmap_multithread = correlate_interface_multithread(a, b, nThreads)
    if np.allclose(corrmap_multithread, corrmap_base):
        s = time.time()
        for i in range(1000):
            corrmap = correlate_interface_multithread(a, b, nThreads)
        print("time correlate_interface_multithread: {}".format(
            time.time() - s))
    else:
        print("correlations not the same")

    pyfftw.interfaces.cache.enable()
    print("enabling cache")
    if np.allclose(corrmap_reg_interface, corrmap_base):
        s = time.time()
        for i in range(1000):
            corrmap = correlate_reg_interface(a, b)
        print("time correlate_reg_interface: {}".format(time.time() - s))
    else:
        print("correlations not the same")

    if np.allclose(corrmap_interface_no_inter, corrmap_base):
        s = time.time()
        for i in range(1000):
            corrmap = correlate_reg_interface_no_intermediates(a, b)
        print("time correlate_reg_interface_no_intermediates: {}".format(
            time.time() - s))
    else:
        print("correlations not the same")

    nThreads = multiprocessing.cpu_count()
    corrmap_multithread = correlate_interface_multithread(a, b, nThreads)
    if np.allclose(corrmap_multithread, corrmap_base):
        s = time.time()
        for i in range(1000):
            corrmap = correlate_interface_multithread(a, b, nThreads)
        print("time correlate_interface_multithread: {}".format(
            time.time() - s))
    else:
        print("correlations not the same")

    # cMapFast = correlate_reg_interface(a, b)
    # cMapFFTW = correlate_fast_fftw(a, b)
    # pyfftw.interfaces.cache.enable()
    # pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    # np.fft = pyfftw.interfaces.numpy_fft

    # print(cMap - cMapFast)

    # if np.all((cMap - cMapFast) < 0.000000001):
    #     print("True")
    # else:
    #     print("false")

    # if np.all((cMap - cMapFFTW) < 0.000000001):
    #     print("True")
    # else:
    #     print("false")

    # print(np.allclose(cMap, cMapFFTW))
    # print(sum(sum(cMapFFTW - cMap)))

    # s = time.time()
    # for i in range(1000):
    #     corrmap = correlate(a, b)
    # print("time reg: {}".format(time.time() - s))

    # s = time.time()
    # for i in range(1000):
    #     corrmap = correlate_fast(a, b)
    # print("time fast: {}".format(time.time() - s))

    # s = time.time()
    # for i in range(1000):
    #     corrmap = correlate_fast_fftw(a, b)
    # print("time using FFTW: {}".format(time.time() - s))

    # if np.all(fftc == fftb):
    #     print("True")
    # else:
    #     print("false")

    # t = timeit.timeit('fa = numpy.fft.rfftn(a, [64, 64])',
    #                   setup='import numpy; a = numpy.random.rand(10, 10)',
    #                   number=10000)
    # print(t)
    # t = timeit.timeit('fa = numpy.fft.fftn(a, [64, 64])',
    #                   setup='import numpy; a = numpy.random.rand(10, 10)',
    #                   number=10000)
    # print(t)
    # t = timeit.timeit('fa = numpy.fft.fft2(a, [64, 64])',
    #                   setup='import numpy; a = numpy.random.rand(10, 10)',
    #                   number=10000)
    # print(t)
