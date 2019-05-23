import numpy as np
import math
import cython


def sym_exp_filt(double[:] X, int X_len, double C0, double z, int K0, int[:] KVec):

    cdef double[:] out
    out = np.zeros(X_len)
    cdef int k

    with cython.boundscheck(False):
        # get first element
        for k in range(K0):
            out[0] += (z**(k)) * X[KVec[k]]

        # filter forwards
        for k in range(1, X_len):
            out[k] = X[k] + z * out[k - 1]

        # update last value
        out[X_len - 1] = (2 * out[X_len - 1] - X[X_len - 1]) * C0

        # filter backwards
        for k in range(X_len - 2, -1, -1):
            out[k] = (out[k + 1] - out[k]) * z

    return out


def bs5_int(double[:, :] IA, int n_rows, int n_cols, double[:, :] new_x, double[:, :] new_y):

    cdef double[:, :] IB
    IB = np.zeros((n_rows, n_cols))
    # cdef int n_rows = np.shape(IA)[0]
    # cdef int n_cols = np.shape(IA)[1]
    cdef double c4 = 1.0 / 120.0
    cdef int i, j, ii, jj, cx, cy, xn, yn, rad1, rad2, rad3, rad4
    cdef double w, w2, w3, w4, w5, bf_a
    cdef double[:] wx, wy
    wx = np.zeros(6)
    wy = np.zeros(6)

    for i in range(n_rows):
        for j in range(n_cols):
            # get nearest pixel
            with cython.boundscheck(False):
                xn = < int > new_x[i, j]
                yn = < int > new_y[i, j]

            rad1 = xn - 2
            rad2 = xn + 3
            rad3 = yn - 2
            rad4 = yn + 3

            with cython.boundscheck(False):
                w = new_x[i, j] - xn
            w2 = w * w
            w3 = w2 * w
            w4 = w3 * w
            w5 = w4 * w
            with cython.boundscheck(False):
                wx[0] = c4 * (1 - 5 * w + 10 * w2 - 10 * w3 + 5 * w4 - 1 * w5)
                wx[1] = c4 * (26 - 50 * w + 20 * w2 +
                              20 * w3 - 20 * w4 + 5 * w5)
                wx[2] = c4 * (66 - 60 * w2 + 30 * w4 - 10 * w5)
                wx[3] = c4 * (26 + 50 * w + 20 * w2 -
                              20 * w3 - 20 * w4 + 10 * w5)
                wx[4] = c4 * (1 + 5 * w + 10 * w2 + 10 * w3 + 5 * w4 - 5 * w5)
                wx[5] = c4 * w5

            with cython.boundscheck(False):
                w = new_y[i, j] - yn
            w2 = w * w
            w3 = w2 * w
            w4 = w3 * w
            w5 = w4 * w
            with cython.boundscheck(False):
                wy[0] = c4 * (1 - 5 * w + 10 * w2 - 10 * w3 + 5 * w4 - 1 * w5)
                wy[1] = c4 * (26 - 50 * w + 20 * w2 +
                              20 * w3 - 20 * w4 + 5 * w5)
                wy[2] = c4 * (66 - 60 * w2 + 30 * w4 - 10 * w5)
                wy[3] = c4 * (26 + 50 * w + 20 * w2 -
                              20 * w3 - 20 * w4 + 10 * w5)
                wy[4] = c4 * (1 + 5 * w + 10 * w2 + 10 * w3 + 5 * w4 - 5 * w5)
                wy[5] = c4 * w5

            # now we can interpolate over the stencil
            bf_a = 0
            cy = 0
            for ii in range(rad3 - 1, rad4):  # loop over rows
                # print("ii", ii)
                if ii < 0:
                    ii = -ii
                if ii > n_rows - 1:
                    ii = 2 * (n_rows - 1) - ii
                # print("ii", ii)

                cx = 0
                for jj in range(rad1 - 1, rad2):  # loop over cols
                    # print("jj", jj)
                    # mirror at image boundary
                    if jj < 0:
                        jj = -jj
                    if jj > n_cols - 1:
                        jj = 2 * (n_cols - 1) - jj

                    with cython.boundscheck(False):
                        bf_a += wx[cx] * wy[cy] * IA[ii, jj]

                    cx += 1

                cy += 1
            with cython.boundscheck(False):
                IB[i, j] = bf_a

    return IB
