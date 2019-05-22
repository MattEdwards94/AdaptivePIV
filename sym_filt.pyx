import numpy as np


def sym_exp_filt(double[:] X, int X_len, double C0, double z, int K0, int[:] KVec):

    cdef double[:] out
    out = np.zeros(X_len)
    cdef int k

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
