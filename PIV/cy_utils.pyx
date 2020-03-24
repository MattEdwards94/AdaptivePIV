import numpy as np
cimport numpy as np


cpdef spatial_var(double [:, ::1] f, double [:, ::1] mean, 
                double [:, ::1] mask, double[:, ::1] area,
                int kern_size):
    """Calculates the spatial variance of each pixel"""

    cdef int ii, jj, mm, nn
    cdef int rad = int((kern_size-1)/2)
    cdef int dim_x, dim_y
    cdef int l, r, b, t
    cdef double sum_sq_dev, mn
    dim_y, dim_x = f.shape[0], f.shape[1]
    cdef double [:, :] var = np.empty((dim_y, dim_x))

    cdef double* f1 = &f[0, 0]
    cdef double* mask1 = &mask[0, 0]
    cdef double* mean1 = &mean[0, 0]
    cdef double* area1 = &area[0, 0]
    cdef int ind, ind_in


    for ii in range(dim_y):
        b = max(ii - rad, 0)
        t = min(ii + rad, dim_y-1)
        for jj in range(dim_x):
            ind = ii*dim_x + jj

            if mask1[ind] == 0:
                var[ii, jj] = 0
                continue

            sum_sq_dev = 0
            l = max(jj - rad, 0)
            r = min(jj + rad, dim_x-1)
            mn = mean1[ind]
            for mm in range(b, t+1):
                ind_in = mm * dim_x
                for nn in range(l, r+1):
                    if mask1[ind_in+nn] == 1:
                        sum_sq_dev += ((f1[ind_in+nn] - mn)*
                                       (f1[ind_in+nn] - mn))
                    else:
                        sum_sq_dev += 0
            
            var[ii, jj] = sum_sq_dev / area1[ind]
    
    return var
