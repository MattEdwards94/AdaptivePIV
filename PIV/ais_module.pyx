import numpy as np
cimport numpy as np
import random
from libc.math cimport floor, ceil, acos, atan2, sqrt, pi, cos, sin
from libc.stdlib cimport rand, RAND_MAX

cdef class SummedAreaTable():
    """Creates a summed area table to allow for rapid extraction of the
    summation of a submatrix of an array
    """
    cdef double[:, :] SAT
    cdef public int[2] img_dim

    def __init__(self, double[:, :] IA):
        """Initialises the summed area table for an input array IA

        Arguments:
            IA {ndarray} -- Input array to create the SAT from
        """

        # sum the rows and then the columns
        # self.SAT = np.cumsum(np.cumsum(IA, axis=1), axis=0)
        cdef int i, j, imax, jmax
        imax, jmax = np.shape(IA)
        self.SAT = np.empty((imax, jmax))
        self.SAT[0, 0] = IA[0, 0]
        # first col
        for i in range(1, imax):
            self.SAT[i, 0] = self.SAT[i-1, 0] + IA[i, 0]
        #first row
        for j in range(1, jmax):
            self.SAT[0, j] = self.SAT[0, j-1] + IA[0, j]

        for i in range(1, imax):
            for j in range(1, jmax):
                self.SAT[i, j] = (IA[i, j] + self.SAT[i, j-1] + 
                                  self.SAT[i-1, j] - self.SAT[i-1, j-1])

        self.img_dim[0] = np.shape(IA)[0]
        self.img_dim[1] = np.shape(IA)[1]


    cdef double get_area_sum(self, 
                     int left, int right, 
                     int bottom, int top):
        """Gets the sum of the region defined by left/right/bottom/top

        The sum is inclusive of all pixels defined by l:r:b:t
        This is DIFFERENT to the standard behaviour of numpy indexing. 
        For example:
            The following is the sum of rows 1-9, inclusive, and 
            columns 4-7 inclusive.
            a = np.sum(A[1:10, 4:8])

            For equivalent behaviour using a summed area table
            st = SummedAreaTable(A)
            a = st.get_area_sum(4, 7, 1, 9)


        Arguments:
            left {int} -- The left most coordinate of the region to search
            right {int} -- The rightmost coordinate of the region to search.
                        This must be greater than the left side
            bottom {int} -- The bottom of the region to search
            top {int} -- The top of the region to search. This must be greater
                        than the bottom of the region
        """
        if right < left:
            raise ValueError("The right must be >= left")
        if top < bottom:
            raise ValueError("The top must be >= bottom")

        # bounds check the inputs
        top = min(max(top, 0), self.img_dim[0]-1)
        right = min(max(right, 0), self.img_dim[1]-1)

        # define the square as
        # A -- B
        # |    |
        # |    |
        # C -- D
        # The sum of the region, including B, excluding the rest, is thus:
        # B - A - D + C
        # note that C is added due to it being doubly subtracted by A and D
        # refer to https://en.wikipedia.org/wiki/Summed-area_table for more
        # information
        #
        # also note that if A or C are on the first column, then they should
        # be 0 in the SAT, likewise if C or D are below the first row
        cdef double A, B, C, D
        A = self.SAT[top, left-1] if left > 0 else 0
        B = self.SAT[top, right]
        C = self.SAT[bottom-1, left-1] if left > 0 and bottom > 0 else 0
        D = self.SAT[bottom-1, right] if bottom > 0 else 0

        return (B - A - D + C)

    cdef double get_total_sum(self):
        """Returns the sum of values over the whole domain

        i.e. the top right value
        """
        return self.SAT[-1, -1]

    def fixed_filter_convolution(self, int filt_size):
        """Gets the effective unity weighted fixed convolution of a filter over
        the whole domain. 

        Is equivalent to looping over every pixel and working out the sum within
        a region equal to filter, centered on each pixel. 
        Values outside of the domain are assumed to be 0

        Must be an odd filter size

        Arguments:
            filt_size {int, odd} -- The size of the filter to apply over 
                                    the domain
        """

        if not filt_size % 2:
            raise ValueError("The filter size must be odd")

        cdef int rad = int((filt_size - 1) / 2)

        # using pad in this way shifts the elements of the array, and fills in
        # to the correct size, using the edge value as the fill.
        # this makes it such that the window will effectively sum 0 values
        # outside of the image

        # note that the comments define the direction that we want to move
        # the desired reference pixel in. This is the opposite to the direction
        # that the actual array is moving in.

        # MODIFY THIS CODE WITH CAUTION

        # shift the top right down and to the left, keeping the values
        # at the edges
        tr = np.pad(self.SAT, ((0, rad), (0, rad)),
                    mode='edge')[rad:, rad:]

        # shift the top left down and to the right. Keep the values along the
        # top edge, set new values to 0 along the left edge
        tl = np.pad(self.SAT, ((0, rad), (0, 0)),
                    mode='edge')[rad:, :]
        tl = np.pad(tl, ((0, 0), (rad+1, 0)),
                    mode='constant')[:, :-(rad+1)]

        # shift the bottom right up and to the left. Keep the values along the
        # right hand edge, set new values to 0 along the bottom
        br = np.pad(self.SAT, ((0, 0), (0, rad)),
                    mode='edge')[:, rad:]
        br = np.pad(br, ((rad+1, 0), (0, 0)),
                    mode='constant')[:-(rad+1), :]

        # shift the bottom left up and to the right.
        # Set all new values to 0
        bl = np.pad(self.SAT, ((rad+1, 0), (rad+1, 0)),
                    mode='constant')[:-(rad+1), :-(rad+1)]

        return tr - tl - br + bl

cdef class Disk():
    """
    Class for adaptive incremental stippling to provide functionality for
    determining whether a disk is valid or not
    """
    cdef int x, y
    cdef double r
    cdef list avail_range
    cdef int n_arcs

    def __init__(self, int x, int y, double r):
        """
        Initialise a Disk with a specific location and radius

        Args:
            x (int): Horizontal location in the domain in pixels
            y (int): Vertical location in the domain in pixels
            r (float): Radius of the disk
        """

        self.x, self.y = x, y
        self.r = r
        # a list of arcs
        self.avail_range = [[0, 2*np.pi]]
        self.n_arcs = 1


    cdef is_range_available(self):
        """
        Returns whether or not there is space around the perimeter of the disk
        to place another disk
        """
        return len(self.avail_range) > 0

    cdef double random_avail_angle(self):
        """
        Returns a random angle from within the range of available arcs, as
        defined by self.avail_range
        """
        # rand_arc = random.choice(self.avail_range)
        cdef list rand_arc 
        cdef double rand_val
        rand_arc = self.avail_range[rand()%self.n_arcs]
        rand_val = rand_arc[0] + (rand_arc[1]-rand_arc[0])*rand()/RAND_MAX
        # np.random.uniform(rand_arc[0], rand_arc[1])
        return rand_val

    cdef overlaps_in_buffer(self, double[:, :] buffer, int bf_refine):
        """
        Determines whether the current disk would overlap any existing disk in
        the buffer.

        To improve the accuracy of whether a disk overlaps of not, the buffer
        may have been 'refined', that is, multiple pixels in the buffer_array
        may refer to a single pixel in the 'actual' buffer.

        Args:
            buffer (ndarray): boolean array indicating where disks already exist
            bf_refine (int): Ratio of number of pixels in the buffer array,
                             to the number of pixels in the domain. Effectively,
                             shape(buffer) = bf_refine*(dim_y, dim_x)
        """

        # get the properties of the disk in the buffer array
        cdef int x_bf, y_bf, n_rows_bf, n_cols_bf
        cdef double r_bf
        x_bf, y_bf = self.x*bf_refine, self.y*bf_refine
        r_bf = self.r*bf_refine

        n_rows_bf, n_cols_bf = np.shape(buffer)*bf_refine

        # get the coordinates of the square of size 2r x 2r
        # cdef int l, r, b, t
        # l = max(0, floor(x_bf - r_bf))
        # r = min(n_cols_bf, ceil(x_bf + r_bf) + 1)
        # b = max(0, floor(y_bf - r_bf))
        # t = min(n_rows_bf, ceil(y_bf + r_bf) + 1)

        # select the points in buffer which are within the radius of the disk
        # if any of these points are unity, then the disk overlaps
        cdef int r_bf_int = int(ceil(r_bf))
        cdef double r_bf2 = r_bf*r_bf
        cdef int i, j
        for i in range(-r_bf_int, r_bf_int+1):
            for j in range(-r_bf_int, r_bf_int+1):
                if (i+y_bf >= 0 and i+y_bf < n_rows_bf and 
                        j+x_bf >=0 and j+x_bf < n_cols_bf):
                    if i*i + j*j <= r_bf2:
                        if buffer[i+y_bf, j+x_bf] == 1:
                            return True
        return False

    cdef update_available_range(self, Disk other_disk):
        """
        Adjusts the available range to reflect the presence of a new disk

        Args:
            other_disk (Disk): The new disk being added
        """
        cdef double cos_beta, beta, alpha, two_pi, dx, dy, dist

        # work out angles
        cos_beta = max(-1, min(1, (self.r + other_disk.r)/(4*other_disk.r)))
        beta = acos(cos_beta)

        if beta < 1e-3:
            # if beta is too small, then break out early.
            return

        dx, dy = other_disk.x - self.x, other_disk.y - self.y
        dist = sqrt(dx**2 + dy**2)
        dx, dy = dx/dist, dy/dist
        alpha = atan2(dy, dx)

        # define 2 pi for simplicity later
        two_pi = pi*2

        if alpha < 0:
            alpha += two_pi

        cdef list clippers = []
        cdef double _from, to
        _from, to = alpha - beta, alpha + beta

        if _from >= 0 and to <= two_pi:
            # simple case, from and to is entirely within 0 and 2pi
            clippers.append([_from, to])
        else:
            # clipper crosses 0, so need to split into two clippings
            if _from < 0:
                if to > 0:
                    # if to == 2pi, then we only need one clipper
                    clippers.append([0, to])
                clippers.append([_from + two_pi, two_pi])

            if to > two_pi:
                if _from < two_pi:
                    # see above comment
                    clippers.append([_from, two_pi])
                clippers.append([0, to - two_pi])

        cdef list remaining = []
        cdef int n_arcs = 0
        for clipper in clippers:
            for arc in self.avail_range:
                if arc[0] >= clipper[0] and arc[1] <= clipper[1]:
                    # arc is completely culled, remove
                    continue
                elif arc[1] < clipper[0] or arc[0] > clipper[1]:
                    # untouched
                    remaining.append(arc)
                    n_arcs += 1
                elif (arc[0] <= clipper[0] and
                      arc[1] >= clipper[0] and
                      arc[1] <= clipper[1]):
                    # if the clipper starts within this arc, and the arc ends
                    # within the clipper
                    _from, to = arc[0], clipper[0]
                    remaining.append([_from, to])
                    n_arcs += 1
                elif (arc[0] >= clipper[0] and
                      arc[0] <= clipper[1] and
                      arc[1] >= clipper[1]):
                    # if the arc starts within the clipper, and the arc ends
                    # outside the clipper
                    _from, to = clipper[1], arc[1]
                    remaining.append([_from, to])
                    n_arcs += 1
                else:
                    # clipper is entirely in the arc, split
                    _from, to = arc[0], clipper[0]
                    remaining.append([_from, to])
                    n_arcs += 1
                    _from, to = clipper[1], arc[1]
                    remaining.append([_from, to])
                    n_arcs += 1

        self.avail_range = remaining
        self.n_arcs = n_arcs

    cdef double approximate_local_density(self, 
                                   SummedAreaTable pdf_sat, 
                                   SummedAreaTable mask_sat):
        """
        Returns an estimate of the local pdf density around the disk.

        The pdf is summed in a square centred on the disk with linear edges
        equal the the Disk's diameter

        Args:
            pdf_sat (SummedAreaTable): The pdf as a summed area table
            mask_sat (SummedAreaTable): The mask as a summed area table
        """
        cdef int l, r, b, t
        cdef double pdf_val, mask_val

        if self.r < 1:
            density = pdf_sat.get_area_sum(self.x, self.x, self.y, self.y)
        else:
            l = int(max(0, floor(self.x - self.r)))
            r = int(min(pdf_sat.img_dim[1], ceil(self.x + self.r)))
            b = int(max(0, floor(self.y - self.r)))
            t = int(min(pdf_sat.img_dim[0], ceil(self.y + self.r)))

            pdf_val = pdf_sat.get_area_sum(l, r, b, t)
            mask_val = mask_sat.get_area_sum(l, r, b, t)
            # scale according to how much of the area was masked
            # area of square / non-masked area
            # also scale according to area of cirle in area of square
            # density = pdf_val * (4*self.r**2) / mask_val) * np.pi / 4
            density = pdf_val * self.r**2 * pi / mask_val

        if density == 0:
            density += 1e-6

        return density

    cdef draw_onto_buffer(self, double [:, :] buffer, int bf_refine):
        """
        Draws a binary representation of the disk onto the buffer.

        Pixels which lie within disk.r of the disk centre will be set to 1 in the
        buffer

        Args:
            buffer (ndarray): Current disk buffer with ones indicating the
                              location of existing disks.
                              Will be modified in place
            disk (Disk): The disk to add to the buffer
            bf_refine (int, optional): Ratio of number of pixels in the disk
                                       buffer to the number of pixels in the
                                       domain. Allows for more precise
                                       evaluation of disk overlap at the
                                       expense of computational cost.
                                       Defaults to 1.
        """
        cdef int x_bf, y_bf, n_rows_bf, n_cols_bf
        cdef int l, r, b, t
        cdef double r_bf

        # get the properties of the disk in the buffer array
        x_bf, y_bf = self.x*bf_refine, self.y*bf_refine
        r_bf = self.r*bf_refine

        n_rows_bf, n_cols_bf = np.shape(buffer) * bf_refine

        # get the coordinates of the square of size 2r x 2r
        # l = int(max(0, floor(x_bf - r_bf)))
        # r = int(min(n_cols_bf, ceil(x_bf + r_bf)))
        # b = int(max(0, floor(y_bf - r_bf)))
        # t = int(min(n_rows_bf, ceil(y_bf + r_bf)))


        cdef int r_bf_int = int(ceil(r_bf))
        cdef double r_bf2 = r_bf*r_bf
        cdef int i, j
        for i in range(-r_bf_int, r_bf_int+1):
            for j in range(-r_bf_int, r_bf_int+1):
                if (i+y_bf >= 0 and i+y_bf < n_rows_bf and 
                        j+x_bf >=0 and j+x_bf < n_cols_bf):
                    if i*i + j*j <= r_bf2:
                        buffer[y_bf+i, x_bf+j] = 1


    cdef change_radius(self, Disk Q, double angle, 
                       double K, SummedAreaTable pdf_sat, 
                       SummedAreaTable mask_sat):
        """
        Changes the disks radius such that it contains the desired amount of 
        underlying pdf

        Args:
            Q (Disk): The central disk. The current disk will be varied in size
                      while maintaining contact with the central disk
            angle (float): Random angle along which the new disk is varied in 
                           size. In radians
            K (float): The amount of the underlying pdf to contain
            pdf_sat (SummedAreaTable): SAT representing the pdf function
        """

        cdef double dens, radius_ratio, eps, rn
        cdef int count, limit, n_rows, n_cols
        dens = self.approximate_local_density(pdf_sat, mask_sat)
        radius_ratio = sqrt(K / dens)
        eps, count, limit = 0.001, 0, 13
        n_rows, n_cols = pdf_sat.img_dim

        while abs(radius_ratio-1) > eps and count < limit:
            rn = max(0.5, self.r*radius_ratio)
            self.x = min(max(0, round(Q.x + (rn+Q.r)*cos(angle))), n_cols-1)
            self.y = min(max(0, round(Q.y + (rn+Q.r)*sin(angle))), n_rows-1)
            self.r = rn
            dens = self.approximate_local_density(pdf_sat, mask_sat)
            radius_ratio = sqrt(K/dens)
            count += 1


cpdef AIS(double[:, :] pdf, double[:, :] mask, 
          int n_points, int bf_refine=1, double[:, :] ex_points=None):
    """
    Distributes approximately n_points samples with a local density similar to
    that described by the pdf. Points will not be placed in the masked region.

    Args:
        pdf (ndarray): Probability density function describing the local target
                       target density of the sample distribution
        mask (ndarray): Binary mask indicating where points should not be placed
                        A mask value of 0 indicates that points should ne be
                        placed here.
                        Must have the same dimensions as the input pdf
        n_points (int): Approximate number of samples to place in the domain
        bf_refine (int, optional): Ratio of number of pixels in the disk
                                   buffer to the number of pixels in the domain.
                                   Allows for more precise evaluation of disk
                                   overlap at the expense of computational cost.
                                   Defaults to 1.
        ex_points (2D list int, optional): List of coordinates to seed the
                                           distribution process with. Should
                                           be a 2D array_like object (i.e. a
                                           list or tuple of lists or tuples
                                           containing the x and y location,
                                           alternatively, a 2D numpy array).
                                           If no seed points are given, the
                                           seed point is randomly chosen.
                                           Defaults to None.
    """

    if not np.any(mask):
        raise ValueError("Mask can't be all 0")

    cdef int n_rows, n_cols
    n_rows, n_cols = np.shape(pdf)

    # initialise the queue, output list, and the disk buffer
    cdef list q, out_list
    cdef double[:, :] disk_buf
    q, out_list, disk_buf = [], [], np.zeros((n_rows, n_cols)*bf_refine)

    pdf = np.multiply(pdf, mask)
    # create summed area table for pdf
    cdef SummedAreaTable pdf_sat, mask_sat
    pdf_sat = SummedAreaTable(pdf)
    mask_sat = SummedAreaTable(mask)

    # determine the initial estimate for r1
    cdef double K, r1
    K = pdf_sat.get_total_sum() / n_points
    r1 = sqrt((n_rows*n_cols)/(pi*n_points))

    cdef int xr, yr, count, limit
    cdef Disk D
    cdef double dens, ratioR, eps

    # initialise AIS
    if ex_points is None:
        # create a seed point
        while True:
            xr, yr = np.random.randint(0, n_cols), np.random.randint(0, n_rows)
            if mask[yr, xr]:  # if not masked
                break

        D = Disk(xr, yr, r1)
        dens = D.approximate_local_density(pdf_sat, mask_sat)
        ratioR = sqrt(K/dens)
        eps, count, limit = 0.001, 0, 20

        while abs(ratioR-1) > eps:
            D.r = max(0.5, D.r*ratioR)
            dens = D.approximate_local_density(pdf_sat, mask_sat)
            ratioR = sqrt(K/dens)
            count += 1
            if count > limit:
                break

        # Add the disk to the queue, and add it to the final points list
        q.append(D)
        out_list.append([D.x, D.y])

        # draw the disk onto the buffer, note that disk_buf is updated inplace
        D.draw_onto_buffer(disk_buf, bf_refine)

    else:
        for point in ex_points:
            D = Disk(point[0], point[1], r1*0.25)
            q.append(D)
            D.draw_onto_buffer(disk_buf, bf_refine)
            out_list.append([D.x, D.y])

    cdef Disk Q, P
    cdef int attempts, xn, yn
    # main AIS loop
    while len(q) > 0:
        # get the last added disk
        Q = q.pop()
        attempts, limit = 0, 20

        while Q.is_range_available() and attempts < limit:
            attempts += 1
            # create disk at random angle with init radius r1
            # checking it isn't masked or out of the domain
            alpha = Q.random_avail_angle()
            xn = int(round(Q.x + (Q.r+r1)*cos(alpha)))
            yn = int(round(Q.y + (Q.r+r1)*sin(alpha)))
            if (yn >= n_rows or xn >= n_cols or
                yn < 0 or xn < 0 or
                    mask[yn, xn] == 0):
                continue
            else:
                P = Disk(xn, yn, r1)

            P.change_radius(Q, alpha, K, pdf_sat, mask_sat)

            if ((P.x >= 0 and P.x < n_cols) and (P.y >= 0 and P.y < n_rows)
                    and not P.overlaps_in_buffer(disk_buf, bf_refine)):
                # point is valid, accept it
                q.append(P)
                out_list.append([P.x, P.y])

                P.draw_onto_buffer(disk_buf, bf_refine)
                Q.update_available_range(P)

    return out_list
