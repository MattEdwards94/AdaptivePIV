import image_info
import scipy.io as sio
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time


class PIVImage:
    """
    Stores the information for a PIV image pair and provides functionality to
    select sub regions/perform pre-processing/image deformation etc.

    Attributes:
        IA (np array): The first image in the pair
        IB (np array): The second image in the pair.
                   Must have the same dimensions as IA
        mask (np array): Image mask indicating regions not to be considered
                     in the analysis
        n_rows(int): The number of rows in the image
        n_cols(int): The number of columns in the image
        img_dim(list): The dimensions of the image [n_rows, n_cols]

    Examples:
        >>>IA = np.random.rand(100, 100)
        >>>IB = np.random.rand(100, 100)
        >>>mask = np.random.randint(0, 2, (100, 100))
        >>>obj = PIVImage(IA, IB, mask)
    """

    def __init__(self, IA, IB, mask=None):
        """
        Stores the two images IA and IB along with the associated mask.

        IA (np array): The first image in the pair
        IB (np array): The second image in the pair.
                       Must have the same dimensions as IA
        mask (np array): Image mask indicating regions not to be considered
                         in the analysis

        Args:
            IA (TYPE): Description
            IB (TYPE): Description
            mask (None, optional): Description

        Raises:
            ValueError: If the shapes of IA, IB, and mask are not the same
        """

        if np.shape(IA) != np.shape(IB):
            raise ValueError("shape of IA must match the shape of IB")

        if mask is not None:
            if np.shape(IA) != np.shape(mask):
                raise ValueError("The shape of the mask must match IA and IB")
        else:
            mask = np.zeros(np.shape(IA))

        self.IA = np.array(IA)
        self.IB = np.array(IB)
        self.mask = np.array(mask)
        self.n_rows = np.shape(IA)[0]
        self.n_cols = np.shape(IA)[1]
        self.img_dim = [self.n_rows, self.n_cols]

    def __eq__(self, other):
        """
        Add method to PIVImage to check for equality

        Allows equality check such as:
            obj1 = MyClass(1)
            obj2 = MyClass(2)
            obj3 = MyClass(1)

            obj1 == obj2
                returns false

            obj1 == obj3
                returns true

            Will return NotImplemted if the classes are not of the same type
            e.g.
            obj1 == OtherClass(1)
                returns NotImplemented

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not isinstance(other, PIVImage):
            return NotImplemented

        for s, o in zip(self.__dict__, other.__dict__):
            if not np.all(s == o):
                return False

        return True

    def get_region(self, x_ctr, y_ctr, rad):
        """retrieves the pixel intensities for the region requested.
        If the region requested extends beyond the image dimensions, then
        such pixels will be set to 0 intensity.

        Args:
            x_ctr (int): The x coord of the centre of the region to be extracted
            y_ctr (int): The y coord of the centre of the region to be extracted
            rad (int): the number of pixels to extend in each directions

        Returns:
            ia (np array): Intensity values from the first image in the region
                           [x_ctr-rad:x_ctr+rad, y_ctr-rad:y_ctr+rad]
                           np.shape(ia) = ((2*rad+1), (2*rad+1))
            ib (np array): Intensity values from the second image in the region
                           [x_ctr-rad:x_ctr+rad, y_ctr-rad:y_ctr+rad]
                           np.shape(ib) = ((2*rad+1), (2*rad+1))
            mask (np array): Mask flag values in the region
                           [x_ctr-rad:x_ctr+rad, y_ctr-rad:y_ctr+rad]
                           np.shape(mask) = ((2*rad+1), (2*rad+1))

        Examples:
        >>> ia, ib, mask = PIVImage.get_region(20, 15, 9)
        >>> np.shape(ia)
        ... (18, 18)

        """

        # initialises the output
        WS = 2 * rad + 1
        ia = np.zeros((WS, WS))
        ib = np.zeros((WS, WS))
        mask = np.zeros((WS, WS))

        # determine the region of the window which lies within the image
        l = max(min(x_ctr - rad - 1, self.img_details.n_cols), 0)
        r = max(min(x_ctr + rad - 1, self.img_details.n_cols), 0)
        b = max(min(y_ctr - rad - 1, self.img_details.n_rows), 0)
        t = max(min(y_ctr + rad - 1, self.img_details.n_rows), 0)

        # now determine where in the output window the data should sit
        # i.e. if the window overlaps over the right hand edge of the image
        # then there are going to be 0's on the rhs of the corr window
        lStart = max(l - (x_ctr - rad), 1)
        rEnd = lStart + (r - l)
        bStart = max(b - (y_ctr - rad), 1)
        tEnd = bStart + t - b

        # now read values
        ia[bStart:tEnd][:, lStart:rEnd] = self.IA[b:t][:, l:r]
        ib[bStart:tEnd][:, lStart:rEnd] = self.IB[b:t][:, l:r]
        if self.img_details.has_mask:
            mask[bStart:tEnd][:, lStart:rEnd] = self.mask[b:t][:, l:r]

        return ia, ib, mask


if __name__ == "__main__":
    img = PIVImage(np.random.rand(55, 55), np.random.rand(55, 55))
    print(img)

    # image_info.list_available_flowtypes()
    # print('loading image details for BFS')
    # img_details = image_info.ImageInfo(22)
    # img = piv_image(img_details, 1)
    # print(img_details)

    # ia, ib, mask = img.get_region(24, 24, 10)
    # print(ia)
    # print(ib)
    # print(mask)

    # start = time.time()
    # for i in range(0, 10000):
    #     ia, ib, mask = img.get_region(24, 24, 10)
    # end = time.time()
    # print(end - start)

    # print(repr(img))
    # img_2 = eval(repr(img))
    # if img_2 == img:
    #     print("Yay, it worked")

    # en = time.time()
    # print("Time: {}".format(en - start))

    # print(ia)
    """for ii in range(1, 5):
        img_details = img_info.ImageInfo(ii)
        print(ii)
        img = piv_image(img_details, 1)
        print(img.IA[0][0:10])
    plt.show()"""
