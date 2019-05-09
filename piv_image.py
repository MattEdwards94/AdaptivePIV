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
        """

        if np.shape(IA) != np.shape(IB):
            raise ValueError("shape of IA must match the shape of IB")

        if mask is not None:
            if np.shape(IA) != np.shape(mask):
                raise ValueError("The shape of the mask must match IA and IB")
        else:
            mask = np.zeros(np.shape(IA))

        self.IA = IA
        self.IB = IB
        self.mask = mask

    def __repr__(self):
        """returns the representation of the PIVImage object
        """
        return "piv_image.PIVImage(np.{}, np.{}, np.{})".format(
            np.array_repr(self.IA),
            np.array_repr(self.IB),
            np.array_repr(self.mask))

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
            x_ctr (TYPE): Description
            y_ctr (TYPE): Description
            rad (TYPE): Description

        Returns:
            TYPE: Description
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
    img = piv_image(np.random.rand(55, 55), np.random.rand(55, 55))
    print(img)
    repr(img)

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
