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
            self.has_mask = True
        else:
            mask = np.zeros(np.shape(IA))
            self.has_mask = False

        self.IA = np.array(IA)
        self.IB = np.array(IB)
        self.mask = np.array(mask)
        self.n_rows = np.shape(IA)[0]
        self.n_cols = np.shape(IA)[1]
        self.img_dim = [self.n_rows, self.n_cols]

    def __eq__(self, other):
        """
        Add method to PIVImage to check for equality

        PIVImage objects are considered equal if IA, IB, and mask all match

        Only compares equality to other PIVImage objects.
        Will return NotImplemented if the classes are not of the same type
            e.g.
            obj1 = PIVImage(IA, IB)
            obj2 = MyClass(a, b)
            obj1 == obj2
                returns NotImplemented

        Args:
            other (PIVImage): The object to be compared against

        Returns:
            Bool: True or False depending on object equality

        Examples:
        >>> obj1 = PIVImage(IA, IB)
        >>> obj2 = PIVImage(IA, IB)
        >>> obj1 == obj2
        ... returns True

        >>> obj3 = PIVImage(IA2, IB2)
        >>> obj3 == obj1
        ... returns False

        >>> obj4 = MyOtherClass(a, b, c)
        >>> obj4 == obj1
        ... returns NotImplemented
        """
        # print(other)
        # print(isinstance(other, PIVImage))
        if not isinstance(other, PIVImage):
            return NotImplemented

        if not np.alltrue(self.IA == other.IA):
            return False

        if not np.alltrue(self.IB == other.IB):
            return False

        if not np.alltrue(self.mask == other.mask):
            return False

        return True

    def get_region(self, x, y, rad):
        """
        Retrieves the pixel intensities for the region requested.
        If the region requested extends beyond the image dimensions, then
        such pixels will be set to 0 intensity.

        Matrix access is base 0 and is ROW major
        -------------
        | 0| 1| 2| 3|
        -------------
        | 4| 5| 6| 7|
        -------------
        | 8| 9|10|11|
        -------------

        Args:
            x (int): x coord of the centre of the region to be extracted
            y (int): y coord of the centre of the region to be extracted
            rad (int): the number of pixels to extend in each directions

        Returns:
            ia (ndarray): Intensity values from the first image in the region
                           [x-rad:x+rad, y-rad:y+rad]
                           np.shape(ia) = ((2*rad+1), (2*rad+1))
            ib (ndarray): Intensity values from the second image in the region
                           [x-rad:x+rad, y-rad:y+rad]
                           np.shape(ib) = ((2*rad+1), (2*rad+1))
            mask (ndarray): Mask flag values in the region
                           [x-rad:x+rad, y-rad:y+rad]
                           np.shape(mask) = ((2*rad+1), (2*rad+1))

        Examples:
        >>> ia, ib, mask = PIVImage.get_region(20, 15, 9)
        >>> np.shape(ia)
        ... (18, 18)

        """

        """
        extract what region we can within the image.
        to do this we need to know where we are with respect to the limits
        of the image
        """
        # if x - rad is < 0, set to 0, if > n_cols, set to n_cols - 1
        # the minus 1 is because of 0 indexing
        left = max(min(x - rad, self.n_cols - 1), 0)
        right = max(min(x + rad, self.n_cols - 1), 0)
        bottom = max(min(y - rad, self.n_rows - 1), 0)
        top = max(min(y + rad, self.n_rows - 1), 0)

        # extract this region out of the images/mask
        # note the +1 is because left:right is not inclusive of right
        ia_tmp = self.IA[bottom:top + 1, left:right + 1]
        ib_tmp = self.IB[bottom:top + 1, left:right + 1]

        # now pad the image with 0's if ctr +- rad overlaps the edge
        pl = max(rad - x, 0)
        pr = max(x + rad - self.n_cols + 1, 0)
        pb = max(rad - y, 0)
        pt = max(y + rad - self.n_rows + 1, 0)
        pad = ((pb, pt), (pl, pr))

        ia = np.pad(ia_tmp, pad, 'constant', constant_values=0)
        ib = np.pad(ib_tmp, pad, 'constant', constant_values=0)
        if self.has_mask:
            mask = np.pad(
                self.mask[bottom:top + 1, left:right + 1], pad,
                'constant', constant_values=0)

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
