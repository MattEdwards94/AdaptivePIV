import image_info
import scipy.io as sio
import h5py
# import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time


class piv_image:

    def __init__(self, im_info, im_number):
        """reads in the image relating to the details stored in imInfo
        imNumber corresponds to the snapshot in the ensemble
        if imNumber is greater than the total number of images in the ensemble
        then it will try to open the image.
        A warning will be passed if this succeeds informing the user to
        update the image information
        An error will raise if the image does not exist

        Args:
            imInfo (ImageInfo): image_info.ImageInfo() object containing
                                information about the flow type
            imNumber (int): the specific image to load from the ensemble
        """
        # save information about the current image
        self.img_details = im_info
        self.img_number = im_number

        # load filenames including mask
        fnames = im_info.formatted_filenames(im_number)

        # image A
        if fnames[0][-4:] == ".mat":
            try:
                img = sio.loadmat(fnames[0])
                self.IA = img['IA']
                pass
            except NotImplementedError:
                img = h5py.File(fnames[0])
                self.IA = np.array(img['IA'])
        else:
            self.IA = np.asarray(Image.open(fnames[0])).copy()

        # image B
        if fnames[1][-4:] == ".mat":
            try:
                img = sio.loadmat(fnames[1])
                self.IB = img['IB']
                pass
            except NotImplementedError:
                img = h5py.File(fnames[1])
                self.IB = np.array(img['IB'])
        else:
            self.IB = np.asarray(Image.open(fnames[1])).copy()

        # mask
        if fnames[2] == "none":
            self.mask = np.zeros(np.shape(self.IA))
        else:
            self.mask = np.asarray(Image.open(fnames[2])).copy()

    def __repr__(self):
        """returns the representation of the piv_image object
        """
        return "piv_image(image_info.ImageInfo({}), {})".format(
            self.img_details.flowtype, self.img_number)

    def __eq__(self, other):
        """
        Add method to image_info to check for equality

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
        if not isinstance(other, piv_image):
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
    image_info.list_available_flowtypes()
    print('loading image details for BFS')
    img_details = image_info.ImageInfo(22)
    img = piv_image(img_details, 1)
    print(img_details)

    ia, ib, mask = img.get_region(24, 24, 10)
    print(ia)
    print(ib)
    print(mask)

    start = time.time()
    for i in range(0, 10000):
        ia, ib, mask = img.get_region(24, 24, 10)
    end = time.time()
    print(end - start)

    print(repr(img))
    img_2 = eval(repr(img))
    if img_2 == img:
        print("Yay, it worked")

    en = time.time()
    print("Time: {}".format(en - start))

    # print(ia)
    """for ii in range(1, 5):
        img_details = img_info.ImageInfo(ii)
        print(ii)
        img = piv_image(img_details, 1)
        print(img.IA[0][0:10])
    plt.show()"""
