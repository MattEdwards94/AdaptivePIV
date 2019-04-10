import image_info as img_info
import scipy.io as sio
import h5py
# import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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
        return "piv_image(image_info.ImageObject({}), {})".format(
            self.img_details.flowtype, self.img_number)


if __name__ == "__main__":
    img_info.list_available_flowtypes()
    print('loading image details for BFS')
    img_details = img_info.ImageInfo(1)
    print(img_details)
    # img = piv_image(img_details, 1)

    img_details = img_info.ImageInfo(19)
    print(img_details)

    for ii in range(1, 27):
        img_details = img_info.ImageInfo(ii)
        print(ii)
        img = piv_image(img_details, 1)
        print(img.IA[0][0:10])
        img = piv_image(img_details, img_details.n_images)
        print(img.IA[0][0:10])
    plt.show()
