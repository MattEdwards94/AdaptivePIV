import image_info as img_info
import matplotlib.image as mpimg


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

        # load filenames
        fnames = im_info.formatted_filenames(im_number)

        # open both images
        self.IA = mpimg.imread(fnames[0])
        self.IB = mpimg.imread(fnames[1])

        print(self.IA[0])
        print(self)

    def __repr__(self):
        """returns the representation of the piv_image object
        """
        return "piv_image(image_info.ImageObject({}), {})".format(
            self.img_details.flowtype, self.img_number)


if __name__ == "__main__":
    print('loading image details for BFS')
    img_details = img_info.ImageInfo(1)
    print(img_details)
    img = piv_image(img_details, 1)
