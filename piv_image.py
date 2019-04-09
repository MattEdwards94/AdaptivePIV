import image_info as img_info


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
        print(im_info)
        print(im_number)


if __name__ == "__main__":
    img_info.print_all_details()

    print('loading image details for BFS')
    img_details = img_info.ImageInfo(1)
    print(img_details)
    img = piv_image(img_details, 1)
