class ImageInfo:

    """Summary
    """

    def __init__(self):
        """initialises
        """
        self.fname_fmt = ""
        self.folder = ""
        self.mask_file = ""
        self.ref_vel_field_fname = ""
        self.max_n_images = 0
        self.img_dim = [0, 0]

    def print_details(self):
        """ Neatly display information about the image type including
            Filename format
            Folder
            Mask file
            Whether there is a known/reference velocity field
            The maximm number of images in the ensemble
            The image dimensions
        """
        print("The filename format: ", self.fname_fmt)
        print("The folder location: ", self.folder)
        print("The mask file: ", self.mask_file)
        print("The ref. vel. field filename: ", self.ref_vel_field_fname)
        print("The number of images in the ensemble: ", self.max_n_images)
        print("The image dimensions: ", self.img_dim)

        def n_cols():
            """returns the number of columns in the image

            Returns:
                INT: number of columns
            """
            return self._img_dim[1]

        def n_rows():
            """returns the number of rows in the image

            Returns:
                INT: number of rows
            """
            return self._img_dim[0]


def bfs():
    """Summary

    Returns:
        TYPE: Description
    """
    img = ImageInfo()
    img.fname_fmt = "test"
    return img


def vor_array():
    """Summary

    Returns:
        TYPE: Description
    """
    img = ImageInfo()
    img.fname_fmt = "test vortex array"
    return img


switcher = {
    0: bfs,
    1: vor_array
}


def get_image_information(flow_type):
    """Summary

    Args:
        flow_type (TYPE): Description

    Returns:
        TYPE: Description
    """
    func = switcher.get(flow_type, "nothing")
    return func()


if __name__ == "__main__":
    obj = get_image_information(0)
    obj.print_details()
    obj = get_image_information(1)
    obj.print_details()
