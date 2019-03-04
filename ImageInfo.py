class ImageInfo:

    """Summary
    """

    def __init__(self):
        """initialises 
        """
        self._fname_fmt = ""
        self._folder = ""
        self._mask_file = ""
        self._known_vel_field_fname = ""
        self._max_n_images = 0
        self._img_dim = [0, 0]

    def printDetails(self):
        """ Neatly display information about the image type including
            Filename format
            Folder
            Mask file
            Whether there is a known/reference velocity field
            The maximm number of images in the ensemble
            The image dimensions
        """
        print("The filename format: ", self._fname_fmt)
        print("The folder location: ", self._folder)
        print("The mask file: ", self._mask_file)
        print("The reference velocity field location: ", self._known_vel_field_fname)
        print("The number of images in the ensemble: ", self._max_n_images)
        print("The image dimensions: ", self._img_dim)


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
    obj.printDetails()
    obj = get_image_information(1)
    obj.printDetails()
