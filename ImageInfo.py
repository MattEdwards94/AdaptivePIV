import csv
import itertools


class ImageInfo:

    """Summary

    Attributes:
        label (string): Short descriptor of the image information
        fname_fmt (str): printf format string for the image filename
        folder (string): name of the subfolder within images\\imageDB
        mask_file (str): filename of the mask file
        ref_vel_field_fname (str): filename of the reference vel field if known
        max_n_images (int): number of images in the ensemble
        img_dim (int list): dimensions of the image in pixels
    """

    def __init__(
            self, label, fname_fmt, folder, mask_file,
            ref_vel_field_fname, max_n_images, img_dim):
        self.label = label
        self.fname_fmt = fname_fmt
        self.folder = folder
        self.mask_file = mask_file
        self.ref_vel_field_fname = ref_vel_field_fname
        self.max_n_images = max_n_images
        self.img_dim = img_dim

    def print_details(self):
        """ Neatly display information about the image type
        """
        col_align = 30
        print("Information for".rjust(col_align), "{}".format(self.label))
        print("filename format:".rjust(col_align), "{}".format(self.fname_fmt))
        print("folder location:".rjust(col_align), "{}".format(self.folder))
        print("mask filename:".rjust(col_align), "{}".format(self.mask_file))
        print("Ref. vel. field filename:".rjust(col_align), "{}".format(
            self.ref_vel_field_fname))
        print("number of images in ensemble:".rjust(col_align), "{}".format(
            self.max_n_images))
        print("image dimensions:".rjust(col_align), "{}x{}".format(
            self.n_rows(), self.n_cols()))

    def n_cols(self):
        """returns the number of columns in the image

        Returns:
            INT: number of columns
        """
        return self.img_dim[1]

    def n_rows(self):
        """returns the number of rows in the image

        Returns:
            INT: number of rows
        """
        return self.img_dim[0]


def get_image_information(flow_type):
    """ searches in the database for image details


    """
    root = 'C:/Users/me12288/Local Documents/PhD - Local/'
    path = root + 'images/imageDB/index.csv'
    with open(path) as imageDB:
        all_information = csv.reader(imageDB)
        for row in itertools.islice(all_information, 1, None):
            if int(row[0]) == flow_type:
                return row
    # if here then we have not found the correct row
    raise ValueError("Item not found")


def list_options():
    root = 'C:/Users/me12288/Local Documents/PhD - Local/'
    path = root + 'images/imageDB/index.csv'
    with open(path) as imageDB:
        all_information = csv.reader(imageDB)
        for row in all_information:
            print("{} - {}".format(row[0], row[1]))


def print_all_details():
    root = 'C:/Users/me12288/Local Documents/PhD - Local/'
    path = root + 'images/imageDB/index.csv'
    with open(path) as imageDB:
        all_information = csv.reader(imageDB)
        for row in all_information:
            print(row)


if __name__ == "__main__":
    row = get_image_information(1)
    print("main")
    print(row)
    print_all_details()
