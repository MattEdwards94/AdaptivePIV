import csv
import itertools
path_to_file_index = 'index.csv'


class ImageInfo:
    """Class containing information relating to images of a single flow type.

    USAGE:
        imgObject = ImageInfo(flowtype)

    Attributes:
        flowtype (int): ID corresponding to a particular flow
        description (string): Short label eg "backwards facing step"
        folder (string): folder location underneath images/imageDB
        filename (string): filename format e.g. a_%05d_%c.tif
        mask_fname (string): filename for the mask if present, or "none"
        vel_field_fname (string): filename for the reference velocity field
        or "none"
        img_dim (string): string containing MxN
        max_n_images (int): how many images are in the ensemble
        is_synthetic (bool): whether the images are synthetic or experimental
        is_time_resolved (bool): whether the images are time resolved or not
    """

    def __init__(self, flowtype):
        """Initialises an image_info object from the data in the CSV file
        who's path is defined by path_to_file_index at the top of
        "image_info.py".

        Args:
            flowtype (integer): ID of the flow to be read in

        """
        # read data from csv file
        row = get_image_information(flowtype)
        self.flowtype = int(row[0])
        self.description = row[1]
        self.folder = row[2]
        self.filename = row[3]
        self.mask_fname = row[4]
        self.vel_field_fname = row[5]
        self.img_dim_text = row[6]
        self.n_images = int(row[7])
        self.is_synthetic = row[8]
        self.is_time_resolved = row[9]

    def __repr__(self):
        """returns the representation of the object,
        i.e. how it is constructed
        """
        return "ImageInfo({})".format(self.flowtype)

    def __str__(self):
        """Returns a textual representation of the object which includes
        information such as the flow type, the description, filename,
        image dimensions, etc... Neatly display information about the image
        type
        """
        col_align = 30
        out = "Flow ID: ".rjust(col_align) + "{}\n"
        out += "Label: ".rjust(col_align) + "{}\n"
        out += "folder location: ".rjust(col_align) + "{}\n"
        out += "filename format: ".rjust(col_align) + "{}\n"
        out += "mask filename: ".rjust(col_align) + "{}\n"
        out += "Ref. vel. field filename: ".rjust(col_align) + "{}\n"
        out += "image dimensions: ".rjust(col_align) + "{}x{}\n"
        out += "number of images in ensemble: ".rjust(col_align) + "{}\n"
        out += "Is synthetic (y/n): ".rjust(col_align) + "{}\n"
        out += "Is time resolved (y/n): ".rjust(col_align) + "{}\n"
        return out.format(self.flowtype,
                          self.description,
                          self.folder,
                          self.filename,
                          self.mask_fname,
                          self.vel_field_fname,
                          self.n_rows(), self.n_cols(),
                          self.n_images,
                          self.is_synthetic,
                          self.is_time_resolved)

    def print_row_details(self):
        """Displays the details of the current flow type as a single row
        """
        if self.mask_fname == "none":
            mask_yn = 'n'
        else:
            mask_yn = 'y'
        if self.vel_field_fname == "none":
            vel_yn = 'n'
        else:
            vel_yn = 'y'

        print("{:^3}|{:^30}|{:^21}|{:^5}|{:^4}|{:^9}|{:^4}".format(
            self.flowtype,
            self.description,
            self.folder,
            mask_yn,
            vel_yn,
            self.img_dim_text,
            self.n_images))

    def n_cols(self):
        """returns the number of columns in the image

        Returns:
            INT: number of columns
        """
        dims = self.img_dim_text.split('x')
        return int(dims[1])

    def n_rows(self):
        """returns the number of rows in the image

        Returns:
            INT: number of rows
        """
        dims = self.img_dim_text.split('x')
        return int(dims[0])

    def img_dim(self):
        """returns the image dimensions MxN
        where M is the height and N is the width

        Returns:
            2x1 int array: the dimensions of the image
        """
        dims = self.img_dim_text.split('x')
        return dims

    def formatted_filenames(self, im_number):
        """returns two file names corresponding to a and b for the
        requested im_number within the ensemble

        Args:
            im_number (int): image number in the ensemble to obtain
        """
        root = "C:/Users/me12288/local documents/PhD - Local/"
        folder = "images/imageDB/" + self.folder + "/"
        filenames = []
        filenames.append(root + folder + self.filename % (im_number, 'a'))
        filenames.append(root + folder + self.filename % (im_number, 'b'))
        return filenames


def get_image_information(flowtype):
    """ searches in the database for image details


    """
    with open(path_to_file_index) as imageDB:
        all_information = csv.reader(imageDB)
        for row in itertools.islice(all_information, 1, None):
            if int(row[0]) == flowtype:
                return row
    # if here then we have not found the correct row
    raise ValueError("Item not found")


def list_available_flowtypes():
    """Displays the image flow types which are available to be
        loaded along with their calling ID
    """
    with open(path_to_file_index) as imageDB:
        all_information = csv.reader(imageDB)
        for row in all_information:
            print("{} - {}".format(row[0], row[1]))


def print_table_header():
    """prints the table header e.g. id/label/nImages etc
        """
    print("{:^3}|{:^30}|{:^21}|{:^5}|{:^4}|{:^9}|{:^4}"
          .format("ID",
                  "label",
                  "Folder",
                  "mask?",
                  "vel?",
                  "dim.",
                  "nImg"))


def print_all_details():
    """Prints all the details for all the possible flow types
    includes printing of the header row
    """

    with open(path_to_file_index) as imageDB:
        all_information = csv.reader(imageDB)
        # print header
        print_table_header()
        for row in itertools.islice(all_information, 1, None):
            img_obj = ImageInfo(int(row[0]))
            img_obj.print_row_details()


if __name__ == "__main__":
    print_all_details()

    print('---------------')
    print(' ')
    img = []

    print('loading backwards facing step details into workspace')
    img.append(ImageInfo(1))
    print('done')
    print('testing dimensions')
    print('dim text: {}'.format(img[0].img_dim_text))
    dim = img[0].img_dim()
    print('dim: {} x {}'.format(dim[0], dim[1]))
    print('width: {}'.format(img[0].n_cols()))
    print('height: {}'.format(img[0].n_rows()))
    print('testing number of images in ensemble')
    print('nImages: {}'.format(img[0].n_images))

    print('also loading weamFlow details')
    img.append(ImageInfo(19))
    print('done')
    print('testing dimensions')
    print('dim text: {}'.format(img[1].img_dim_text))
    dim1 = img[1].img_dim()
    print('dim: {} x {}'.format(dim1[0], dim1[1]))
    print('width: {}'.format(img[1].n_cols()))
    print('height: {}'.format(img[1].n_rows()))
    print('testing number of images in ensemble')
    print('nImages: {}'.format(img[1].n_images))

    print(img[1])

    list_available_flowtypes()

    for im in img:
        fnames = im.formatted_filenames(2)
        print(fnames)
