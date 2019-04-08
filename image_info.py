import csv
import itertools
path_to_file_index = 'index.csv'


class ImageInfo:
    """Class containing information relating to images of a single flow type.

    USAGE:
        imgObject = ImageInfo(flow_type)

    Attributes:
        flow_type (int): ID corresponding to a particular flow
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

    def __init__(self, flow_type):
        """Initialises an image_info object from the data in the CSV file
        who's path is defined by path_to_file_index at the top of
        "image_info.py".

        Args:
            flow_type (integer): ID of the flow to be read in

        """
        # read data from csv file
        row = get_image_information(flow_type)
        self.flow_type = int(row[0])
        self.description = row[1]
        self.folder = row[2]
        self.filename = row[3]
        self.mask_fname = row[4]
        self.vel_field_fname = row[5]
        self.img_dim = row[6]
        self.max_n_images = int(row[7])
        self.is_synthetic = row[8]
        self.is_time_resolved = row[9]

    def __repr__(self):
        """returns the representation of the object,
        i.e. how it is constructed
        """
        return "ImageInfo({})".format(self.flow_type)

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
        return out.format(self.flow_type,
                          self.description,
                          self.folder,
                          self.filename,
                          self.mask_fname,
                          self.vel_field_fname,
                          self.n_rows(), self.n_cols(),
                          self.max_n_images,
                          self.is_synthetic,
                          self.is_time_resolved)

    def print_table_header(self):
        """prints the table header e.g. id/label/nImages etc
        """
        print("{:^5} | {:^14} | {:^8} | {:^10} | ".format("ID",
                                                          "label",
                                                          "Folder",
                                                          "filename"))

    def print_row_details(self):
        """Displays the details of the current flow type as a single row
        """
        print("{:^5} | {:^14} | {:^8} | {:^10} |".format(
            self.flow_type,
            self.description,
            self.folder,
            self.filename))

    def n_cols(self):
        """returns the number of columns in the image

        Returns:
            INT: number of columns
        """
        dims = self.img_dim.split('x')
        return int(dims[1])

    def n_rows(self):
        """returns the number of rows in the image

        Returns:
            INT: number of rows
        """
        dims = self.img_dim.split('x')
        return int(dims[0])


def get_image_information(flow_type):
    """ searches in the database for image details


    """
    with open(path_to_file_index) as imageDB:
        all_information = csv.reader(imageDB)
        for row in itertools.islice(all_information, 1, None):
            if int(row[0]) == flow_type:
                return row
    # if here then we have not found the correct row
    raise ValueError("Item not found")


def list_options():
    """Displays the image flow types which are available to be
        loaded along with their calling ID
    """
    with open(path_to_file_index) as imageDB:
        all_information = csv.reader(imageDB)
        for row in all_information:
            print("{} - {}".format(row[0], row[1]))


def print_all_details():
    """Prints all the details for all the possible flow types
    """
    with open(path_to_file_index) as imageDB:
        all_information = csv.reader(imageDB)
        # print header
        print("ID Description Folder")
        for row in itertools.islice(all_information, 1, None):
            img_obj = ImageInfo(int(row[0]))
            print(img_obj)


if __name__ == "__main__":
    img = ImageInfo(1)
    img.print_table_header()
    img.print_row_details()
