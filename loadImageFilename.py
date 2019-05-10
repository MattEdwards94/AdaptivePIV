import csv


def loadImageFilename(flowType):
    """ Switches based upon the input flowType

        Options:
        Experimental
            1   - Backwards Facing Step
            2   - Experimental Vortex
            3   - Corner flow
            4   - Jet Flow from PIV Challenge
            5   - Buildings flow
            6   - High Pressure Jet
            7   - Wall Jet
            8   - Lungs Bifurcation
            9   - PIV Challenge Jet
            10  - PIV Challenge 3 - A1
            11  - PIV Challenge 3 - A2
            12  - PIV Challenge 3 - A3
            13  - PIV Challenge 3 - A4
            14  - festip
            15  - Renault plain with mask
            16  - Renault plain without mask
            17  - Renault filtered with mask
            18  - Renault filtered without mask
            19  - Weam wake flow
            20  - Weam wake flow background subtracted
            21  - PIV challenge 4 - Micro channel

        Synthetic
            101 - Vortex array 2x2
            102 - Lamb Oseen
            103 - Gaussian smoothed with freestream
            104 - Gaussian smoothed
            105 - Gaussian smoothed with variable kernel

    """


with open('C:/Users/me12288/Local Documents/PhD - Local/images/imageDB/index.csv') as csvfile:
    file_info = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file_info:
        print(row)


if __name__ == "__main__":
    loadImageFilename(1)
