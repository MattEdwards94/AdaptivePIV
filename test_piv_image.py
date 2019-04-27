import unittest
import piv_image
import image_info


class TestPIVImage(unittest.TestCase):
    def test_initialisation(self):
        img = piv_image.piv_image(image_info.ImageInfo(1), 1)
        self.assertEqual(img.img_number, 1)


if __name__ == "__main__":
    unittest.main()
