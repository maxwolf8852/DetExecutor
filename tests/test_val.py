import unittest
from det_executor import DetExecutor

import cv2


class Test_ValModel(unittest.TestCase):
    def process(self, arch, image):
        det = DetExecutor(arch=arch, device="cpu", cache_dir="./weights/")
        return det(image)

    def test_all_models(self):
        img = cv2.imread("tests/img.jpg")

        self.assertFalse(img is None)

        archs = DetExecutor.get_available_models().keys()
        self.assertFalse(len(archs) == 0)

        for arch in archs:
            classes, boxes, scores = self.process(arch, img)
            self.assertFalse(len(classes[0]) == 0)


if __name__ == "__main__":
    unittest.main()
