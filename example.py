import time

from det_executor import DetExecutor, draw_on_image
import cv2
import numpy as np


def test(name):
    ex = DetExecutor(name, half=None, device="cpu", cache_dir="./weights/")
    img = ex.load_image("tests/img.jpg")
    start = time.perf_counter()
    classes, boxes, scores = ex(img)
    print(f"Time: {time.perf_counter() - start}")
    print(classes[0])
    img = draw_on_image(img, boxes[0], scores[0], classes[0])
    cv2.imshow("image", img)
    cv2.waitKey()


if __name__ == "__main__":
    DetExecutor.list_arch()
    # test('yolov8n')
    # test("yolov8s")
    # test("yolov8m")
    # test('yolov8l')
    # test('yolov8x')
    # test('yolov7')
    # test('yolov7-traced')
    # test("yolov7-tiny")
    # test('yolov7x')
    # test('yolov7-d6')
    # test('yolov7-e6e')
    test("yolos-tiny")
