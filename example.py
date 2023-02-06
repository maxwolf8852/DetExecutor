import time

from det_executor import DetExecutor, draw_on_image
import cv2


def test(name):
    img = cv2.imread('test/img.jpg')
    ex = DetExecutor(name, half=None)
    start = time.perf_counter()
    classes, boxes, scores = ex.predict(img)
    print(f'Time: {time.perf_counter() - start}')
    img = draw_on_image(img, boxes[0], scores[0], classes[0])
    #cv2.imshow("image", img)
    #cv2.waitKey()


if __name__ == '__main__':
    DetExecutor.list_arch()
    #test('yolov8n')
    #test('yolov8s')
    #test('yolov8m')
    #test('yolov8l')
    #test('yolov8x')
    #test('yolov7')
    #test('yolov7-traced')
    #test('yolov7-tiny')
    #test('yolov7x')
    #test('yolov7-d6')
    #test('yolov7-e6e')
