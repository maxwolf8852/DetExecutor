![GitHub](https://img.shields.io/github/license/maxwolf8852/DetExecutor?style=plastic)
[![PyPI version](https://badge.fury.io/py/det_executor.svg)](https://badge.fury.io/py/det_executor)
![PyPI - Downloads](https://img.shields.io/pypi/dm/det_executor?style=plastic)

# DetExecutor
Python package with latest versions of YOLO architecture for training and inference
## Install
Installing is quite simple, just use pip:
```shell
pip3 install det_executor
```
## Train
Training support is still in progress!
## Inference
### Loading model
```python3
from det_executor import DetExecutor
# print list of supported arches
DetExecutor.list_arch()

# loading model
name = 'yolov7'
ex = DetExecutor(name)
```
### Predict and draw
```python3
from det_executor import DetExecutor, draw_on_image
import cv2

# loading model
name = 'yolov7'
ex = DetExecutor(name)

# loading image
img = cv2.imread('test/img.jpg')

# predict
classes, boxes, scores = ex.predict(img)

# draw
img = draw_on_image(img, boxes[0], scores[0], classes[0])
cv2.imshow("image", img)
cv2.waitKey()
```
