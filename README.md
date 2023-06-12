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

### Get available models
```python
from det_executor import DetExecutor
# print list of supported arches
DetExecutor.list_arch()
```
<details>
  <summary>Output</summary>

```JSON
{
    "yolov7": YoloArch(
        version="7",
        img_size=(640, 640),
        size="75.6MB",
        params="37.6M",
        flops="",
        module="yolov7_package",
        load_link="yolov7.pt",
        trainable=False,
        traced=False,
    ),
    "yolov7x": YoloArch(
        version="7",
        img_size=(640, 640),
        size="75.6MB",
        params="71.3M",
        flops="",
        module="yolov7_package",
        load_link="yolov7x.pt",
        trainable=False,
        traced=False,
    ),
    "yolov7-w6": YoloArch(
        version="7",
        img_size=(1280, 1280),
        size="141.3MB",
        params="70.4M",
        flops="",
        module="yolov7_package",
        load_link="yolov7-w6.pt",
        trainable=False,
        traced=False,
    ),
    "yolov7-e6": YoloArch(
        version="7",
        img_size=(1280, 1280),
        size="195.0MB",
        params="97.2M",
        flops="",
        module="yolov7_package",
        load_link="yolov7-e6.pt",
        trainable=False,
        traced=False,
    ),
    "yolov7-d6": YoloArch(
        version="7",
        img_size=(1280, 1280),
        size="286.3MB",
        params="133.8M",
        flops="",
        module="yolov7_package",
        load_link="yolov7-d6.pt",
        trainable=False,
        traced=False,
    ),
    "yolov7-e6e": YoloArch(
        version="7",
        img_size=(1280, 1280),
        size="304.4MB",
        params="151.8M",
        flops="",
        module="yolov7_package",
        load_link="yolov7-e6e.pt",
        trainable=False,
        traced=False,
    ),
    "yolov7-traced": YoloArch(
        version="7",
        img_size=(640, 640),
        size="74.3MB",
        params="36.9M",
        flops="",
        module="yolov7_package",
        load_link="1L8mPcUvabUscEk6Nr8ck5EFgopgPAMDW",
        trainable=False,
        traced=True,
    ),
    "yolov7-tiny": YoloArch(
        version="7",
        img_size=(640, 640),
        size="12.6MB",
        params="6.2M",
        flops="",
        module="yolov7_package",
        load_link="yolov7-tiny.pt",
        trainable=False,
        traced=False,
    ),
    "yolov7-tiny-traced": YoloArch(
        version="7",
        img_size=(640, 640),
        size="12.7MB",
        params="6.2M",
        flops="",
        module="yolov7_package",
        load_link="18zJyljtolPENDI_kFw3FlRFnQTnaLuDF",
        trainable=False,
        traced=True,
    ),
    "yolov8n": YoloArch(
        version="8",
        img_size=(640, 640),
        size="6.5MB",
        params="3.2M",
        flops="",
        module="yolov8",
        load_link="yolov8n.pt",
        trainable=False,
        traced=False,
    ),
    "yolov8s": YoloArch(
        version="8",
        img_size=(640, 640),
        size="22.6MB",
        params="11.2M",
        flops="",
        module="yolov8",
        load_link="yolov8s.pt",
        trainable=False,
        traced=False,
    ),
    "yolov8m": YoloArch(
        version="8",
        img_size=(640, 640),
        size="52.1MB",
        params="25.9M",
        flops="",
        module="yolov8",
        load_link="yolov8m.pt",
        trainable=False,
        traced=False,
    ),
    "yolov8l": YoloArch(
        version="8",
        img_size=(640, 640),
        size="87.8MB",
        params="43.7M",
        flops="",
        module="yolov8",
        load_link="yolov8l.pt",
        trainable=False,
        traced=False,
    ),
    "yolov8x": YoloArch(
        version="8",
        img_size=(640, 640),
        size="136.9MB",
        params="68.2M",
        flops="",
        module="yolov8",
        load_link="yolov8x.pt",
        trainable=False,
        traced=False,
    ),
    "yolos-tiny": YoloArch(
        version="s",
        img_size=None,
        size="136.9MB",
        params="6.5M",
        flops="512x*>18.8G|256x*>3.4G",
        module="yolos",
        load_link="hustvl/yolos-tiny",
        trainable=False,
        traced=False,
    ),
}
```
</details>
<br/>

### Loading model
```python
from det_executor import DetExecutor

# loading model
name = 'yolov7'
ex = DetExecutor(name)
```
### Predict and draw
```python
from det_executor import DetExecutor, draw_on_image
import cv2

# loading model
name = 'yolov7'
ex = DetExecutor(name)

# loading image
img = ex.load_image('test/img.jpg')
# or img = cv2.imread('test/img.jpg')

# predict
classes, boxes, scores = ex.predict(img)

# draw
img = draw_on_image(img, boxes[0], scores[0], classes[0])
cv2.imshow("image", img)
cv2.waitKey()
```

## Roadmap
 - [ ] Training pipeline for all models
 - [ ] Load from custom weights
 - [ ] More models
## Citation
```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

```
@misc{fang2021look,
      title={You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection}, 
      author={Yuxin Fang and Bencheng Liao and Xinggang Wang and Jiemin Fang and Jiyang Qi and Rui Wu and Jianwei Niu and Wenyu Liu},
      year={2021},
      eprint={2106.00666},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
