from .utils import load_script_model, non_max_suppression, SCRIPT_16, SCRIPT_16_TINY
from .models.experimental import attempt_load
from pthflops import count_ops
import torch
import time
from collections import namedtuple
import os
import sys
import pathlib
import black

import cv2
import numpy as np
from colorama import Fore, init

init(autoreset=True)


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


ModelArch = namedtuple('YoloArch',
                       ['version', 'img_size', 'size', 'params', 'module', 'load_link', 'trainable', 'traced'])

arches = {
    'yolov7': ModelArch('7', (640, 640), '75.6MB', '37.6M', 'yolov7_package', 'yolov7.pt', False, False),
    'yolov7x': ModelArch('7', (640, 640), '75.6MB', '71.3M', 'yolov7_package', 'yolov7x.pt', False, False),
    'yolov7-w6': ModelArch('7', (1280, 1280), '141.3MB', '70.4M', 'yolov7_package', 'yolov7-w6.pt', False, False),
    'yolov7-e6': ModelArch('7', (1280, 1280), '195.0MB', '97.2M', 'yolov7_package', 'yolov7-e6.pt', False, False),
    'yolov7-d6': ModelArch('7', (1280, 1280), '286.3MB', '133.8M', 'yolov7_package', 'yolov7-d6.pt', False, False),
    'yolov7-e6e': ModelArch('7', (1280, 1280), '304.4MB', '151.8M', 'yolov7_package', 'yolov7-e6e.pt', False, False),
    'yolov7-traced': ModelArch('7', (640, 640), '74.3MB', '36.9M', 'yolov7_package', SCRIPT_16, False, True),
    'yolov7-tiny': ModelArch('7', (640, 640), '12.6MB', '6.2M', 'yolov7_package', 'yolov7-tiny.pt', False, False),
    'yolov7-tiny-traced': ModelArch('7', (640, 640), '12.7MB', '6.2M', 'yolov7_package', SCRIPT_16_TINY, False, True),
    'yolov8n': ModelArch('8', (640, 640), '6.5MB', '3.2M', 'yolov8', 'yolov8n.pt', False, False),
    'yolov8s': ModelArch('8', (640, 640), '22.6MB', '11.2M', 'yolov8', 'yolov8s.pt', False, False),
    'yolov8m': ModelArch('8', (640, 640), '52.1MB', '25.9M', 'yolov8', 'yolov8m.pt', False, False),
    'yolov8l': ModelArch('8', (640, 640), '87.8MB', '43.7M', 'yolov8', 'yolov8l.pt', False, False),
    'yolov8x': ModelArch('8', (640, 640), '136.9MB', '68.2M', 'yolov8', 'yolov8x.pt', False, False),
}


def count_parameters(model):
    return f'{sum(p.numel() for p in model.parameters()) / 1000000:.1f}M'


class DetExecutor:
    def __init__(self, arch: str, device='cuda:0', half=None, cache_dir=None):
        self._arch: ModelArch = arches[arch]
        if cache_dir is None:
            cache_dir = f'{pathlib.Path(__file__).parent.resolve()}'
        self._weights = f'{cache_dir}/{arch}.pt'
        self._model = None
        cpu = device.lower() == 'cpu'
        cuda = not cpu and torch.cuda.is_available()
        self._device = torch.device(device if cuda else 'cpu')
        if half is None:
            self._half = self._device.type != 'cpu'
        else:
            self._half = half

        print(f'Loading \'{arch}\' model... ', end='')
        start = time.perf_counter()
        blockPrint()
        try:
            if self._arch.module == 'yolov7_package':
                if not self._arch.traced:
                    sys.path.append(os.path.join(
                        os.path.dirname(__file__), ""))
                    self._model = attempt_load(
                        self._weights, map_location=self._device)
                else:  # load traced version
                    if not os.path.isfile(self._weights):
                        load_script_model(self._arch.load_link, self._weights)
                    self._model = torch.jit.load(
                        self._weights, map_location=self._device).float().eval()

                # print()

                if self._half:
                    self._model.half()

            elif self._arch.module == 'yolov8':
                from ultralytics import YOLO
                self._model = YOLO(model=self._arch.load_link)
        except Exception as e:
            enablePrint()
            print(
                f'{Fore.RED}ERROR {{{e}}} {Fore.RESET}[{time.perf_counter() - start}]')
        else:
            enablePrint()
            print(
                f'{Fore.GREEN}SUCCESS {Fore.RESET}[{time.perf_counter() - start}]')

        """try:
            print(count_parameters(self._model.model))
            inp = torch.rand(1, 3, self._arch.img_size[0], self._arch.img_size[1]).to(self._device)
            print(count_ops(self._model.model, inp))
        except:
            print(count_parameters(self._model))
            inp = torch.rand(1, 3, self._arch.img_size[0], self._arch.img_size[1]).to(self._device)
            print(count_ops(self._model, inp))"""

    def predict(self,
                images: np.ndarray | list[np.ndarray],
                conf_thres=0.25,
                iou_thres=0.3,
                multi_label=False):
        if type(images) != list:
            x = [images]
        else:
            x = images
        classes = []
        boxes = []
        confs = []
        img_sizes = []
        y = []
        if self._arch.module == 'yolov7_package':
            with torch.inference_mode():
                for i, img in enumerate(x):
                    if type(img) != np.array:
                        img = np.array(img)
                    img_sizes.append(x[i].shape)
                    img_inf = cv2.resize(img, self._arch.img_size)
                    img_inf = img_inf[:, :, ::-1].transpose(2, 0, 1)
                    img_inf = np.ascontiguousarray(img_inf)

                    img_inf = torch.from_numpy(img_inf).to(self._device)
                    img_inf = img_inf.half() if self._half else img_inf.float()  # uint8 to fp16/32
                    img_inf /= 255.0  # 0 - 255 to 0.0 - 1.0
                    y.append(img_inf)

                y = torch.stack(y)
                pred = self._model(y, )[0]
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                           agnostic=False, multi_label=multi_label)

                for i, det in enumerate(pred):  # detections per image
                    local_classes = []
                    local_boxes = []
                    local_confs = []
                    old_shape = img_sizes[i]
                    if len(det):
                        dx = old_shape[0] / self._arch.img_size[0]
                        dy = old_shape[1] / self._arch.img_size[1]

                        for *xyxy, conf, cls in reversed(det):
                            coords = torch.tensor(xyxy).tolist()
                            xyxy_scaled = [coords[0] * dy, coords[1]
                                           * dx, coords[2] * dy, coords[3] * dx]
                            local_classes.append(int(cls.cpu().item()))
                            local_boxes.append(xyxy_scaled)
                            local_confs.append(float(conf.cpu().item()))

                    classes.append(local_classes)
                    boxes.append(local_boxes)
                    confs.append(local_confs)
        elif self._arch.module == 'yolov8':
            results = self._model.predict(source=x, stream=True)
            for r in results:
                outs = r.boxes.cpu().numpy()
                # print(outs, type(outs))
                # local_boxes = [x.boxes for x in outs]
                # local_classes = [int(x.cls) for x in outs]
                # local_confs = [x.conf for x in outs]

                classes.append([int(x) for x in outs.cls])
                boxes.append(outs.boxes)
                confs.append(outs.conf)

        return classes, boxes, confs

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f'Invalid video path: [{video_path}]')

        number = -1
        while cap.isOpened():
            ret, frame = cap.read()
            number += 1

            if ret:
                yield number, frame,
            else:
                break
        cap.release()

    @staticmethod
    def list_arch():
        print(black.format_str(repr(arches), mode=black.Mode()))
