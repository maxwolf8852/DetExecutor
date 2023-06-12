import os
import sys
import numpy as np
import torch
import cv2

from .utils import load_script_model, non_max_suppression
from .models.experimental import attempt_load
from .base import BaseExecutor, ModelArch


class Yolov7Executor(BaseExecutor):
    def __init__(
        self, arch: ModelArch, device="cuda:0", half=None, cache_dir=None
    ) -> None:
        super().__init__(arch, device, half, cache_dir)
        _weights = f"{self.cache_dir}/{self._arch.load_link}"
        if not self._arch.traced:
            sys.path.append(os.path.join(os.path.dirname(__file__), ""))
            self._model = attempt_load(_weights, map_location=self._device)
        else:  # load traced version
            if not os.path.isfile(_weights):
                load_script_model(self._arch.load_link, _weights)
            self._model = (
                torch.jit.load(_weights, map_location=self._device).float().eval()
            )
        if self._half:
            self._model.half()

    def predict(
        self,
        images: np.ndarray | list[np.ndarray],
        conf_thres=0.25,
        iou_thres=0.3,
        multi_label=False,
    ):
        if type(images) != list:
            x = [images]
        else:
            x = images
        classes = []
        boxes = []
        confs = []
        img_sizes = []
        y = []

        with torch.inference_mode():
            for i, img in enumerate(x):
                if type(img) != np.array:
                    img = np.array(img)
                img_sizes.append(x[i].shape)
                img_inf = cv2.resize(img, self._arch.img_size)
                img_inf = img_inf[:, :, ::-1].transpose(2, 0, 1)
                img_inf = np.ascontiguousarray(img_inf)

                img_inf = torch.from_numpy(img_inf).to(self._device)
                img_inf = (
                    img_inf.half() if self._half else img_inf.float()
                )  # uint8 to fp16/32
                img_inf /= 255.0  # 0 - 255 to 0.0 - 1.0
                y.append(img_inf)

            y = torch.stack(y)
            pred = self._model(
                y,
            )[0]
            pred = non_max_suppression(
                pred,
                conf_thres,
                iou_thres,
                classes=None,
                agnostic=False,
                multi_label=multi_label,
            )

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
                        xyxy_scaled = [
                            coords[0] * dy,
                            coords[1] * dx,
                            coords[2] * dy,
                            coords[3] * dx,
                        ]
                        local_classes.append(int(cls.cpu().item()))
                        local_boxes.append(xyxy_scaled)
                        local_confs.append(float(conf.cpu().item()))

                classes.append(local_classes)
                boxes.append(local_boxes)
                confs.append(local_confs)

        return classes, boxes, confs
