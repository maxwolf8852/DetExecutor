import numpy as np
from .base import BaseExecutor, ModelArch


class Yolov8Executor(BaseExecutor):
    def __init__(
        self, arch: ModelArch, device="cuda:0", half=None, cache_dir=None
    ) -> None:
        super().__init__(arch, device, half, cache_dir)
        from ultralytics import YOLO

        load_link = f"{self.cache_dir}/{self._arch.load_link}"
        self._model = YOLO(model=load_link)

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
        results = self._model.predict(
            source=x,
            stream=True,
            conf=conf_thres,
            iou=1 - iou_thres,
            half=self._half,
            device=self._device,
        )
        for r in results:
            outs = r.boxes.cpu().numpy()
            classes.append([int(x) for x in outs.cls])
            boxes.append(outs.boxes)
            confs.append(outs.conf)

        return classes, boxes, confs
