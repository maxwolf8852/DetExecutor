import numpy as np
import cv2
from .base import BaseExecutor, ModelArch
from .arches import count_flops


class YoloSExecutor(BaseExecutor):
    def __init__(
        self, arch: ModelArch, device="cuda:0", half=None, cache_dir=None
    ) -> None:
        super().__init__(arch, device, half, cache_dir)

        from transformers import YolosImageProcessor, YolosForObjectDetection

        self._model = YolosForObjectDetection.from_pretrained(
            self._arch.load_link, cache_dir=self.cache_dir
        ).to(self._device)
        self._image_proc = YolosImageProcessor.from_pretrained(
            self._arch.load_link, cache_dir=self.cache_dir
        )

    def predict(
        self,
        images: np.ndarray | list[np.ndarray],
        conf_thres=0.9,
        iou_thres=0.3,
        multi_label=False,
    ):
        if type(images) != list:
            x = [images]
        else:
            x = images

        x = [cv2.cvtColor(y, cv2.COLOR_RGB2BGR) for y in x]
        classes = []
        boxes = []
        confs = []

        inputs = self._image_proc(images=x, return_tensors="pt")
        outputs = self._model(**inputs)
        # target_sizes = torch.tensor([image.size[::-1]])
        results_list = self._image_proc.post_process_object_detection(
            outputs, threshold=conf_thres, target_sizes=[y.shape for y in x]
        )

        for results in results_list:
            classes.append([_convert_array[int(x)] for x in results["labels"]])
            boxes.append([[round(i, 2) for i in x.tolist()] for x in results["boxes"]])
            confs.append([round(x.item(), 3) for x in results["scores"]])

        return classes, boxes, confs


_convert_array = [
    None,
    0,
    1,
    2,
    None,
    None,
    5,
    6,
    7,
    8,
    9,
    10,
    None,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    None,
    24,
    25,
    None,
    None,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    None,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    None,
    None,
    59,
    None,
    None,
    None,
    None,
    61,
    None,
    None,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    None,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
]
