import numpy as np
from PIL import Image
import cv2
import imageio.v3 as iio
import pathlib
from .arches import ModelArch


class BaseExecutor:
    def __init__(
        self, arch: ModelArch, device="cuda:0", half=None, cache_dir=None
    ) -> None:
        self._arch = arch
        self._device = device
        self._half = half
        self.cache_dir = cache_dir
        # self._weights = f"{cache_dir}/{arch.}.pt"
        self._model = None

    @staticmethod
    def load_image(
        path: str | pathlib.Path | list[str] | list[pathlib.Path],
    ) -> np.ndarray | list[np.ndarray]:
        if type(path) == str:
            return cv2.cvtColor(iio.imread(path), cv2.COLOR_RGB2BGR)
        elif isinstance(path, list):
            return [cv2.cvtColor(iio.imread(x), cv2.COLOR_RGB2BGR) for x in path]
        else:
            raise ValueError('value "path" must be string path or pathlib.Path object')

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(
        self,
        images: np.ndarray | list[np.ndarray],
        conf_thres=0.25,
        iou_thres=0.3,
        multi_label=False,
    ):
        raise NotImplementedError()
