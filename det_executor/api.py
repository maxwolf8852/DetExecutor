from .base import BaseExecutor
from .arches import ModelArch, arches_list
from .v7 import Yolov7Executor
from .v8 import Yolov8Executor
from .transformer import YoloSExecutor
import torch
import time
import os
import sys
import pathlib
import black

from colorama import Fore, init

init(autoreset=True)


def blockPrint():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class DetExecutor:
    def __new__(
        cls, arch: str, device="cuda:0", half=None, cache_dir=None
    ) -> BaseExecutor:
        print(f"Loading '{arch}' model... ", end="")
        start = time.perf_counter()
        blockPrint()
        try:
            _arch: ModelArch = arches_list[arch]
            cuda = not (device.lower() == "cpu") and torch.cuda.is_available()
            _device = torch.device(device if cuda else "cpu")
            if cache_dir is None:
                cache_dir = f"{pathlib.Path(__file__).parent.resolve()}"
            print(_arch)
            if half is None:
                _half = _device.type != "cpu"
            else:
                _half = half

            if _arch.module == "yolov7_package":
                instance = super().__new__(Yolov7Executor)
            elif _arch.module == "yolov8":
                instance = super().__new__(Yolov8Executor)
            elif _arch.module == "yolos":
                instance = super().__new__(YoloSExecutor)
            else:
                raise NotImplementedError(_arch.module)

            instance.__init__(_arch, _device, _half, cache_dir)

        except Exception as e:
            enablePrint()
            print(
                f"{Fore.RED}ERROR {{{e}}} {Fore.RESET}[{time.perf_counter() - start}]"
            )
        else:
            enablePrint()
            print(f"{Fore.GREEN}SUCCESS {Fore.RESET}[{time.perf_counter() - start}]")
        return instance

    @staticmethod
    def list_arch():
        print(black.format_str(repr(arches_list), mode=black.Mode()))

    @staticmethod
    def get_available_models(trainable=None):
        if trainable is None:
            return arches_list.copy()
        else:
            return {
                key: value
                for key, value in arches_list.items()
                if value.trainable == trainable
            }

    # def process_video(self, video_path: str):
    #     cap = cv2.VideoCapture(video_path)
    #     if not cap.isOpened():
    #         raise ValueError(f"Invalid video path: [{video_path}]")

    #     number = -1
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         number += 1

    #         if ret:
    #             yield number, frame,
    #         else:
    #             break
    #     cap.release()
