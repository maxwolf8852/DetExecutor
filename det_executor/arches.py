from collections import namedtuple
from pthflops import count_ops
import torch
from .utils import SCRIPT_16, SCRIPT_16_TINY


ModelArch = namedtuple(
    "YoloArch",
    [
        "version",
        "img_size",
        "size",
        "params",
        "flops",
        "module",
        "load_link",
        "trainable",
        "traced",
    ],
)


arches_list = {
    "yolov7": ModelArch(
        "7",
        (640, 640),
        "75.6MB",
        "37.6M",
        "",
        "yolov7_package",
        "yolov7.pt",
        False,
        False,
    ),
    "yolov7x": ModelArch(
        "7",
        (640, 640),
        "75.6MB",
        "71.3M",
        "",
        "yolov7_package",
        "yolov7x.pt",
        False,
        False,
    ),
    "yolov7-w6": ModelArch(
        "7",
        (1280, 1280),
        "141.3MB",
        "70.4M",
        "",
        "yolov7_package",
        "yolov7-w6.pt",
        False,
        False,
    ),
    "yolov7-e6": ModelArch(
        "7",
        (1280, 1280),
        "195.0MB",
        "97.2M",
        "",
        "yolov7_package",
        "yolov7-e6.pt",
        False,
        False,
    ),
    "yolov7-d6": ModelArch(
        "7",
        (1280, 1280),
        "286.3MB",
        "133.8M",
        "",
        "yolov7_package",
        "yolov7-d6.pt",
        False,
        False,
    ),
    "yolov7-e6e": ModelArch(
        "7",
        (1280, 1280),
        "304.4MB",
        "151.8M",
        "",
        "yolov7_package",
        "yolov7-e6e.pt",
        False,
        False,
    ),
    "yolov7-traced": ModelArch(
        "7", (640, 640), "74.3MB", "36.9M", "", "yolov7_package", SCRIPT_16, False, True
    ),
    "yolov7-tiny": ModelArch(
        "7",
        (640, 640),
        "12.6MB",
        "6.2M",
        "",
        "yolov7_package",
        "yolov7-tiny.pt",
        False,
        False,
    ),
    "yolov7-tiny-traced": ModelArch(
        "7",
        (640, 640),
        "12.7MB",
        "6.2M",
        "",
        "yolov7_package",
        SCRIPT_16_TINY,
        False,
        True,
    ),
    "yolov8n": ModelArch(
        "8", (640, 640), "6.5MB", "3.2M", "", "yolov8", "yolov8n.pt", False, False
    ),
    "yolov8s": ModelArch(
        "8", (640, 640), "22.6MB", "11.2M", "", "yolov8", "yolov8s.pt", False, False
    ),
    "yolov8m": ModelArch(
        "8", (640, 640), "52.1MB", "25.9M", "", "yolov8", "yolov8m.pt", False, False
    ),
    "yolov8l": ModelArch(
        "8", (640, 640), "87.8MB", "43.7M", "", "yolov8", "yolov8l.pt", False, False
    ),
    "yolov8x": ModelArch(
        "8", (640, 640), "136.9MB", "68.2M", "", "yolov8", "yolov8x.pt", False, False
    ),
    "yolos-tiny": ModelArch(
        "s",
        None,
        "136.9MB",
        "6.5M",
        "512x*>18.8G|256x*>3.4G",
        "yolos",
        "hustvl/yolos-tiny",
        False,
        False,
    ),
}


def count_parameters(model):
    return f"{sum(p.numel() for p in model.parameters()) / 1000000:.1f}M"


def count_flops(model, _device, img_size):
    print(count_parameters(model))
    inp = torch.rand(1, 3, img_size[0], img_size[1]).to(_device)
    print(count_ops(model, inp))
