import colorsys
import cv2


def conf2color(conf):
    if (conf <= 0.6):
        color = colorsys.hsv_to_rgb(15 / 360.0, 1, 1)
    else:
        color = colorsys.hsv_to_rgb((25 + 100 * (2 * conf - 1.2)) / 360.0, 1, 1)
    return [k * 255 for k in reversed(color)]


def draw_on_image(img, boxes: list, scores: list, class_ids: list, thickness=1):
    for i, box in enumerate(boxes):
        name = coco_names[class_ids[i]]
        color = conf2color(scores[i])
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        img = cv2.rectangle(img, c1, c2, color, thickness=thickness)
        img = cv2.putText(img, name, (c1[0], c1[1] - 2), 0, 1, color, thickness=thickness)
    return img


coco_names = ["person",
              "bicycle",
              "car",
              "motorbike",
              "aeroplane",
              "bus",
              "train",
              "truck",
              "boat",
              "traffic light",
              "fire hydrant",
              "stop sign",
              "parking meter",
              "bench",
              "bird",
              "cat",
              "dog",
              "horse",
              "sheep",
              "cow",
              "elephant",
              "bear",
              "zebra",
              "giraffe",
              "backpack",
              "umbrella",
              "handbag",
              "tie",
              "suitcase",
              "frisbee",
              "skis",
              "snowboard",
              "sports ball",
              "kite",
              "baseball bat",
              "baseball glove",
              "skateboard",
              "surfboard",
              "tennis racket",
              "bottle",
              "wine glass",
              "cup",
              "fork",
              "knife",
              "spoon",
              "bowl",
              "banana",
              "apple",
              "sandwich",
              "orange",
              "broccoli",
              "carrot",
              "hot dog",
              "pizza",
              "donut",
              "cake",
              "chair",
              "sofa",
              "pottedplant",
              "bed",
              "diningtable",
              "toilet",
              "tvmonitor",
              "laptop",
              "mouse",
              "remote",
              "keyboard",
              "cell phone",
              "microwave",
              "oven",
              "toaster",
              "sink",
              "refrigerator",
              "book",
              "clock",
              "vase",
              "scissors",
              "teddy bear",
              "hair drier",
              "toothbrush"]
