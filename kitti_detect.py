from yolo import YOLO, detect_video

import numpy as np
import cv2
from PIL import Image

image_path = "../kitti/training/image_02/0000/000000.png"
img = Image.open(image_path).convert("RGB")
img = img.resize((832, 416), Image.BICUBIC)
print(type(img))
print(img.size)
# img = np.asarray(img)
# print(img.size)

yolo = YOLO()
detected = yolo.detect_image(img)
detected.save("detected.png")

