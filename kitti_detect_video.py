from yolo import YOLO, detect_video

import os
import numpy as np
import cv2
from PIL import Image

image_folder = '../kitti/data_tracking_image_2/training/image_02/0000'
result_path = "result.mp4"

yolo = YOLO()
img_rows = 812
img_cols = 416
img_size = (832, 416)

images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img[0] != "."]
images.sort()
print(os.path.join(image_folder, images[0]))
# frame = Image.open(os.path.join(image_folder, images[0]))
# frame = frame.resize(img_size, Image.BICUBIC)

#video = cv2.VideoWriter(result_path, 0, 1, (img_rows, img_cols))


for i, image in enumerate(images):
	frame = Image.open(os.path.join(image_folder, image))
	frame = frame.resize(img_size)
	detected = yolo.detect_image(frame)
	detected.save('output/0000/%05d.png' % i)
	#video.write(np.asarray(detected))

# cv2.destroyAllWindows()
# video.release()
