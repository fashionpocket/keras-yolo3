#coding: utf-8
import cv2
from yolo import YOLO

OPENCV_TRACKER_HANDLER = {
    "mil": cv2.TrackerMIL_create(),
    'kcf': cv2.TrackerKCF_create(),
    'tld': cv2.TrackerTLD_create(),
    'median_flow': cv2.TrackerMedianFlow_create(),
    'goturn': cv2.TrackerGOTURN_create(),
}

class OpenCVyolo(YOLO):
    def __init__(self, method='kcf', **kwargs):
        self.tracker = OPENCV_TRACKER_HANDLER.get(method.lower())
        super().__init__(**kwargs)

    def infer_video(self, video_path):
        pass

    

    def video_inference(self, video_path):
        # When the `video_path` directory has the sequential images.
        if os.path.isdir(video_path):
            self._seq_image_inference(video_path)        
        # When the `video_path` is the video (.mp4, .mov)
        else:
            self._seq_image_inference(video_path)

    def _video_data_inference(self, video_path):
        video = cv2.VideoCapture(video_path)

        im_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        im_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps  = video.get(cv2.CAP_PROP_FPS)
        frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
        int_ms = 1000 // int(fps)
        


    def _seq_image_inference(self, ):
        pass
        