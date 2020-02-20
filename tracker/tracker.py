#coding: utf-8
import os
import cv2
import matplotlib.cm as cm
from yolo import YOLO
from utils import correction_bboxes

class BaseTrackingYolo(YOLO):
    """ 
    The following names cannot be used as instance variables
        - class_names
        - num_classes
        - anchors
        - sess
        - boxes
        - scores
        - classes
        - model_path
        - anchors_path
        - classes_path
        - score (default: 0.3)
        - iou (default: 0.45)
        - model_image_size (default: (416, 416))
        - gpu_num (default: 1)
    """
    def __init__(self, reliability=0.7, **kwargs):
        super().__init__(**kwargs)
        self.reliability = reliability
        self.label2color = dict(zip(self.class_names,[cm.jet(n/self.num_classes) for n in range(self.num_classes)]))

    def drawBBoxes(frame, bboxes):
        for (label,conf,t,b,l,r) in prev_bboxes:
            cv2.rectangle(frame, (left, top), (right, bottom), self.label2color[label], 2)
            cv2.putText(frame, f"{label} ({conf*100:.3f}%)", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    def realtime_correction(self, video, to_path="result.png", **kwargs):
        if isinstance(video, str):
            video = cv2.VideoCapture(video_path)

        im_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        im_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps  = video.get(cv2.CAP_PROP_FPS)
        frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
        int_ms = 1000 // int(fps)

        path,ext = to_path.split(".")
        if ext == "png":
            mode = "img"
            fn_format = "%05d.".join([path,ext])
        else:
            mode = "video"
            frame_size = kwargs.get("frame_size", (1920, 1080))
            fourcc = kwargs.get("fourcc", cv2.VideoWriter_fourcc(*"MJPG"))
            video_result = cv2.VideoWriter(to_path, fourcc, fps, frame_size)

        prev_bboxes = []
        prev_frame = None
        i = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break
            # YOLO's inference Results.
            yolo_results = self.infer_bounding_box(Image.fromarray(frame))
            
            # Initialize tracking Results.
            tracking_results = [tuple() for _ in range(len(prev_bboxes))]
            for (p_label,p_conf,pt,pb,pl,pr) in prev_bboxes:
                pw = pr-pl
                ph = pb-pt
                prev_bbox = (pl,pt,pw,ph)
                # === Create Tracker ===
                # TODO: 
                # Since the tracker keeps the past information as an internal state, 
                # recreating it every time is not enough.
                tracker = self.Tracker_create()
                tracker.init(prev_frame, prev_bbox)
                ret_val, (x,y,w,h) = tracker.update(frame)
                tracking_results[i] = (p_label,p_conf,x,y,x+w,y+h)
            current_bboxes = correction_bboxes(yolo_results, tracking_results)
            
            prev_frame = np.copy(frame)
            self.drawBBoxes(frame, current_bboxes)

            if mode == "img":
                cv2.imwrite(fn_format % i, frame)
                i += 1
            else:
                video_result.write(frame)

            prev_frame = frame
            prev_bboxes = current_bboxes
        if mode=="video":
            video_result.release()
