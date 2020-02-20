#coding: utf-8
import cv2
from yolo import YOLO
from tracker import BaseTrackingYolo

OPENCV_TRACKER_CREATER = {
    'boosting': cv2.TrackerBoosting_create,
    'mil': cv2.TrackerMIL_create,
    'kcf': cv2.TrackerKCF_create,
    'tld': cv2.TrackerTLD_create,
    'median_flow': cv2.TrackerMedianFlow_create,
    'goturn': cv2.TrackerGOTURN_create,
    'mosse': cv2.TrackerMOSSE_create,
    'csrt': cv2.TrackerCSRT_create,
}

class OpenCVyolo(BaseTrackingYolo):
    def __init__(self, method='kcf', reliability=0.7, **kwargs):
        super().__init__(reliability=reliability, **kwargs)
        self.Tracker_create = OPENCV_TRACKER_CREATER.get(method.lower())
