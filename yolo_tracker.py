# -*- coding: utf-8 -*-

import argparse
import cv2
from PIL import Image
from timeit import default_timer as timer

import json
import numpy as np
import os
import time

from yolo import YOLO

class YoloTracker(object):

	def __init__(self, method, yolo=YOLO()):
		self._method = method
		self._target_bbox = None
		self._ret_bboxes = []
		self._proc_mss = []
		self._yolo = yolo
		self._frame_size = (1920, 1120)
		self._fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

		self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
							'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
							'dog', 'horse', 'motorbike', 'person', 'pottedplant',
							'sheep', 'sofa', 'train', 'tvmonitor', 'clock']

		self.num_classes = len(self.class_names)
		self.class_colors = []
		for i in range(0, self.num_classes):
			hue = 255*i/ self.num_classes
			col = np.zeros((1,1,3)).astype("uint8")
			col[0][0][0] = hue
			col[0][0][1] = 128
			col[0][0][2] = 255
			cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
			col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
			self.class_colors.append(col)


		self._create_tracker_instance(self._method)


	def _create_tracker_instance(self, method):
		if method == 'boosting':  # Boosting
			self.tracker = cv2.TrackerBoosting_create()
		elif method == 'mil':     # MIL
			self.tracker = cv2.TrackerMIL_create()
		elif method == 'kcf':     # KCF
			self.tracker = cv2.TrackerKCF_create()
		elif method == 'tld':     # TLD
			self.tracker = cv2.TrackerTLD_create()
		elif method == 'median_flow':  # MedianFlow
			self.tracker = cv2.TrackerMedianFlow_create()
		elif method == 'goturn':  # GOTURN
			# Can't open "goturn.prototxt" in function 'ReadProtoFromTextFile'
			# - https://github.com/opencv/opencv_contrib/issues/941#issuecomment-343384500
			# - https://github.com/Auron-X/GOTURN-Example
			# - http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
			self.tracker = cv2.TrackerGOTURN_create()

	def _get_tracker_instance(self, method):
		if method == 'boosting':  # Boosting
			tracker = cv2.TrackerBoosting_create()
		elif method == 'mil':     # MIL
			tracker = cv2.TrackerMIL_create()
		elif method == 'kcf':     # KCF
			tracker = cv2.TrackerKCF_create()
		elif method == 'tld':     # TLD
			tracker = cv2.TrackerTLD_create()
		elif method == 'median_flow':  # MedianFlow
			tracker = cv2.TrackerMedianFlow_create()
		elif method == 'goturn':  # GOTURN
			# Can't open "goturn.prototxt" in function 'ReadProtoFromTextFile'
			# - https://github.com/opencv/opencv_contrib/issues/941#issuecomment-343384500
			# - https://github.com/Auron-X/GOTURN-Example
			# - http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
			tracker = cv2.TrackerGOTURN_create()

		return tracker

	def iou(self, a, b):
		"""
		a, b have to have [left, top, right, bottom]
		"""
		# copied from https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_91_100/answers/answer_93.py
		# get area of a
		area_a = (a[2] - a[0]) * (a[3] - a[1])
		# get area of b
		area_b = (b[2] - b[0]) * (b[3] - b[1])

		# get left top x of IoU
		iou_x1 = np.maximum(a[0], b[0])
		# get left top y of IoU
		iou_y1 = np.maximum(a[1], b[1])
		# get right bottom of IoU
		iou_x2 = np.minimum(a[2], b[2])
		# get right bottom of IoU
		iou_y2 = np.minimum(a[3], b[3])

		# get width of IoU
		iou_w = iou_x2 - iou_x1
		# get height of IoU
		iou_h = iou_y2 - iou_y1

		# no overlap
		if iou_w < 0 or iou_h < 0:
			return 0.0

		# get area of IoU
		area_iou = iou_w * iou_h
		# get overlap ratio between IoU and all area
		iou = area_iou / (area_a + area_b - area_iou)

		return iou

	def assignment_bbox(self, tracked_result, detect_results):
		"""
		This method assigns tracked result to detected results
		by using label and bbox information.
		tracked_result and each element ofdetect_results are expected to have
		following format.

		tracked_result = {
			"label": hoge,
			"bbox": {
				"left": x,
				"top": y,
				"right": x+w,
				"bottom": y+h
			}
		}
		"""

		matched = []
		for dresult in detect_results:
			#print(tracked_result["label"], dresult["label"])
			if tracked_result["label"] == dresult["label"]:
				tbbox = tracked_result["bbox"]
				dbbox = dresult["bbox"]
				a = np.array((tbbox["left"], tbbox["top"], tbbox["right"], tbbox["bottom"]), dtype=np.float32)
				b = np.array((dbbox["left"], dbbox["top"], dbbox["right"], dbbox["bottom"]), dtype=np.float32)
				iou = self.iou(a, b)
				print(a, b, "  iou = ", iou)
				matched.append(
					{"iou": iou, "dresult": dresult}
				)

		matched.sort(key=lambda x: x["iou"], reverse=True)
		if matched == []:
			return None
		else:
			return matched[0]["dresult"]

	def refine_bbox(self, tracked_result, assigned_result):
		# 一度シンプルに平均をとる．
		tbbox = tracked_result["bbox"]
		abbox = assigned_result["bbox"]

		new_bbox = {
			"left": int((tbbox["left"] + abbox["left"]) / 2),
			"top": int((tbbox["top"] + abbox["top"]) / 2),
			"right": int((tbbox["right"] + abbox["right"]) / 2),
			"bottom": int((tbbox["bottom"] + abbox["bottom"]) / 2)
		}

		refined_result = {
			"label": tracked_result["label"],
			"bbox": new_bbox
		}

		return refined_result


	def detect_video(self, video):

		im_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
		im_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
		fps = video.get(cv2.CAP_PROP_FPS)
		frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)

		video_result = cv2.VideoWriter("sample_data/yolo_tracker_result.mp4", self._fourcc, fps, self._frame_size)

		int_ms = 1000 // int(fps)

		bbox_is_ready = False
		initial_case = True

		old_frame = None
		detector_bboxes_old = []

		while video.isOpened():

			final_bboxes = []

			ret, frame = video.read()

			if not ret: # end of the frame
				break

			fimage = Image.fromarray(frame)
			fimage_resized = fimage.resize(self._frame_size)
			detect_results = self._yolo.infer_bounding_box(fimage_resized)

			if initial_case: # initialize tracking objective
				old_frame = frame
				detector_bboxes_old = detect_results
				initial_case = False

			else:
				# foreach detected instance
				for old_object in detector_bboxes_old:
					old_label = old_object["label"]
					old_bbox_dict = old_object["bbox"]

					old_top = old_bbox_dict["top"]
					old_bottom = old_bbox_dict["bottom"]
					old_left = old_bbox_dict["left"]
					old_right = old_bbox_dict["right"]

					old_width = old_right - old_left
					old_height = old_bottom - old_top

					# to (x,y,width,height)
					old_bbox = (old_left, old_top, old_width, old_height)
					#print("initialize grid...", old_bbox)
					tracker = self._get_tracker_instance(self._method)
					tracker.init(old_frame, old_bbox)

					# conduct tracking
					track, raw_result = tracker.update(frame)
					#print("update result... ", raw_result)
					tracked_result = {
						"label": old_label,
						"bbox": {
							"left": raw_result[0],
							"top": raw_result[1],
							"right": raw_result[0] + raw_result[2],
							"bottom": raw_result[1] + raw_result[3]
						}
					}

					if track: # tracking successfully
						assigned_result = self.assignment_bbox(tracked_result, detect_results)
						if assigned_result in detect_results:
							detect_results.remove(assigned_result)
							refined_result = self.refine_bbox(tracked_result, assigned_result)
							detect_results.append(refined_result)

					del tracker

				detector_bboxes_old = detect_results
				old_frame = frame

			for i, result in enumerate(detect_results):
				label = result["label"]
				bbox = result["bbox"]
				top, left, bottom, right = bbox["top"], bbox["left"], bbox["bottom"], bbox["right"]
				#print("refined result:  ", label, (left, top), (right, bottom))

				class_num = 0
				for i in self.class_names:
					if label == i:
						class_num = self.class_names.index(i)
						break

				# make bbox
				#print(self.class_colors[class_num])
				cv2.rectangle(frame, (left, top), (right, bottom), self.class_colors[class_num], 2)

				# ラベルの作成
				text = label
				cv2.rectangle(frame, (left, top - 15), (left + 100, top + 5), self.class_colors[class_num], -1)
				cv2.putText(frame, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

			# cv2.imshow("Show FLAME Image", frame)
			video_result.write(frame)

			# # # escを押したら終了。
			# k = cv2.waitKey(10)
			# if k == ord('q'):  break

		# finally video has to be released
		video_result.release()


if  __name__ == "__main__":

	parser = argparse.ArgumentParser(
		prog='yolo_tracker.py',
		description='Object Detector YOLO + Tracking',
		add_help=True,
	)
	parser.add_argument('-v', '--video', type=str, required=True)
	parser.add_argument('-m', '--method', type=str, default='kcf',
						choices=['boosting', 'mil', 'kcf', 'tld', 'median_flow', 'goturn'],)

	# 1. parse args
	args = parser.parse_args()

	method = args.method
	video_path = args.video
	if not os.path.exists(video_path):
		raise Exception(video_path + ' does not exist.')
	print('- INPUT VIDEO : ' + video_path)
	print('- TRACKING METHOD : ' + method)

	# load video & track a target
	video = cv2.VideoCapture(video_path)

	# create YOLO instance
	yolo = YOLO()

	# create yolo-tracker instance
	yoloTracker = YoloTracker(method, yolo)

	print("start to detect video")

	# conduct object detection
	yoloTracker.detect_video(video)

	video.release()
	cv2.destroyAllWindows()
