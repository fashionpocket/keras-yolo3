import os
import csv
import copy
persons_class = ["1"]

kitti_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']

"""
Created by kyomin on 
Modified by anno on 2020/02/12
"""

def xywh2xyxy(bbox):
    """
    convert bbox from [x,y,w,h] to [x1, y1, x2, y2]
    :param bbox: bbox in string [x, y, w, h], list
    :return: bbox in float [x1, y1, x2, y2], list
    """
    copy.deepcopy(bbox)
    bbox[0] = float(bbox[0])
    bbox[1] = float(bbox[1])
    bbox[2] = float(bbox[2]) + bbox[0]
    bbox[3] = float(bbox[3]) + bbox[1]

    return bbox
def reorder_frameID(frame_dict):
    """
    reorder the frames dictionary in a ascending manner
    :param frame_dict: a dict with key = frameid and value is a list of lists [object id, x, y, w, h] in the frame, dict
    :return: ordered dict by frameid
    """
    keys_int = sorted([int(i) for i in frame_dict.keys()])

    new_dict = {}
    for key in keys_int:
        new_dict[str(key)] = frame_dict[str(key)]
    return new_dict

def read_txt_gt4V(textpath):
    # TODO: this method will be no longer used
    """
    read gt.txt in MOT17Det dataset to a dict
    :param textpath: text path, string
    :return: a dict with key = frameid and value is a list of lists [object id, x1, y1, x2, y2] in the frame, dict
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            # we only consider "pedestrian" class #
            if len(line) < 7 or (line[7] not in persons_class and "MOT2015" not in textpath) or int(float(line[6]))==0:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([line[1]]+bbox)
    ordered = reorder_frameID(frames)
    return ordered

def read_txt_gt4mot17det(textpath):
    """
    read gt.txt in MOT17Det dataset to a dict
    :param textpath: text path, string
    :return: a dict with key = frameid and value is a tuple (class name str, [x1, y1, x2, y2]) in the frame, dict
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            if len(line) < 7 or ("MOT2015" not in textpath) or int(float(line[6]))==0:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([line[1]]+bbox)
    ordered = reorder_frameID(frames)
    return ordered

def read_txt_gt4kitti(textpath):
    """
    read dddd.txt in KITTI multi object tracking to a dict
    This label data has the following schema. The columns is 17.

    1   frame        Frame within the sequence where the object appearers
    1   track id     Unique tracking id of this object within this sequence
    1   type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
    1   truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
    1   occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
    1   alpha        Observation angle of object, ranging [-pi..pi]
    4   bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
    3   dimensions   3D object dimensions: height, width, length (in meters)
    3   location     3D object location x,y,z in camera coordinates (in meters)
    1   rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1   score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

    This dataset contains the following classes.
    ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare']
    These classes will be converted into the following number labeling.
    [0, 1, 2, 3, 4, 5, 6 (, 7)]
    The class "'Misc' or 'DontCare'" will be ignored.

    :param textpath: text path, string
    :return: a dict with key = frameid and value is a tuple (class name str, [x1, y1, x2, y2]) in the frame, dict
    """

    with open(textpath) as f:
        frames = {}
        for line in f.readlines():
            line = line.rstrip()
            line = line.split(" ")
            # print(line)
            if len(line) < 17 or line[2] in ['Misc', 'DontCare']:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            # frame[6:10] -> [left, top, height, bottom] in String
            bbox = [float(line[6]), float(line[7]), float(line[8]), float(line[9])]
            # object_id = kitti_classes.index(line[2])
            object_class = line[2]
            frames[line[0]].append((object_class, bbox))
    ordered = reorder_frameID(frames)
    return ordered




