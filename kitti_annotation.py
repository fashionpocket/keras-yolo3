import argparse
import os
from collections import defaultdict
from PIL import Image

from utils.io_utils import read_txt_gt4kitti

"""
created by anno on 2020/02/12

Generating KITTI Annotation for training yolov3
from ground truth label or inference results by pre-trained model.

KITTI MOT dataset has the following ground truth labels.
['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare']

In utils.io_utils, 'Misc' or 'DontCare' classes are ignored.
So we will get a number labeling like following.
[0, 1, 2, 3, 4, 5, 6]
"""

label_numbering = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person': 4,
    'Cyclist': 5,
    'Tram': 6
}


def make_gt_annotation(image_dirpath, label_dirpath, out_path):
    """
    This method makes ground truth annotation txt.
    :param image_dirpath: path to image dir (end with /image_dd), string
    :param label_dirpath: path to label_dir (end with /label_dd), string
    """
    name_box_id = defaultdict(list)
    label_files = sorted(os.listdir(label_dirpath))
    for file in label_files:
        label_path = os.path.join(label_dirpath, file)
        gt_kitti = read_txt_gt4kitti(label_path)
        for key, bboxes in gt_kitti.items():
            frame_num = int(key)
            sequence_num = file.rstrip(".txt")
            image_path = os.path.join(image_dirpath, "%s/%6d" % (sequence_num, frame_num))
            for label_bbox in bboxes:
                if label_bbox[0] not in list(label_numbering.keys()):
                    continue
                class_id = label_numbering[label_bbox[0]]
                bbox = label_bbox[1]  # [left, top, height, bottom] in float
                name_box_id[image_path].append([bbox, class_id])

    f = open(out_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = int(info[0][2])
            y_max = int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()


def make_inference_annotation(image_dirpath, out_path):
    """
    This method makes inferenced annotation txt by pre-trained yolov3.
    :param image_dirpath: path to image dir (end with /image_dd), string
    """
    from yolo import YOLO
    name_box_id = defaultdict(list)
    image_seqs = sorted(os.listdir(image_dirpath))
    yolo = YOLO()

    for seq in image_seqs:
        seq_path = os.path.join(image_dirpath, seq)
        images = sorted(os.listdir(seq_path))
        for i in images:
            image_path = os.path.join(seq_path, i)
            img = Image.open(image_path)
            #img = img.resize(image_size)
            infer_results = yolo.infer_bounding_box(img)
            for ano in infer_results:
                class_id = ano["class_id"]
                bbox = [ano["bbox"]["left"], ano["bbox"]["top"], ano["bbox"]["right"], ano["bbox"]["bottom"]]
                name_box_id[image_path].append([bbox, class_id])

    f = open(out_path, 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = int(info[0][2])
            y_max = int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='kitti_annotation.py',
        description='Generate annotation for KITTI dataset',
        add_help=True,
    )
    parser.add_argument('-m', '--mode', type=str, default='inference', choices=['inference', 'gt'])
    parser.add_argument('-i', '--image_dirpath', type=str, required=True)
    parser.add_argument('-l', '--label_dirpath', type=str, default=None, required=False)
    parser.add_argument('-o', '--out_path', type=str, default='annotation/train.txt', required=True)
    parser.add_argument('-s', '--image_size', type=str, default='1584x480', required=False)

    args = parser.parse_args()

    mode = args.mode
    image_dirpath = args.image_dirpath
    label_dirpath = args.label_dirpath
    out_path = args.out_path
    rows, cols = args.image_size.split("x")


    if not os.path.exists(image_dirpath):
        raise Exception(image_dirpath + ' does not exist.')
    if label_dirpath is not None and (not os.path.exists(label_dirpath)):
        raise Exception(label_dirpath + ' does not exist.')

    # print('- KITTI IMAGE DIR : ' + image_dirpath)
    # print('- KITTI LABEL DIR : ' + label_dirpath)
    print('- ANNOTATION OUTPUT FILEPATH : ' + out_path)
    print('- ANNOTATION MODE : ' + mode)

    if mode == 'gt':
        make_gt_annotation(image_dirpath, label_dirpath, out_path)
    else:
        make_inference_annotation(image_dirpath, out_path)

    print("Making annotation file correctly")