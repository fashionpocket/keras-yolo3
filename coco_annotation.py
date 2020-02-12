import argparse
import os
import json
from collections import defaultdict


def make_gt_annotation(label_path):
    """
    This method makes annotation txt on MS COCO dataset.
    label_path ... e.g. "mscoco2017/annotations/instances_train2017.json"
    :param label_path: path to label json, string
    """
    name_box_id = defaultdict(list)
    id_name = dict()
    f = open(
        label_path,
        encoding='utf-8')
    data = json.load(f)

    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = 'mscoco2017/train2017/%012d.jpg' % id
        cat = ant['category_id']

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat])

    f = open('train.txt', 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='coco_annotation.py',
        description='Generate annotation for MSCOCO2017 dataset',
        add_help=True,
    )
    parser.add_argument('-l', '--label_path', type=str, required=True)

    args = parser.parse_args()
    label_path = args.label_path

    if not os.path.exists(label_path):
        raise Exception(label_path + ' does not exist.')

    print('- MSCOCO2017 LABEL DIR : ' + label_path)
    make_gt_annotation(label_path)
    print("Making annotation file correctly")
