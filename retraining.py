"""
This class is responsible for the whole experimentals
"""

# TODO: implement unfinished
import argparse
import colorsys
import datetime
import os
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model

from utils.io_utils import write_csv
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss, yolo_eval
from yolo3.utils import get_random_data, letterbox_image

from utils.coco_tools import ExportGroundtruthToCOCO, ExportDetectionsToCOCO, COCOWrapper, COCOEvalWrapper


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

class Trainer(object):
    _defaults = {
        "score": 0.3,
        "iou": 0.45,
        "gpu_num": 1,
    }

    def __init__(self,
                 train_annotation_path='train.txt',
                 valid_annotation_path='valid.txt',
                 log_dir='logs/000/',
                 output_dir='outputs/000',
                 classes_path='model_data/coco_classes.txt',
                 anchors_path='model_data/yolo_anchors.txt',
                 input_shape=(416, 416),
                 gpu_num=1
                 ):
        self.train_annotation_path = train_annotation_path
        self.valid_annotation_path = valid_annotation_path
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.class_names = self.get_classes()
        self.num_classes = len(self.class_names)
        self.anchors = self.get_anchors()

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.training_log_file = open(os.path.join(self.log_dir, "progress.csv"), "w")

        self.categories_dict = [{'id': i, 'name': name} for i, name in enumerate(self.class_names)] # for COCO API

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        self.sess = K.get_session()

        self.input_shape = input_shape
        self.gpu_num = gpu_num
        self.score = 0.3
        self.iou = 0.45
        self.ready_for_model = False
        self.ready_for_data = False
        self.is_trained = False

    def print_categories_dict(self): # for debugging
        print(self.categories_dict)

    def get_classes(self):
        '''loads the classes'''
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        '''loads the anchors from a file'''
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def create_model(self, load_pretrained=True, freeze_body=2, weights_path='model_data/yolo_weights.h5'):
        '''create the training model'''
        # K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape
        num_anchors = len(self.anchors)

        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                               num_anchors // 3, self.num_classes + 5)) for l in range(3)]

        self.model_body = yolo_body(image_input, num_anchors // 3, self.num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, self.num_classes))

        if self.gpu_num>=2:
            self.model_body = multi_gpu_model(self.model_body, gpus=self.gpu_num)

        # Generate YOLOv3 core model for learning
        if load_pretrained:
            self.model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))

            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(self.model_body.layers) - 3)[freeze_body - 1]
                for i in range(num): self.model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(self.model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5})(
            [*self.model_body.output, *y_true])
        model = Model([self.model_body.input, *y_true], model_loss)

        # Generate output tensor targets for filtered bounding boxes
        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.model_body.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)

        self.ready_for_model = True
        return model, boxes, scores, classes

    def make_detection_list(self, annotation_lines):
        assert self.ready_for_data, "Ready for validation data first"
        image_ids = []
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        for line in annotation_lines:
            line = line.split()
            image_id = line[0]
            image = Image.open(line[0])
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')
            # preprocessing image
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
            # inference
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model_body.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            image_ids.append(image_id)
            detection_scores.append(np.array(out_scores))
            detection_classes.append(np.array(out_classes))
            if len(out_boxes) == 0:
                detection_boxes.append(np.zeros((0, 4)))
            else:
                detection_boxes.append(
                    np.array([np.array([left, top, right, bottom]) for top, left, bottom, right in out_boxes])
                )
            del image

        return image_ids, detection_boxes, detection_scores, detection_classes

    def make_groundtruth_list(self, annotation_lines):
        assert self.ready_for_data, "Ready for validation data first"
        image_ids = []
        detection_boxes = []
        detection_classes = []
        for line in annotation_lines:
            line = line.split()
            image_id = line[0]
            line_boxes_ids = [l.split(",") for l in line[1:]]
            # print(line[1:])
            if len(line_boxes_ids) == 0:
                boxes = np.zeros((0, 4))
            else:
                boxes = np.array([np.array([float(xmin), float(ymin), float(xmax), float(ymax)])
                                  for xmin, ymin, xmax, ymax, _ in line_boxes_ids])
            classes = np.array([int(class_id) for _, _, _, _, class_id in line_boxes_ids])
            image_ids.append(image_id)
            detection_boxes.append(boxes)
            detection_classes.append(classes)
        return image_ids, detection_boxes, detection_classes

    def train(self, freeze_body=2, weights_path='model_data/yolo_weights.h5', batch_size=32, epochs=1000):

        self.model, self.boxes, self.scores, self.classes = self.create_model(
            load_pretrained=True,
            freeze_body=freeze_body,
            weights_path=weights_path
        )

        # logging = TensorBoard(log_dir=self.log_dir)
        # checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        #                              monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        # load annotation data as list
        with open(self.train_annotation_path) as f:
            self.train_lines = f.readlines()
        with open(self.valid_annotation_path) as f:
            self.valid_lines = f.readlines()

        self.ready_for_data = True

        # shuffle annotation data for train
        np.random.seed(10101)
        np.random.shuffle(self.train_lines)
        np.random.seed(None)

        num_train = len(self.train_lines)
        num_valid = len(self.valid_lines)

        self.model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        self.train_data_generator = data_generator_wrapper(self.train_lines,
                                                      batch_size,
                                                      self.input_shape,
                                                      self.anchors,
                                                      self.num_classes)
        self.valid_data_generator = data_generator_wrapper(self.valid_lines,
                                                      batch_size,
                                                      self.input_shape,
                                                      self.anchors,
                                                      self.num_classes)

        first_valid_result = self.model.evaluate_generator(self.valid_data_generator, max(1, num_valid // batch_size))
        print("[Epoch 0] (at the pretrained stage) [validation loss : %.4f]" % first_valid_result)

        start_time = datetime.datetime.now()
        steps_per_epoch = max(1, num_train // batch_size)
        for epoch in range(epochs):
            for batch_i in range(steps_per_epoch):

                # train manually
                X, y = next(self.train_data_generator)
                train_loss = self.model.train_on_batch(X, y)

                elapsed_time = datetime.datetime.now() - start_time
                print("[Epoch %d] [Batch %d/%d] [YOLO loss: %f] time: %s"
                      % (epoch, batch_i, steps_per_epoch, train_loss, elapsed_time))

            valid_loss = self.model.evaluate_generator(self.valid_data_generator, max(1, num_valid // batch_size))
            print("[Epoch %d] [validation loss: %f]" % (epoch, valid_loss))
            self.evaluate(epoch)

            if not os.path.exists("{}/models".format(self.output_dir)):
                os.makedirs("{}/models".format(self.output_dir), exist_ok=True)

            self.model.save("{}/models/ep%3d-val_loss%.3f.h5".format(self.output_dir, epoch, valid_loss))
            self.model.save_weights("{}/models/ep%3d-val_loss%.3f.weights".format(self.output_dir, epoch, valid_loss))

    def train_automatically(self, freeze_body=2, weights_path='model_data/yolo_weights.h5', batch_size=32):

        self.model, self.boxes, self.scores, self.classes = self.create_model(
            load_pretrained=True,
            freeze_body=freeze_body,
            weights_path=weights_path
        )

        logging = TensorBoard(log_dir=self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        # load annotation data as list
        with open(self.train_annotation_path) as f:
            self.train_lines = f.readlines()
        with open(self.valid_annotation_path) as f:
            self.valid_lines = f.readlines()

        self.ready_for_data = True

        # shuffle annotation data for train
        np.random.seed(10101)
        np.random.shuffle(self.train_lines)
        np.random.seed(None)

        num_train = len(self.train_lines)
        num_valid = len(self.valid_lines)

        self.model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        self.train_data_generator = data_generator_wrapper(self.train_lines,
                                                           batch_size,
                                                           self.input_shape,
                                                           self.anchors,
                                                           self.num_classes)
        self.valid_data_generator = data_generator_wrapper(self.valid_lines,
                                                           batch_size,
                                                           self.input_shape,
                                                           self.anchors,
                                                           self.num_classes)

        first_valid_result = self.model.evaluate_generator(self.valid_data_generator, max(1, num_valid // batch_size))
        print("Epoch 0 (at the pretrained stage) [validation loss : %.4f]" % first_valid_result)
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_valid, batch_size))
        self.model.fit_generator(self.train_data_generator,
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=self.valid_data_generator,
                            validation_steps=max(1, num_valid // batch_size),
                            epochs=100,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])

        self.model.save_weights(self.log_dir + 'trained_weights.h5')
        print("training finished correctly")

    def evaluate(self, epoch, out_path_base="evaluation"):
        # for train data
        image_ids_gt, groundtruth_boxes_list, groundtruth_classes_list = self.make_groundtruth_list(self.train_lines)
        image_ids_dt, detection_boxes_list, detection_scores_list, detection_classes_list = self.make_detection_list(self.train_lines)

        groundtruth_list = ExportGroundtruthToCOCO(
            image_ids_gt, groundtruth_boxes_list, groundtruth_classes_list, self.categories_dict
        )
        detections_list = ExportDetectionsToCOCO(
            image_ids_dt, detection_boxes_list, detection_scores_list, detection_classes_list, self.categories_dict
        )

        groundtruth = COCOWrapper(groundtruth_list)
        detections = groundtruth.LoadAnnotations(detections_list)
        evaluator = COCOEvalWrapper(groundtruth, detections)
        summary_metrics_train, per_cat_ap_train = evaluator.ComputeMetrics()

        # for validation data
        image_ids_gt, groundtruth_boxes_list, groundtruth_classes_list = self.make_groundtruth_list(self.valid_lines)
        image_ids_dt, detection_boxes_list, detection_scores_list, detection_classes_list = self.make_detection_list(self.valid_lines)

        groundtruth_list = ExportGroundtruthToCOCO(
            image_ids_gt, groundtruth_boxes_list, groundtruth_classes_list, self.categories_dict
        )
        detections_list = ExportDetectionsToCOCO(
            image_ids_dt, detection_boxes_list, detection_scores_list, detection_classes_list, self.categories_dict
        )

        groundtruth = COCOWrapper(groundtruth_list)
        detections = groundtruth.LoadAnnotations(detections_list)
        evaluator = COCOEvalWrapper(groundtruth, detections)
        summary_metrics_valid, per_cat_ap_valid = evaluator.ComputeMetrics()

        print(summary_metrics_valid)

        # # print evaluated metrics to std_out
        # print("Evaluation on TRAIN DATA -----")
        # for metric_name, metric_value in summary_metrics_train:
        #     print(metric_name + " : " + metric_value)
        # print("Evaluation on VALIDATION DATA -----")
        # for metric_name, metric_value in summary_metrics_valid:
        #     print(metric_name + " : " + metric_value)

        # save evaluated metrics to single csv file
        outlines = [epoch] + list(summary_metrics_train.values()) + list(summary_metrics_valid.values())
        self.training_log_file.write(",".join([str(x) for x in outlines])+"\n")
        self.training_log_file.flush()

        # save evaluated metrics to csv file by epoch
        summary_filename = out_path_base + "_summary-ep%3d.csv" % epoch
        summary_path = os.path.join(self.log_dir, summary_filename)

        # per_cat_filename = out_path_base + "_per_category-ep%3d.csv" % epoch
        # per_cat_path = os.path.join(self.log_dir, per_cat_filename)

        write_csv(summary_path, summary_metrics_valid)
        # write_csv(per_cat_path, per_cat_ap_valid)

        # plot training process
        names = ["epoch"] \
                + [n + "_train" for n in list(summary_metrics_train.keys())] \
                + [n + "_valid" for n in list(summary_metrics_valid.keys())]
        d = pd.read_csv(os.path.join(self.log_dir, "progress.csv"), header=None, names=names)
        if d.shape[0]==1: return
        d = d.interpolate()
        p = d.plot(x="epoch",
                   y=['Precision/mAP_train', 'Precision/mAP_valid',
                      'Precision/mAP@.50IOU_train', 'Precision/mAP@.50IOU_valid',
                      'Recall/AR@1_train', 'Recall/AR@1_valid'])
        fig = p.get_figure()
        fig.savefig(os.path.join(self.log_dir, "/graph.png"))
        plt.close()

    def sample_detection(self, epoch):
        os.makedirs("%s/sample_images" % self.output_dir, exist_ok=True)
        # r, c = 1, 2
        samples = self.valid_lines[:3]
        for idx, line in samples:
            line = line.split()
            image_gt = Image.open(line[0])
            image_dt = Image.open(line[0])
            line_boxes_ids = [l.split(",") for l in line[1:]]
            gt_boxes = np.array([np.array([float(xmin), float(ymin), float(xmax), float(ymax)])
                              for xmin, ymin, xmax, ymax, _ in line_boxes_ids])
            gt_classes = np.array([int(class_id) for _, _, _, _, class_id in line_boxes_ids])

            new_image_size = (image_gt.width - (image_gt.width % 32),
                              image_gt.height - (image_gt.height % 32))
            boxed_image = letterbox_image(image_gt, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model_body.input: image_data,
                    self.input_image_shape: [image_gt.size[1], image_gt.size[0]],
                    K.learning_phase(): 0
                })

            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image_gt.size[1] + 0.5).astype('int32'))
            thickness = (image_gt.size[0] + image_gt.size[1]) // 300

            # ground truth result
            for i, c in reversed(line(enumerate(gt_classes))):
                groundtruth_class = self.class_names[c]
                box = gt_boxes[i]
                label = '{}'.format(groundtruth_class)
                draw = ImageDraw.Draw(image_gt)
                label_size = draw.textsize(label, font)
                left, top, right, bottom = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image_gt.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image_gt.size[0], np.floor(right + 0.5).astype('int32'))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            # for detection result
            for i, c in reversed(line(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image_dt)
                label_size = draw.textsize(label, font)
                left, top, right, bottom = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image_gt.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image_gt.size[0], np.floor(right + 0.5).astype('int32'))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            imgs = [np.array(image_gt, dtype='float32'), np.array(image_dt, dtype='float32')]

            titles = ["Ground Truth", "Detected by YOLOv3"]

            fig, axs = plt.subplots(1, 2)
            for i in range(2):
                axs[0, 0].imshow(imgs[i])
                axs[0, 0].set_title(titles[i])
                axs[0, 0].axis('off')
            fig.savefig("%s/images/sample_ep%3d-%d" % (self.output_dir, epoch, idx))
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='retraining.py',
        description='Retraining yolov3 on your own dataset.',
        add_help=True,
    )

    parser.add_argument('--train_path', type=str, help='path to train annotation txt path', required=True)
    parser.add_argument('--valid_path', type=str, help='path to validation annotation txt path', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--log_dir', type=str, default='logs/000', help='path to log directory')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/000', help='path to output directory')
    parser.add_argument('--classes_path', type=str, default='model_data/coco_classes.txt')
    parser.add_argument('--anchors_path', type=str, default='model_data/yolo_anchors.txt')
    parser.add_argument('-s', '--input_size', type=str, default='416x416', help='HEIGHT x WIDTH')
    parser.add_argument('--pretrained_weights', type=str, default='model_data/yolo_weights.h5', help='path to pretrained yolov3 weights')
    parser.add_argument('-n', '--gpu_num', type=int, default=1, help='number of GPU')

    args = parser.parse_args()
    train_annotation_path = args.train_path
    valid_annotation_path = args.valid_path
    batch_size = args.batch_size
    log_dir = args.log_dir
    output_dir = args.output_dir
    classes_path = args.classes_path
    anchors_path = args.anchors_path
    height, width = args.input_size.split("x")
    input_shape = (int(height), int(width))
    weights_path = args.pretrained_weights
    gpu_num = args.gpu_num

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainer = Trainer(
        train_annotation_path=train_annotation_path,
        valid_annotation_path=valid_annotation_path,
        log_dir=log_dir,
        output_dir = output_dir,
        classes_path=classes_path,
        anchors_path=anchors_path,
        input_shape=input_shape,
        gpu_num=gpu_num
    )

    trainer.train(
        weights_path=weights_path,
        batch_size=batch_size
    )
