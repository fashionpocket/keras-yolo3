"""
This class is responsible for the whole experimentals
"""

# TODO: implement unfinished
import argparse
import os

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data


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

    def __init__(self,
                 train_annotation_path='train.txt',
                 valid_annotation_path='valid.txt',
                 log_dir='logs/000/',
                 classes_path='model_data/coco_classes.txt',
                 anchors_path='model_data/yolo_anchors.txt',
                 input_shape=(416, 416)
                 ):
        self.train_annotation_path = train_annotation_path
        self.valid_annotation_path = valid_annotation_path
        self.log_dir = log_dir
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.class_names = self.get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = self.get_anchors(self.anchors_path)

        self.input_shape = input_shape

        self.is_trained = False

    def get_classes(self, classes_path):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def create_model(self, input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                     weights_path='model_data/yolo_weights.h5'):
        '''create the training model'''
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))
        h, w = input_shape
        num_anchors = len(anchors)

        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                               num_anchors // 3, num_classes + 5)) for l in range(3)]

        model_body = yolo_body(image_input, num_anchors // 3, num_classes)
        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (185, len(model_body.layers) - 3)[freeze_body - 1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model

    def train_automatically(self, freeze_body=2, weights_path='model_data/yolo_weights.h5', batch_size=32):
        model = self.create_model(
            self.input_shape,
            self.anchors,
            self.num_classes,
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
            train_lines = f.readlines()
        with open(self.valid_annotation_path) as f:
            valid_lines = f.readlines()

        # shuffle annotation data for train
        np.random.seed(10101)
        np.random.shuffle(train_lines)
        np.random.seed(None)

        num_train = len(train_lines)
        num_valid = len(valid_lines)

        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        train_data_generator = data_generator_wrapper(train_lines,
                                                           batch_size,
                                                           self.input_shape,
                                                           self.anchors,
                                                           self.num_classes)
        valid_data_generator = data_generator_wrapper(valid_lines,
                                                           batch_size,
                                                           self.input_shape,
                                                           self.anchors,
                                                           self.num_classes)

        first_valid_result = model.evaluate_generator(valid_data_generator, max(1, num_valid // batch_size))
        print("Epoch 0 (at the pretrained stage) [validation loss : %.4f]" % first_valid_result[0])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_valid, batch_size))
        model.fit_generator(train_data_generator,
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=valid_data_generator,
                            validation_steps=max(1, num_valid // batch_size),
                            epochs=100,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])

        model.save_weights(self.log_dir + 'trained_weights.h5')
        print("training finished correctly")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='retraining.py',
        description='Retraining yolov3 on your own dataset.',
        add_help=True,
    )

    parser.add_argument('--train_path', type=str, help='path to train annotation txt path', required=True)
    parser.add_argument('--valid_path', type=str, help='path to validation annotation txt path', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--log_dir', type=str, default='logs/000', help='path to log directory')
    parser.add_argument('--classes_path', type=str, default='model_data/coco_classes.txt')
    parser.add_argument('--anchors_path', type=str, default='model_data/yolo_anchors.txt')
    parser.add_argument('-s', '--input_size', type=str, default='416x416', help='HEIGHT x WIDTH')
    parser.add_argument('--pretrained_weights', type=str, default='model_data/yolo_weights.h5', help='path to pretrained yolov3 weights')

    args = parser.parse_args()
    train_annotation_path = args.train_path
    valid_annotation_path = args.valid_path
    batch_size = args.batch_size
    log_dir = args.log_dir
    classes_path = args.classes_path
    anchors_path = args.anchors_path
    height, width = args.input_size.split("x")
    input_shape = (int(height), int(width))
    weights_path = args.pretrained_weights

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    trainer = Trainer(
        train_annotation_path=train_annotation_path,
        valid_annotation_path=valid_annotation_path,
        log_dir=log_dir,
        classes_path=classes_path,
        anchors_path=anchors_path,
        input_shape=input_shape
    )

    trainer.train_automatically(
        weights_path=weights_path,
        batch_size=batch_size
    )
