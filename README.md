# yolo3 video object detection
ビデオデータの持つtempralな情報を活かし、yolo-v3の精度向上を狙う。
1. （online）ビデオデータの持つtemporalな情報を用い、フレーム間のbboxのアサインメントを行うことで精度向上を狙う。
2. （offline）ビデオベースの定量的指標を用いた自動アノテーションによる自己学習の比較評価提案する **"auto annotations"** が、baselineとなる **"YOLOv3 pretrained on COCO dataset"** よりも性能が高く、manual annotationsと同等(?)の性能を示すことを提示する。
3. （offline）データセットのドメインへの過学習抑制に向けて、**正則化**を加えた場合の比較評価。上で提案した**self-trained**の手法に正則化を加えることで性能が上昇することを提示する。
## How to use?
### Data Annotation
Annotation for **COCO 2017**, which is used for pre_training and validation.
```sh
# Generate MS COCO 2017 train Annotations. (instances)
$ python coco_annotation.py -l /home/CVPRanno/data/coco/annotations/instances_train2017.json \ 
                            -i /home/CVPRanno/data/coco/train2017 \ 
                            -o annotation/coco_train.txt
# Generate MS COCO 2017 validation Annotations. (instances)
$ python coco_annotation.py -l /home/CVPRanno/data/coco/annotations/instances_val2017.json \ 
                            -i /home/CVPRanno/data/coco/val2017 \ 
                            -o annotation/coco_valid.txt
```
Annotation for **KITTI**, which is used for **self-training** **(proposed method)**
```
# Generate KITTI Annotations. (inference)
$ python kitti_annotation.py -m inference \ 
                             -i /home/CVPRanno/data/kitti/training/image_02 \ 
                             -l /home/CVPRanno/data/kitti/training/image_02 \ 
                             -o kitti_inference_train.txt \ 
                             -s 1584x480
```
### Retraining
We show that the performance of yolo-v3, which was pre-trained on MS COCO 2017, is improved on MS COCO 2017 validation data by self-training using KITTI data (different domains).
```sh
$ python retraining.py --train_path annotation/kitti_inference_train.txt \ 
                       --valid_path annotation/coco_valid.txt \ 
                       --pretrained_weights model_data/yolov3_pretrained.weights \ 
                       -b 32 -n 3
```
<summary>Original <a href="https://github.com/qqwweee/keras-yolo3"><code>README.md</code></a>
<details>
# keras-yolo3
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
## Introduction
A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).
---
## Quick Start
1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.
```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```
For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.
### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]
positional arguments:
  --input        Video input path
  --output       Video output path
optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---
4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).
## Training
1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.
3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.
If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py
---
## Some issues to know
1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0
2. Default anchors are used. If you use your own anchors, probably some changes are needed.
3. The inference result is not totally the same as Darknet but the difference is small.
4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.
5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.
6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.
7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
</details>
</summary>
