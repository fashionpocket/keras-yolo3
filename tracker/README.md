# Trackr
Straightening the pretrained yolo-v3 inference by using Tracker.

We have to cope with the fo

- confidence `score` is low.
- `class label` is different from the surrounding (t-1, t+1) classes.
- bbox is not detected.
- bbox is far away from the surrounding (t-1, t+1) bboxes.

### OpenCV



