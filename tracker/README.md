## self-corrected yoloTrackr
Straightening the pretrained yolo-v3 inference by using Tracker.

We have to cope with the followings:

- confidence `score` is low.
- `class label` is different from the surrounding (t-1, t+1) classes.
- bbox is not detected.
- bbox is far away from the surrounding (t-1, t+1) bboxes.

### TODO:

1. Create the format class named `BaseTracker`.
2. Add OpenCV Tracker (**Online** : corrected inference)
3. Add OpenCV Tracker (**Offline** : self-annotation.)
4. Add additional Tracker.

#### Ideal Class format

```python
class BaseTracker(YOLO):
    def __init__(self, reliability=0.7):
        self.reliability = reliability
        self.multi_tracker = "INITIALIZATION"

    def add(self, frame, bbox):
        """ Add Initialized Tracker when object appears for the first time. """

    def remove(self, idx):
        """ Remove Tracker when object goes off the screen. """

    def update(self, image):
        """ update(image) -> retval, bboxes
        Update the tracker status and return the bboxes.
        =================================================
        @params bboxes : the tracking result, represent a list of ROIs of the tracked objects.
        @return retval : (bool) Whether tracking is successful.
        @return bboxes : (list) The tracking result, represent a list of ROIs of the tracked objects.
        """
        # YOLO's inference Results.
        yolo_results = self.infer_bounding_box(Image.fromarray(image))
        # Initialize tracking Results.
        tracker_results = [(label,conf,top,bottom,left,right) for (label,conf,top,bottom,left,right) in self.multi_tracker.update(image)]
        current_bboxes = correction_bboxes(yolo_results, tracking_results)
```

#### Bouding Boxes Notations

```txt
[OpenCV]
x,y,w,h = bbox

   (x,y) ------- (x+w,y)
     |              |
     |              |
     |              |
     |              |
  (x,y+h) ----- (x+w,y+h)

[YOLO]
t,b,l,r = bbox

   (l,t) -------- (r,t)
     |              |
     |              |
     |              |
     |              |
   (x,b) -------- (r,b)
```
