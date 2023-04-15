import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from embedding.encode_patches import create_box_encoder

from detection.yolov8 import load_model, get_classes, detect_objects
from detection.utils import convert_boxes_to_tlwh

from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.preprocessing import non_max_suppression 
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


MAX_COSINE_DISTANCE = 0.5
NN_BUDGET = None
NMS_MAX_OVERLAP = 0.8
CLASSES_TO_DETECT = ['person', 'car',  'bus']  # Set to None to detect all classes

# load model for object detection
class_names = get_classes()
detection_model = load_model('./detection/weights/yolov8n.pt')

# load resnet18 for feature extraction
encoder = create_box_encoder(batch_size=32)

# load deep sort tracker for object tracking
metric = NearestNeighborDistanceMetric(
    'cosine', MAX_COSINE_DISTANCE, NN_BUDGET
)
tracker = Tracker(metric)

# input video
vid = cv2.VideoCapture('./data/videos/MOT17-12-FRCNN-raw.webm')

# output video
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/videos/results.avi', codec, vid_fps, (vid_width, vid_height))

total_detection_time = 0.
total_encoding_time = 0.
total_nms_time = 0.
total_tracking_time = 0.
total_time = 0.

while True:
    # read a frame from video
    read, img = vid.read()
    if not read:
        break

    t1 = time.time()

    # detect objects
    boxes, scores, classes, names = detect_objects(detection_model, img, class_names, CLASSES_TO_DETECT)
    total_detection_time += time.time() - t1
    t = time.time()
    
    # encode detected objects
    converted_boxes = convert_boxes_to_tlwh(boxes)
    features = encoder(img, converted_boxes)
    total_encoding_time += time.time() - t
    t = time.time()

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores, names, features)]

    # run non-maxima supression
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = non_max_suppression(boxes, classes, NMS_MAX_OVERLAP, scores)
    detections = [detections[i] for i in indices]
    total_nms_time += time.time() - t
    t = time.time()

    # update tracker
    tracker.predict()
    tracker.update(detections)
    total_tracking_time += time.time() - t
    total_time += time.time() - t1

    # color for different classes
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    # draw bounding boxes on the image
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

    # show fps
    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.imshow('output', img)
    # cv2.resizeWindow('output', 1024, 768)
    out.write(img)

    # press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

# release resources
vid.release()
out.release()
cv2.destroyAllWindows()

# print time taken by each part
print('Detection Time: {:.3f}'.format(total_detection_time))
print('Encoding Time: {:.3f}'.format(total_encoding_time))
print('NMS Time: {:.3f}'.format(total_nms_time))
print('Tracking Time: {:.3f}'.format(total_tracking_time))
print('Total Time: {:.3f}'.format(total_time))