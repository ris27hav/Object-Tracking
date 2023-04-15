import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from resnet.generate_detections import create_box_encoder

from yolov5.model import load_model
from yolov5.inference import detect_objects
from yolov5.utils import convert_boxes_to_tlwh


from detr.inference import *

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


MAX_COSINE_DISTANCE = 0.5
NN_BUDGET = None
NMS_MAX_OVERLAP = 0.8

# load yolov5 for object detection
yolo = load_model('./yolov5/weights/yolov5s.pt')
detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)

# load resnet18 for feature extraction
encoder = create_box_encoder(batch_size=32)

# load deep sort tracker for object tracking
metric = nn_matching.NearestNeighborDistanceMetric(
    'cosine', MAX_COSINE_DISTANCE, NN_BUDGET
)
tracker = Tracker(metric)

# input video
vid = cv2.VideoCapture('test.mp4')

# output video
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/videos/results.avi', codec, vid_fps, (vid_width, vid_height))


while True:
    # read a frame from video
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    t1 = time.time()

    # detect objects
    # boxes, scores, classes, names = detect_objects(yolo, img)

    # print the shape of the output
    # print(boxes.shape, scores.shape, classes.shape, names.shape)

    try:
        boxes, scores, classes, names = final_detect(img,detr,transform)
    
    except:
        # img=Image.fromarray(img)
        # print(img1.size)
        # print(img.shape)
        boxes, scores, classes, names = final_detect(img,detr,transform)
    # encode detected objects

    # print(boxes.shape, scores.shape, classes.shape, names.shape)

    converted_boxes = convert_boxes_to_tlwh(boxes)
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores, names, features)]

    # run non-maxima supression
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxes, classes, NMS_MAX_OVERLAP, scores)
    detections = [detections[i] for i in indices]

    # update tracker
    tracker.predict()
    tracker.update(detections)

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