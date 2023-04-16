import cv2
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
# frameID, trackID, bbox_top, bbox_left, bbox_width, bbox_height, confident score, class, unknown

from resnet.generate_detections import create_box_encoder

from yolov5.model import load_model
from yolov5.inference import detect_objects
from yolov5.utils import convert_boxes_to_tlwh

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


MAX_COSINE_DISTANCE = 0.10
NN_BUDGET = None
NMS_MAX_OVERLAP = 0.8
class_dict = {
    'Pedestrian': 1,
    'Person on vehicle': 2,
    'Car': 3,
    'Bicycle': 4,
    'Motorbike': 5,
    'Non-motorized vehicle': 6,
    'Static person': 7,
    'Distractor': 8,
    'Occluder': 9,
    'Occluder on the ground': 10,
    'Occluder full': 11,
    'Reflection': 12
}
class_dict1 = {
    'person': 1,
    'car': 3,
    'bicycle': 4,
    'motorcycle': 5,
    'Non-motorized vehicle': 6,
    'Person': 7,
    'Distractor': 8,
    'Occluder': 9,
    'Occluder on the ground': 10,
    'Occluder full': 11,
    'Reflection': 12
}
# load yolov5 for object detection
yolo = load_model('./yolov5/weights/yolov5s.pt')
# yolo = load_model('../yolov5/runs/train/exp2/weights/best.pt')
yolo.conf = 0.4

# load resnet18 for feature extraction
encoder = create_box_encoder(batch_size=32)

# load deep sort tracker for object tracking
metric = nn_matching.NearestNeighborDistanceMetric(
    'cosine', MAX_COSINE_DISTANCE, NN_BUDGET
)
tracker = Tracker(metric)

# input video
# vid = cv2.VideoCapture('./data/test.mp4')
# input video
vid_path = "../data/train/MOT16-09/img1/"

# output video
codec = cv2.VideoWriter_fourcc(*'XVID')
# vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
num=1
vid_fps = int(30)
# vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_width, vid_height = int(1920), int(1080)
out = cv2.VideoWriter('./data/videos/results.avi', codec, vid_fps, (vid_width, vid_height))
# out = cv2.VideoWriter('./data/videos/results.avi', codec, vid_fps, (vid_width, vid_height))


while True:
    # read a frame from video
    digits =  str(num).zfill(6)
    # if num > 900:
    #     break
    # num += 1
    img = cv2.imread(vid_path + digits + '.jpg')
    if img is None:
        print('Completed')
        break
    # print(img.shape)
    # _, img = vid.read()
    # if img is None:
    #     print('Completed')
    #     break

    t1 = time.time()

    # detect objects
    boxes, scores, classes, names = detect_objects(yolo, img)
    
    # encode detected objects
    converted_boxes = convert_boxes_to_tlwh(boxes)
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature, track_id=None) for bbox, score, class_name, feature in
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
	
    for track in tracker.tracks:
        # print the prob in a single line
        print(track.prob, end=" ")
    print() 

    # color for different classes
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
    with open('detections.txt', 'a') as f:
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 2:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_dict1.get(class_name) is not None:
                # print("Exists")
                f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{}\n".format(num, track.track_id, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], -1, class_dict1[class_name], -1))
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                
                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
						+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
						(255, 255, 255), 2)
            # confidence = track.confidence
            

    # show fps
    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.imshow('output', img)
    # cv2.resizeWindow('output', 1024, 768)
    out.write(img)
    num+=1

    # press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

# release resources
# vid.release()
out.release()
cv2.destroyAllWindows()


    # draw bounding boxes on the image
    # with open('detections.txt', 'w') as f:
    #     for track in tracker.tracks:
	# 		if not track.is_confirmed() or track.time_since_update > 1:
	# 			continue
	# 		bbox = track.to_tlbr()
	# 		class_name = track.get_class()
	# 		confidence = track.confidence
	# 		f.write("{},{},{},{},{},{},{},{},{}\n".format(num, track.track_id, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], confidence, -1, -1))
	# 		color = colors[int(track.track_id) % len(colors)]
	# 		color = [i * 255 for i in color]

	# 		cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
	# 		cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
	# 					+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
	# 		cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
	# 					(255, 255, 255), 2)