import numpy as np
from ultralytics import YOLO

def load_model():
    """Loads the yolov8 model from the path."""
    model = YOLO('yolov8n.pt')
    return model

def get_classes():
    """Returns the classes that the model can detect."""
    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    return classes

def detect_objects(model, image, class_names, keep_classes=None):
    """Predicts the bounding boxes and class probabilities for a given image."""
    output = model(image, verbose=False)
    boxes, scores, classes, names = [], [], [], []
    for i in range(len(output[0].boxes.data)):
        data = output[0].boxes.data[i]
        boxes.append(data[:4].cpu().numpy().tolist())
        scores.append(float(data[4].cpu()))
        classes.append(int(data[5].cpu()))
        names.append(output[0].names[int(data[5].cpu())])
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    names = np.array(names)
    
    if keep_classes is not None:
        i = 0
        while i < len(boxes):
            if names[i] not in keep_classes:
                boxes = np.delete(boxes, i, axis=0)
                scores = np.delete(scores, i, axis=0)
                classes = np.delete(classes, i, axis=0)
                names = np.delete(names, i, axis=0)
            else:
                i += 1

    return boxes, scores, classes, names