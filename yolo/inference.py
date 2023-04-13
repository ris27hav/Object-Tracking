import numpy as np

def detect_objects(model, image, yolo_version=5, keep_classes=None):
    """Predicts the bounding boxes and class probabilities for a given image."""
    if yolo_version == 5:
        output = model(image)
        output = output.pandas().xyxy[0]
        boxes = output[['xmin', 'ymin', 'xmax', 'ymax']].values
        scores = output['confidence'].values
        classes = output['class'].values
        names = output['name'].values
    elif yolo_version == 8:
        output = model(image, verbose=False)
        boxes, scores, classes, names = [], [], [], []
        for i in range(len(output[0].boxes.data)):
            data = output[0].boxes.data[i]
            boxes.append(data[:4].cpu().numpy().tolist())
            scores.append(float(data[4].cpu()))
            classes.append(int(data[5].cpu()))
            names.append(output[0].names[int(data[5].cpu())])
    else:
        raise ValueError(f'Invalid yolo version: {yolo_version}')
    
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