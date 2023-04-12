import numpy as np

def detect_objects(model, image):
    """Predicts the bounding boxes and class probabilities for a given image."""
    output = model(image)
    output = output.pandas().xyxy[0]
    boxes = np.array(output[['xmin', 'ymin', 'xmax', 'ymax']].values)
    scores = np.array(output['confidence'].values)
    classes = np.array(output['class'].values)
    names = np.array(output['name'].values)
    return boxes, scores, classes, names