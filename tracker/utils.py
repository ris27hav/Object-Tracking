import numpy as np

def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress non-maximal boxes."""
    # return empty list if there are no boxes
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(float)
    remaining_boxes_ids = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    while len(idxs) > 0:
        # choose the bounding box with the highest score
        last = len(idxs) - 1
        i = idxs[last]
        remaining_boxes_ids.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        # overlap greater than the provided overlap threshold
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return remaining_boxes_ids


def cosine_dist(x, y):
    """Computes cosine distance between two arrays of vectors."""
    x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
    y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)
    distances = 1. - np.dot(x, y.T)
    dist = distances.min(axis=0)
    return dist


def euclidean_dist(x, y):
    """Computes euclidean distance between two arrays of vectors."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.zeros((len(x), len(y)))
    x2 = np.square(x).sum(axis=1)
    y2 = np.square(y).sum(axis=1)
    distances = -2. * np.dot(x, y.T) + x2[:, None] + y2[None, :]
    distances = np.clip(distances, 0., float(np.inf))
    dist = np.maximum(0.0, distances.min(axis=0))
    return dist


class NNDistanceMetric(object):
    """Computes distance between two arrays of embeddings."""
    def __init__(self, metric, matching_threshold):
        if metric == "euclidean":
            self._metric = euclidean_dist
        else:
            self._metric = cosine_dist
        self.matching_threshold = matching_threshold
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data."""
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Computes distance between features and targets."""
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
