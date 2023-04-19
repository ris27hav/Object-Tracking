import numpy as np
from . import matching

def penalty_factor(bbox, candidates):
    """Compute the penalty factor between a bounding box and
    a set of candidates."""
    candidate_heights = candidates[:, 3]
    candidate_heights = np.maximum(candidate_heights, 1e-6)
    bbox_height = bbox[3]
    bbox_height = np.maximum(bbox_height, 1e-6)
    candidate_widths = candidates[:, 2]
    candidate_widths = np.maximum(candidate_widths, 1e-6)
    bbox_width = bbox[2]
    bbox_width = np.maximum(bbox_width, 1e-6)
    penalty = np.maximum(
            candidate_heights / bbox_height,
            bbox_height / candidate_heights
        ) + np.maximum(
            candidate_widths / bbox_width,
            bbox_width / candidate_widths
        )
    return penalty


def iou(bbox, candidates):
    """Compute the intersection over union between a bounding box and a set of
    candidates."""
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    iou = area_intersection / (area_bbox + area_candidates - area_intersection)
    return iou


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """Computes the matching cost between tracks and detections using the
    intersection over union metric."""
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = matching.INFTY_COST
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - (iou(bbox, candidates)/penalty_factor(bbox, candidates))
    return cost_matrix
