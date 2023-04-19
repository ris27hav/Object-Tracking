import numpy as np


class Detection(object):
    """Represents a bounding box detection in a single image."""
    def __init__(self, tlwh, confidence, class_name, feature):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=float)

    def get_class(self):
        return self.class_name

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def convert_boxes_to_tlwh(boxes):
    """Converts the bounding boxes from [xmin, ymin, xmax, ymax]
    to [tl_x, tl_y, w, h]."""
    output_boxes = []
    for box in boxes:
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        box = box.astype(int)
        output_boxes.append(box)
    return np.array(output_boxes)