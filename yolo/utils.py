import numpy as np

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