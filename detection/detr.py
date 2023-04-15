import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.hub import load as torch_hub_load


def load_model(model_path):
    """Loads the yolov5 model from the path."""
    model = torch_hub_load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval()
    return model

def preprocess(img):
    """Preprocesses the image for the model."""
    img = Image.fromarray(np.uint8(img)).convert('RGB')
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img

def box_cxcywh_to_xyxy(x):
    """Converts the bounding boxes from [cx, cy, w, h] to [xmin, ymin, xmax, ymax]."""
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    """Rescales the bounding boxes from [0; 1] to image scales."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def get_classes():
    classes = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    return classes

def detect_objects(model, image, class_names, keep_classes=None):
    """Predicts the bounding boxes and class probabilities for a given image."""
    image = preprocess(image)
    output = model(image)

    # keep only predictions with 0.85+ confidence
    probas = output['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.85
    bboxes_scaled = rescale_bboxes(output['pred_boxes'][0, keep], image.shape[-2:])

    boxes, scores, classes, names = [], [], [], []
    for p, bbox in zip(probas[keep], bboxes_scaled):
        boxes.append(bbox.detach().cpu().numpy())
        scores.append(p.max().item())
        classes.append(p.argmax().item())
        names.append(class_names[p.argmax().item()])
    
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