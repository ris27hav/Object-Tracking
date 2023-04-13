from torch.hub import load as torch_hub_load
from ultralytics import YOLO

def load_model(model_path, yolo_version=5):
    """Loads the yolov5/v8 model from the path."""
    if yolo_version == 5:
        model = torch_hub_load('ultralytics/yolov5', 'custom',
                            path=model_path, verbose=False)
    elif yolo_version == 8:
        model = YOLO(model_path)
    else:
        raise ValueError(f'Invalid yolo version: {yolo_version}')
    return model