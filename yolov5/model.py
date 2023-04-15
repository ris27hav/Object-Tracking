from torch.hub import load as torch_hub_load

def load_model(model_path):
    """Loads the yolov5 model from the path."""
    model = torch_hub_load('ultralytics/yolov5', 'custom',
                           path=model_path, verbose=False)
    return model

