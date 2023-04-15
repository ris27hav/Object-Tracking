import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18     # resnet18 originally


def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()
    return hook


def add_hooks(model, outputs, output_layer_names):
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(
            get_activation(outputs, output_layer_name)
        )


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=True):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = [self.outputs[output_layer_name]
                       for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals


class BBResNet18(object):
    def __init__(self, batch_size=32):
        self.image_shape = (224, 224)
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = resnet18(pretrained=True)
        self.model.eval()
        self.model = ModelWrapper(self.model, ['avgpool'], True)
        self.model.eval()
        self.model.to(self.device)

    def feature_extraction(self, x:np.ndarray):
        x = torch.from_numpy(x).to(self.device)
        all_features = None
        with torch.no_grad():
            for i in range(0, x.shape[0], self.batch_size):
                batch = x[i:i+self.batch_size]
                features = self.model(batch)
                features = features.cpu().numpy()
                if i == 0:
                    all_features = features
                else:
                    all_features = np.concatenate((all_features, features), axis=0)
        all_features = all_features.reshape(all_features.shape[0], -1)
        return all_features