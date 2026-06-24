import os
import torch
import torch.nn as nn
from torchvision import models

from .config import DEVICE, CLASSIFIER_WEIGHTS


def _infer_num_classes(state_dict):
    for key in ('classifier.1.4.bias', 'classifier.1.bias', 'fc.weight', 'fc.5.bias'):
        if key in state_dict:
            return state_dict[key].shape[0]
    for key in state_dict:
        if key.endswith('.weight') and 'classifier' in key:
            return state_dict[key].shape[0]
    raise RuntimeError("Could not infer num_classes from state_dict keys")


def build_model(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_classifier(weights_path=None):
    path = weights_path or CLASSIFIER_WEIGHTS
    if not os.path.exists(path):
        raise FileNotFoundError(f"Classifier weights not found: {path}")

    state = torch.load(path, map_location=DEVICE)
    num_classes = _infer_num_classes(state)
    model = build_model(num_classes)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    print(f"EfficientNetV2-S loaded ({num_classes} classes).")
    return model
