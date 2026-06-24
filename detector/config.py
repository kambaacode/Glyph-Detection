import os
import json
import torch
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAPPING_PATH = r"C:\Users\moham\OneDrive\Desktop\grad-documentation\project\H2A_cv\classifier\efficientNetV2s\class_mapping.json"
CLASSIFIER_WEIGHTS = r"C:\Users\moham\OneDrive\Desktop\grad-documentation\project\H2A_cv\classifier\efficientNetV2s\weights\best_efficientNets_hieroglyph.pth"
SAM_CHECKPOINT = r"C:\Users\moham\OneDrive\Desktop\grad-documentation\project\H2A_cv\detector\sam_weights\sam_vit_h_4b8939.pth"
TARGET_IMAGE = r"C:\Users\moham\OneDrive\Desktop\grad-documentation\project\H2A_cv\glyph1.webp"

inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    first_key = list(loaded_data.keys())[0]
    if not first_key.isdigit():
        idx_to_class = {int(v): k for k, v in loaded_data.items()}
    else:
        idx_to_class = {int(k): v for k, v in loaded_data.items()}
    num_classes = len(idx_to_class)
    print(f"Configured {num_classes} classes from JSON map.")
else:
    raise FileNotFoundError(f"class_mapping.json not found: {MAPPING_PATH}")

DEFAULT_SEGMENTER_CONFIG = {
    'min_area': 250, 'max_area': 15000,
    'min_width': 10, 'min_height': 10,
    'max_width': 180, 'max_height': 180,
    'max_aspect_ratio': 6.0, 'min_aspect_ratio': 0.10,
    'min_solidity': 0.10, 'min_extent': 0.10,
    'crop_padding': 8, 'line_threshold': 55,
    'point_proximity': 15, 'nms_iou_threshold': 0.5,
}

SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass
