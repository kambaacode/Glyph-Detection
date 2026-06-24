import cv2
import torch
import numpy as np
from PIL import Image


class HieroglyphClassifier:
    def __init__(self, model, idx_to_class, transform, min_confidence=0.25):
        self.model = model
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.min_confidence = min_confidence
        self.device = next(model.parameters()).device
        self.num_classes = len(idx_to_class)
        self.max_entropy = np.log(self.num_classes)

    def classify(self, crop_bgr):
        pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            top_predictions = [
                (self.idx_to_class[idx.item()], prob.item())
                for idx, prob in zip(top3_indices[0], top3_probs[0])
            ]
        return {
            'label': self.idx_to_class[pred.item()],
            'confidence': conf.item(),
            'entropy': entropy.item(),
            'entropy_ratio': entropy.item() / self.max_entropy,
            'top3': top_predictions,
        }

    def is_valid_prediction(self, prediction, max_entropy_ratio=0.65):
        if prediction['confidence'] < self.min_confidence:
            return False, "low_confidence"
        if prediction['entropy_ratio'] > max_entropy_ratio:
            return False, "high_entropy"
        top3 = prediction['top3']
        if len(top3) >= 2 and (top3[0][1] - top3[1][1]) < 0.05:
            return False, "small_gap"
        return True, "valid"
