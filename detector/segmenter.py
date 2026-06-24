import os
import cv2
import torch
import numpy as np

from .config import SAM_AVAILABLE, DEFAULT_SEGMENTER_CONFIG
from .preprocessing import preprocess_for_segmentation, preprocess_for_classification


def _compute_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / max(union, 1)


class HieroglyphSegmenter:
    def __init__(self, sam_checkpoint_path, config=None, device=None):
        if not SAM_AVAILABLE:
            raise ImportError("The 'segment_anything' package is required.")
        if not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint_path}")

        from segment_anything import sam_model_registry, SamPredictor

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or dict(DEFAULT_SEGMENTER_CONFIG)

        print(f"Loading SAM (ViT-H) onto {self.device}...")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
        sam.to(self.device).eval()
        self.predictor = SamPredictor(sam)
        print("SAM loaded successfully.")

    def _generate_point_prompts(self, gray_img):
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        binary_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 8
        )
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        binary_mask = cv2.subtract(binary_mask, horizontal_lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_points = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) < self.config['min_area'] * 0.4:
                continue
            if (w / float(max(h, 1))) > 3.5 or (w / float(max(h, 1))) < 0.2:
                continue
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                raw_points.append([
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"])
                ])
        if not raw_points:
            return []
        proximity = self.config.get('point_proximity', 15)
        sorted(raw_points, key=lambda p: p[1])
        kept = [raw_points[0]]
        for pt in raw_points[1:]:
            too_close = False
            for kp in kept:
                if abs(pt[0] - kp[0]) < proximity and abs(pt[1] - kp[1]) < proximity:
                    too_close = True
                    break
            if not too_close:
                kept.append(pt)
        return kept

    def _predict_sam_masks(self, image_rgb, point_prompts):
        h, w = image_rgb.shape[:2]
        if not point_prompts:
            return np.zeros((h, w), dtype=np.uint8), []
        self.predictor.set_image(image_rgb)
        collected = []
        nms_iou = self.config.get('nms_iou_threshold', 0.5)
        with torch.no_grad():
            for point in point_prompts:
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([point], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True
                )
                best_mask, best_score = None, -1
                for i in range(len(masks)):
                    y_idx, x_idx = np.where(masks[i] > 0)
                    if len(y_idx) == 0:
                        continue
                    if (np.count_nonzero(masks[i]) < self.config['max_area'] and
                        (x_idx.max() - x_idx.min()) < self.config['max_width'] and
                        (y_idx.max() - y_idx.min()) < self.config['max_height'] and
                        scores[i] > best_score):
                        best_mask, best_score = masks[i], scores[i]
                if best_mask is not None:
                    collected.append({
                        'mask': (best_mask.astype(np.uint8) * 255),
                        'score': float(best_score),
                        'centroid': point,
                    })
        if not collected:
            return np.zeros((h, w), dtype=np.uint8), []
        collected.sort(key=lambda x: x['score'], reverse=True)
        surviving = []
        for item in collected:
            suppressed = False
            for existing in surviving:
                iou = _compute_iou(item['mask'] > 0, existing['mask'] > 0)
                if iou > nms_iou:
                    suppressed = True
                    break
            if not suppressed:
                surviving.append(item)
        unified = np.zeros((h, w), dtype=np.uint8)
        for item in surviving:
            unified = cv2.bitwise_or(unified, item['mask'])
        return unified, surviving

    @staticmethod
    def _mask_to_tight_bbox(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x, y = xs.min(), ys.min()
        w, h = xs.max() - x + 1, ys.max() - y + 1
        return (int(x), int(y), int(w), int(h))

    def _filter_contour(self, contour):
        cfg = self.config
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        rect_area = w * h
        if not (cfg['min_area'] <= area <= cfg['max_area']):
            return False, None
        if not (cfg['min_width'] <= w <= cfg['max_width']):
            return False, None
        if not (cfg['min_height'] <= h <= cfg['max_height']):
            return False, None
        aspect_ratio = float(w) / max(h, 1)
        if not (cfg['min_aspect_ratio'] <= aspect_ratio <= cfg['max_aspect_ratio']):
            return False, None
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and (float(area) / hull_area) < cfg['min_solidity']:
            return False, None
        if rect_area > 0 and (float(area) / rect_area) < cfg['min_extent']:
            return False, None
        return True, (x, y, w, h)

    def _sort_boxes_by_reading_order(self, boxes):
        if not boxes:
            return []
        boxes_array = np.array(boxes)
        centers_y = boxes_array[:, 1] + boxes_array[:, 3] / 2
        sorted_indices = np.argsort(centers_y)
        lines, current_line = [], [sorted_indices[0]]
        current_y = centers_y[sorted_indices[0]]
        for idx in sorted_indices[1:]:
            if abs(centers_y[idx] - current_y) < self.config['line_threshold']:
                current_line.append(idx)
            else:
                current_line.sort(key=lambda i: boxes_array[i, 0])
                lines.append(current_line)
                current_line, current_y = [idx], centers_y[idx]
        if current_line:
            current_line.sort(key=lambda i: boxes_array[i, 0])
            lines.append(current_line)
        return [tuple(boxes_array[idx].astype(int)) for line in lines for idx in line]

    def extract_symbols(self, image_path):
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise ValueError(f"Image not found: {image_path}")

        seg_img = preprocess_for_segmentation(original_bgr)
        class_img = preprocess_for_classification(original_bgr)

        h_img, w_img = seg_img.shape[:2]
        gray_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        point_prompts = self._generate_point_prompts(gray_img)
        unified_mask, surviving = self._predict_sam_masks(
            cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB), point_prompts
        )
        if surviving:
            boxes = []
            for item in surviving:
                bbox = self._mask_to_tight_bbox(item['mask'])
                if bbox is not None:
                    x, y, w, h = bbox
                    area = np.count_nonzero(item['mask'])
                    if (self.config['min_area'] <= area <= self.config['max_area'] and
                        self.config['min_width'] <= w <= self.config['max_width'] and
                        self.config['min_height'] <= h <= self.config['max_height']):
                        boxes.append(bbox)
        else:
            boxes = []
        sorted_boxes = self._sort_boxes_by_reading_order(boxes)
        crops = []
        base_pad = self.config['crop_padding']
        for x, y, w, h in sorted_boxes:
            adaptive_pad = max(base_pad, min(w, h) // 6)
            crops.append((
                class_img[
                    max(0, y - adaptive_pad):min(h_img, y + h + adaptive_pad),
                    max(0, x - adaptive_pad):min(w_img, x + w + adaptive_pad)
                ],
                (x, y, w, h)
            ))
        return seg_img, class_img, original_bgr, unified_mask, crops, len(point_prompts), len(boxes)
