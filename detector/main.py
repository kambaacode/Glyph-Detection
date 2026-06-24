import os
import cv2
import traceback

from .config import DEVICE, SAM_CHECKPOINT, TARGET_IMAGE, idx_to_class, inference_transforms
from .models import load_classifier
from .segmenter import HieroglyphSegmenter
from .classifier import HieroglyphClassifier
from .pipeline import process_hieroglyphic_image
from .visualization import display_results


def run_detection(image_path=None, sam_checkpoint=None, show_debug=True, max_entropy_ratio=0.75):
    image_path = image_path or TARGET_IMAGE
    sam_checkpoint = sam_checkpoint or SAM_CHECKPOINT

    print(f"Using device: {str(DEVICE).upper()}")

    model = load_classifier()

    print("System ready.\n")

    segmenter = HieroglyphSegmenter(
        sam_checkpoint_path=sam_checkpoint,
        config={
            'min_area': 250,
            'max_area': 15000,
            'min_width': 10,
            'min_height': 10,
            'max_width': 180,
            'max_height': 180,
            'max_aspect_ratio': 6.0,
            'min_aspect_ratio': 0.10,
            'min_solidity': 0.10,
            'min_extent': 0.10,
            'crop_padding': 8,
            'line_threshold': 55,
        }
    )

    classifier = HieroglyphClassifier(
        model=model,
        idx_to_class=idx_to_class,
        transform=inference_transforms,
        min_confidence=0.25,
    )

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    pipeline_result = process_hieroglyphic_image(
        image_path=image_path,
        segmenter=segmenter,
        classifier=classifier,
        show_debug=show_debug,
        max_entropy_ratio=max_entropy_ratio,
    )

    original_img = pipeline_result.get('original_img')
    if original_img is None:
        original_img = cv2.imread(image_path)
    display_results(pipeline_result, original_img)

    stats = pipeline_result['stats']
    print("\nPipeline Performance Metrics:")
    seg_eff = stats['valid_contours'] / max(stats['total_contours'], 1) * 100
    cls_eff = stats['accepted'] / max(stats['final_crops'], 1) * 100
    tot_eff = stats['accepted'] / max(stats['total_contours'], 1) * 100
    print(f"  SAM Prompts Triggered: {stats['total_contours']}")
    print(f"  Valid Bounding Boxes: {stats['valid_contours']} ({seg_eff:.1f}%)")
    print(f"  Classification Head: {stats['accepted']}/{stats['final_crops']} ({cls_eff:.1f}%)")
    print(f"  Overall Pipeline Yield: {stats['accepted']}/{stats['total_contours']} ({tot_eff:.1f}%)")


if __name__ == "__main__":
    try:
        run_detection()
    except Exception as e:
        print(f"\nPipeline error: {e}")
        traceback.print_exc()
