import cv2


def process_hieroglyphic_image(image_path, segmenter, classifier,
                               show_debug=False, max_entropy_ratio=0.65):
    print(f"Processing image: {image_path}")
    seg_img, class_img, original_img, binary_img, crops, total_contours, valid_contours = \
        segmenter.extract_symbols(image_path)
    print(f"Prompts: {total_contours} | Valid Boxes: {valid_contours} | Crops: {len(crops)}")
    output_img = original_img.copy()
    rejected_img = original_img.copy() if show_debug else None
    results, rejected = [], []
    for crop, bbox in crops:
        x, y, w, h = bbox
        prediction = classifier.classify(crop)
        is_valid, reason = classifier.is_valid_prediction(prediction, max_entropy_ratio)
        if is_valid:
            results.append({**prediction, 'bbox': bbox, 'crop': crop})
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 220, 0), 2)
            label = f"{prediction['label']} ({prediction['confidence']:.0%})"
            cv2.putText(output_img, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)
        else:
            rejected.append({**prediction, 'bbox': bbox, 'reject_reason': reason})
            if show_debug:
                cv2.rectangle(rejected_img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    sequence = [r['label'] for r in results]
    print(f"Accepted: {len(results)} | Rejected: {len(rejected)}")
    reasons = {}
    for r in rejected:
        reasons[r['reject_reason']] = reasons.get(r['reject_reason'], 0) + 1
    for reason, count in reasons.items():
        print(f"  - {reason}: {count}")
    return {
        'sequence': sequence,
        'results': results,
        'rejected': rejected,
        'annotated_img': output_img,
        'rejected_img': rejected_img,
        'binary_img': binary_img,
        'processed_img': seg_img,
        'original_img': original_img,
        'stats': {
            'total_contours': total_contours,
            'valid_contours': valid_contours,
            'final_crops': len(crops),
            'accepted': len(results),
            'rejected': len(rejected),
        },
    }
