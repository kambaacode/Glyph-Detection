import cv2
import matplotlib.pyplot as plt


def display_results(pipeline_result, original_img):
    results = pipeline_result['results']
    rejected = pipeline_result['rejected']
    sequence = pipeline_result['sequence']
    annotated_img = pipeline_result['annotated_img']
    binary_img = pipeline_result['binary_img']
    processed_img = pipeline_result.get('processed_img')
    rejected_img = pipeline_result['rejected_img']

    n_plots = 4 if processed_img is not None else 3
    if rejected_img is None:
        n_plots -= 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 8))
    idx = 0
    axes[idx].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[idx].set_title("Original Image (Before)", fontsize=13, fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    if processed_img is not None:
        axes[idx].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title("Enhanced Image (After)", fontsize=13, fontweight='bold', color='blue')
        axes[idx].axis('off')
        idx += 1
    axes[idx].imshow(binary_img, cmap='gray')
    axes[idx].set_title("SAM Unified Mask", fontsize=13, fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    axes[idx].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f"Detected: {len(sequence)} symbols", fontsize=13, fontweight='bold', color='green')
    axes[idx].axis('off')
    plt.tight_layout()
    plt.show()

    if rejected_img is not None and rejected:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(rejected_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Rejected: {len(rejected)} boxes", fontsize=14, fontweight='bold', color='red')
        plt.axis('off')
        plt.show()

    print("\n" + "=" * 75)
    print("Detected Symbols Details:")
    print("=" * 75)
    print(f"{'#':<4}{'Gardiner':<22}{'Confidence':<12}{'2nd Best':<25}{'Location (x,y,w,h)'}")
    print("-" * 75)
    for i, r in enumerate(results):
        bbox = r['bbox']
        top2 = f"{r['top3'][1][0]} ({r['top3'][1][1]:.0%})" if len(r['top3']) > 1 else "-"
        print(f"{i + 1:<4}{r['label']:<22}{r['confidence']:.1%}    {top2:<25}"
              f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")
    print("=" * 75)

    if results:
        n_symbols = len(results)
        cols = min(6, n_symbols)
        rows = (n_symbols + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 3.5 * rows))
        axes = axes.flatten() if n_symbols > 1 else [axes]
        for i, ax in enumerate(axes):
            if i < n_symbols:
                r = results[i]
                ax.imshow(cv2.cvtColor(r['crop'], cv2.COLOR_BGR2RGB))
                ax.set_title(f"{r['label']}\n{r['confidence']:.0%}", fontsize=9, fontweight='bold')
                ax.axis('off')
                color = 'green' if r['confidence'] >= 0.8 else 'orange' if r['confidence'] >= 0.6 else 'red'
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(2)
            else:
                ax.axis('off')
        plt.suptitle("Individual Cropped Symbols", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 50)
    print("Final Detected Sequence:")
    print("=" * 50)
    if sequence:
        print(f"  {' -> '.join(sequence)}")
        print(f"  Total: {len(sequence)} symbols")
    else:
        print("  No valid symbols detected")
    print("=" * 50 + "\n")
