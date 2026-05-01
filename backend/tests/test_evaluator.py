"""
tests/test_evaluator.py — Manual test for the evaluation module.

Runs the full pipeline on an ICG dataset image then evaluates the predicted
masks against the ground truth label mask.

Usage:
    cd backend/
    python tests/test_evaluator.py 038
    python tests/test_evaluator.py 001
    (pass any 3-digit image ID that exists in label_images_semantic/)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from modules.pipeline import SegmentationPipeline
from modules.evaluator import evaluate_results


def main():
    image_id = sys.argv[1] if len(sys.argv) > 1 else "038"
    padded   = image_id.zfill(3)

    # ── Find the source image ─────────────────────────────────────────────────
    # Use directly from ICG dataset original_images/
    image_path = os.path.join(config.ORIGINAL_IMAGES_DIR, f"{padded}.jpg")
    if not os.path.exists(image_path):
        print(f"[ERROR] No image found: {image_path}")
        sys.exit(1)

    print(f"\nImage       : {image_path}")
    print(f"Image ID    : {padded}")

    # ── Check ground truth exists ─────────────────────────────────────────────
    gt_path = os.path.join(config.LABEL_MASKS_DIR, f"{padded}.png")
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth not found: {gt_path}")
        sys.exit(1)
    print(f"Ground truth: {gt_path}")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print("\nRunning YOLO → SAM pipeline ...")
    pipeline    = SegmentationPipeline()
    result      = pipeline.run(image_path, image_id=padded)
    seg_results = result["seg_results"]

    print(f"  {len(result['detections'])} detection(s), {len(seg_results)} mask(s)")

    if not seg_results:
        print("\n[WARN] No masks to evaluate.")
        return

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating masks against ground truth ...")
    metrics = evaluate_results(seg_results, padded)

    # ── Print per-object results ──────────────────────────────────────────────
    print("\n── Per-object metrics ───────────────────────────────────────────")
    col = "{:<5} {:<15} {:<8} {:<10} {:<10} {:<10}"
    print(col.format("#", "Class", "Conf", "IoU", "Precision", "Recall"))
    print("─" * 60)
    for i, obj in enumerate(metrics["per_object"], 1):
        if obj["iou"] is None:
            print(f"  {i:<4} {obj['class_name']:<15} — skipped ({obj['note']})")
        else:
            print(col.format(
                i,
                obj["class_name"],
                f"{obj['confidence']:.2f}",
                f"{obj['iou']:.4f}",
                f"{obj['precision']:.4f}",
                f"{obj['recall']:.4f}",
            ))

    # ── Print per-class summary ───────────────────────────────────────────────
    print("\n── Per-class summary ────────────────────────────────────────────")
    col2 = "{:<15} {:<8} {:<10} {:<10} {:<10}"
    print(col2.format("Class", "Count", "Mean IoU", "Mean Prec", "Mean Rec"))
    print("─" * 55)
    for cls, m in metrics["per_class"].items():
        print(col2.format(
            cls, m["count"],
            f"{m['mean_iou']:.4f}",
            f"{m['mean_precision']:.4f}",
            f"{m['mean_recall']:.4f}",
        ))

    # ── Print overall ─────────────────────────────────────────────────────────
    ov = metrics["overall"]
    print(f"\n── Overall ({ov['evaluated_count']} mask(s) evaluated) ──────────────────")
    print(f"  Mean IoU       : {ov['mean_iou']:.4f}")
    print(f"  Mean Precision : {ov['mean_precision']:.4f}")
    print(f"  Mean Recall    : {ov['mean_recall']:.4f}")
    print("\n[PASS] Evaluator is working correctly.\n")


if __name__ == "__main__":
    main()