"""
tests/test_pipeline.py — Manual test for the full YOLO → SAM pipeline.

Runs SegmentationPipeline.run() on an image and saves three output images:
    <image_id>_boxes.jpg    — YOLO bounding boxes only
    <image_id>_masks.jpg    — SAM segmentation masks only
    <image_id>_combined.jpg — Boxes + masks together

Usage:
    cd backend/
    python tests/test_pipeline.py                          # uses test_sample.jpg
    python tests/test_pipeline.py path/to/your_image.jpg   # uses your own image
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pipeline import SegmentationPipeline
import config


def main():
    # ── Resolve image path ────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            sys.exit(1)
    else:
        image_path = os.path.join(config.UPLOAD_DIR, "test_sample.jpg")
        if not os.path.exists(image_path):
            print("[ERROR] No test image found.")
            print("        Pass an image path as argument: python tests/test_pipeline.py path/to/image.jpg")
            sys.exit(1)

    print(f"\nInput image  : {image_path}")
    print("Running pipeline (YOLO → SAM) ...\n")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline = SegmentationPipeline()
    result   = pipeline.run(image_path, image_id="test")

    # ── Print detection summary ───────────────────────────────────────────────
    detections = result["detections"]
    seg_results = result["seg_results"]

    print(f"Detections   : {len(detections)} object(s)")
    print(f"Masks        : {len(seg_results)} mask(s)\n")

    if detections:
        col = "{:<5} {:<18} {:<8} {:<28} {}"
        print(col.format("#", "Class", "Conf", "Bbox [x1,y1,x2,y2]", "Mask coverage"))
        print("─" * 75)
        for i, (det, seg) in enumerate(zip(detections, seg_results), start=1):
            mask_pct = 100 * seg["mask"].sum() / seg["mask"].size
            print(col.format(
                i,
                det["class_name"],
                f"{det['confidence']:.2f}",
                str(det["bbox"]),
                f"{mask_pct:.1f}% of image",
            ))
    else:
        print("  No objects detected above confidence threshold.")

    # ── Print output file paths ───────────────────────────────────────────────
    print("\nOutput images saved:")
    for key, path in result["output_paths"].items():
        print(f"  {key:<10} → {path}")

    print("\n[PASS] Pipeline is working correctly.\n")


if __name__ == "__main__":
    main()
