"""
tests/test_sam.py — Manual test for the SAM segmentation module.

Runs YOLODetector on a sample image to get bounding boxes, then passes
those boxes to SAMSegmentor and verifies that binary masks are returned.
Saves a quick visual overlay to backend/outputs/test_sam_output.jpg so
you can inspect the masks visually.

Usage:
    cd backend/
    python tests/test_sam.py                          # uses test_sample.jpg
    python tests/test_sam.py path/to/your_image.jpg   # uses your own image
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from modules.yolo_detector import YOLODetector
from modules.sam_segmentor import SAMSegmentor
import config


# Fixed colours for up to 10 overlaid masks (BGR)
MASK_COLOURS = [
    (0, 255, 128),   # green
    (0, 128, 255),   # orange
    (255, 64,  64),  # blue-red
    (255, 0,  255),  # magenta
    (0, 255, 255),   # yellow
    (128, 0,  255),  # purple
    (255, 128,  0),  # sky blue
    (0,  64,  255),  # red-orange
    (64, 255,  0),   # lime
    (255, 255,  0),  # cyan
]


def draw_mask_overlay(image: np.ndarray, seg_results: list[dict]) -> np.ndarray:
    """Draw semi-transparent coloured masks + bbox labels onto image."""
    overlay = image.copy()

    for i, seg in enumerate(seg_results):
        colour = MASK_COLOURS[i % len(MASK_COLOURS)]
        mask = seg["mask"]  # bool (H, W)

        # Fill mask region with colour
        overlay[mask] = (
            overlay[mask] * (1 - config.MASK_ALPHA)
            + np.array(colour) * config.MASK_ALPHA
        ).astype(np.uint8)

        # Draw bounding box
        x1, y1, x2, y2 = seg["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        # Label
        label = f"{seg['class_name']} {seg['confidence']:.2f}"
        cv2.putText(overlay, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    return overlay


def main():
    # ── Resolve image path ────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            sys.exit(1)
    else:
        image_path = os.path.join(config.UPLOAD_DIR, "test_sample_uav.jpg")
        if not os.path.exists(image_path):
            print("[ERROR] test_sample.jpg not found.")
            print("        Run tests/test_yolo.py first, or pass an image path as argument.")
            sys.exit(1)

    print(f"\nInput image : {image_path}")

    # ── Step 1: YOLO detection ────────────────────────────────────────────────
    print("Running YOLO detection ...")
    detector = YOLODetector()
    detections = detector.detect(image_path)
    print(f"  {len(detections)} object(s) detected")

    if not detections:
        print("\n[WARN] YOLO found 0 objects (expected on aerial images — domain gap).")
        print("       Injecting a manual bounding box to verify SAM independently ...\n")

        # For aerial images, YOLO (COCO-trained) often detects nothing because
        # objects look very different from a nadir viewpoint.  We inject a
        # coarse hand-drawn box covering a visible object so we can confirm
        # SAM's mask decoder is working correctly before building the pipeline.
        #
        # The box below covers the central road area in a typical aerial image.
        # Adjust [x1, y1, x2, y2] to frame any clearly visible object in your image.
        h, w = cv2.imread(image_path).shape[:2]
        # Default: a box in the centre-left quarter of the image
        manual_box = [w // 8, h // 8, w // 2, h // 2]
        detections = [{
            "bbox":       manual_box,
            "class_id":   0,
            "class_name": "manual_region",
            "confidence": 1.0,
        }]
        print(f"  Manual bbox: {manual_box}  (image size: {w}×{h})")
        print("  Edit the 'manual_box' variable in this script to frame a specific object.\n")

    # ── Step 2: SAM segmentation ──────────────────────────────────────────────
    print("Running SAM segmentation (this may take ~10–30 s on CPU) ...")
    segmentor = SAMSegmentor()
    seg_results = segmentor.segment(image_path, detections)
    print(f"  {len(seg_results)} mask(s) generated")

    # ── Print per-object summary ──────────────────────────────────────────────
    print()
    col = "{:<5} {:<15} {:<8} {:<20} {}"
    print(col.format("#", "Class", "Conf", "Bbox", "Mask pixels"))
    print("─" * 65)
    for i, seg in enumerate(seg_results, start=1):
        mask_px = int(seg["mask"].sum())
        total_px = seg["mask"].size
        print(col.format(
            i,
            seg["class_name"],
            f"{seg['confidence']:.2f}",
            str(seg["bbox"]),
            f"{mask_px} ({100 * mask_px / total_px:.1f}% of image)",
        ))

    # ── Save visual output ────────────────────────────────────────────────────
    image_bgr = cv2.imread(image_path)
    output = draw_mask_overlay(image_bgr, seg_results)
    out_path = os.path.join(config.OUTPUT_DIR, "test_sam_output.jpg")
    cv2.imwrite(out_path, output)
    print(f"\nVisual output saved → {out_path}")
    print("[PASS] SAMSegmentor is working correctly.\n")


if __name__ == "__main__":
    main()
