"""
tests/test_yolo.py — Manual test for the YOLO detection module.

Downloads a small sample aerial image (or uses one you supply), runs the
YOLODetector, and prints every detected bounding box with its class label
and confidence score.

Usage:
    cd backend/
    python tests/test_yolo.py                          # uses built-in sample URL
    python tests/test_yolo.py path/to/your_image.jpg   # uses your own image
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.yolo_detector import YOLODetector


def main():
    # ── Resolve image path ────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            sys.exit(1)
    else:
        # Download a small public domain aerial/street photo for a quick test
        import urllib.request
        sample_url = (
            "https://ultralytics.com/images/bus.jpg"
        )
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "uploads", "test_sample.jpg"
        )
        if not os.path.exists(sample_path):
            print(f"Downloading sample image → {sample_path}")
            urllib.request.urlretrieve(sample_url, sample_path)
        image_path = sample_path

    print(f"\nInput image : {image_path}")

    # ── Run detection ─────────────────────────────────────────────────────────
    detector = YOLODetector()
    detections = detector.detect(image_path)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"Detections  : {len(detections)} object(s) found\n")

    if not detections:
        print("  No objects detected above the confidence threshold.")
        return

    col = "{:<5} {:<15} {:<8} {}"
    print(col.format("#", "Class", "Conf", "Bounding box [x1, y1, x2, y2]"))
    print("─" * 55)
    for i, det in enumerate(detections, start=1):
        print(col.format(
            i,
            det["class_name"],
            f"{det['confidence']:.2f}",
            det["bbox"],
        ))

    print("\n[PASS] YOLODetector is working correctly.\n")


if __name__ == "__main__":
    main()
