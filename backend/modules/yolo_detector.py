"""
modules/yolo_detector.py — YOLO-based object detection wrapper.

Wraps the Ultralytics YOLOv8 model to produce bounding box prompts that are
later consumed by SAM.  Each detected object is returned as a plain dict so
the rest of the pipeline has no dependency on Ultralytics internals.

Detection output format (one dict per object):
    {
        "bbox":        [x1, y1, x2, y2],   # absolute pixel coordinates
        "class_id":    int,                 # COCO class index
        "class_name":  str,                 # human-readable label
        "confidence":  float                # detector confidence score [0, 1]
    }
"""

from ultralytics import YOLO
import config


class YOLODetector:
    """Thin wrapper around YOLOv8 for bounding-box detection."""

    def __init__(self):
        """Load the YOLOv8 model from the path defined in config."""
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.confidence_threshold = config.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = config.YOLO_IOU_THRESHOLD

    def detect(self, image_path: str) -> list[dict]:
        """
        Run object detection on a single image.

        Args:
            image_path: Absolute or relative path to the input image.

        Returns:
            List of detection dicts, each containing bbox, class_id,
            class_name, and confidence.  Empty list if nothing detected.
        """
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=config.DEVICE,
            verbose=False,
        )

        detections = []
        # results is a list with one entry per image; we always pass one image
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                "class_id":   int(box.cls[0]),
                "class_name": self.model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
            })

        return detections
