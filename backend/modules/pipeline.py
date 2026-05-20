"""
modules/pipeline.py — YOLO → SAM segmentation pipeline.

Chains YOLODetector and SAMSegmentor into a single pipeline object.
Uses dual YOLO detection: VisDrone model for vehicles, COCO model for
people (better person recall on close-range images).

Output schema returned by SegmentationPipeline.run():
    {
        "image_id":       str,           # unique ID derived from the filename
        "detections":     list[dict],    # merged YOLO detections (both models)
        "seg_results":    list[dict],    # SAM masks + metadata (mask as bool array)
        "output_paths": {
            "boxes":    str,             # path to boxes-only visualization
            "masks":    str,             # path to masks-only visualization
            "combined": str,             # path to combined visualization
        }
    }
"""

import os
import uuid
import cv2
from ultralytics import YOLO

from modules.yolo_detector import YOLODetector
from modules.sam_segmentor import SAMSegmentor
from utils.visualization import draw_boxes, draw_masks, draw_combined, save_visualization
import config


# COCO class 0 is "person" — the only class we pull from the COCO model
_COCO_PERSON_CLASS_ID = 0


def _box_iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _cross_class_nms(detections: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    """
    Suppress duplicate detections across different class names.

    YOLO's built-in NMS runs per-class, so the same physical object can survive
    as both "car" and "van" with overlapping boxes.  This greedy NMS keeps only
    the highest-confidence box when two boxes of ANY class overlap above the
    threshold.
    """
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in sorted_dets:
        if not any(_box_iou(det["bbox"], k["bbox"]) > iou_threshold for k in kept):
            kept.append(det)
    return kept


def _merge_detections(visdrone_dets: list[dict], coco_person_dets: list[dict],
                      iou_threshold: float = 0.4) -> list[dict]:
    """
    Merge VisDrone and COCO person detections.

    First applies cross-class NMS within VisDrone detections (removes car/van
    duplicates on the same vehicle).  Then adds each COCO person detection only
    if it does not overlap with any existing detection.
    """
    merged = _cross_class_nms(visdrone_dets, iou_threshold)
    for coco_det in coco_person_dets:
        overlaps_existing = any(
            _box_iou(coco_det["bbox"], d["bbox"]) > iou_threshold
            for d in merged
        )
        if not overlaps_existing:
            merged.append(coco_det)
    return merged


class SegmentationPipeline:
    """Chains dual-YOLO detection → SAM segmentation and saves visualizations."""

    def __init__(self):
        self.detector  = YOLODetector()
        self.segmentor = SAMSegmentor()
        self.coco_model = YOLO(config.COCO_MODEL_PATH)

    def run(self, image_path: str, image_id: str = None, confidence: float = None,
            models: list[str] = None) -> dict:
        """
        Run the full dual-YOLO → SAM pipeline on a single image.

        Args:
            image_path:  Path to the input image file.
            image_id:    Optional identifier used as the output filename prefix.
                         If omitted, a short UUID is generated.
            confidence:  Optional YOLO confidence threshold override.
            models:      List of model keys to use: "visdrone", "coco", or both.
                         Defaults to both when None.

        Returns:
            Result dict (see module docstring for schema).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        if image_id is None:
            image_id = uuid.uuid4().hex[:8]

        if models is None:
            models = ["visdrone", "coco"]

        use_visdrone = "visdrone" in models
        use_coco     = "coco"     in models

        conf_value = confidence if confidence is not None else config.YOLO_CONFIDENCE_THRESHOLD

        # ── Stage 1a: VisDrone detection (vehicles + aerial pedestrians) ─────────
        visdrone_dets = self.detector.detect(image_path, conf=confidence) if use_visdrone else []

        # ── Stage 1b: COCO person detection (close-range pedestrians) ───────────
        coco_person_dets = []
        if use_coco:
            coco_results = self.coco_model.predict(
                source=image_path,
                conf=conf_value,
                iou=config.YOLO_IOU_THRESHOLD,
                classes=[_COCO_PERSON_CLASS_ID],
                device=config.DEVICE,
                verbose=False,
            )
            for box in coco_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                coco_person_dets.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "class_id":   _COCO_PERSON_CLASS_ID,
                    "class_name": "person",
                    "confidence": round(float(box.conf[0]), 4),
                })

        # ── Stage 1c: Merge — VisDrone + COCO people (no duplicates) ────────────
        detections = _merge_detections(visdrone_dets, coco_person_dets)

        # ── Stage 2: Segmentation (skipped if no detections) ─────────────────────
        seg_results = []
        if detections:
            seg_results = self.segmentor.segment(image_path, detections)

            for det, seg in zip(detections, seg_results):
                seg["class_id"] = det.get("class_id", 0)

        # ── Stage 3: Visualizations ───────────────────────────────────────────────
        image_bgr = cv2.imread(image_path)

        boxes_img    = draw_boxes(image_bgr, detections)
        masks_img    = draw_masks(image_bgr, seg_results)
        combined_img = draw_combined(image_bgr, detections, seg_results)

        output_paths = {
            "boxes":    save_visualization(boxes_img,    image_id, "boxes"),
            "masks":    save_visualization(masks_img,    image_id, "masks"),
            "combined": save_visualization(combined_img, image_id, "combined"),
        }

        return {
            "image_id":     image_id,
            "detections":   detections,
            "seg_results":  seg_results,
            "output_paths": output_paths,
        }
