"""
modules/pipeline.py — YOLO → SAM segmentation pipeline.

Chains YOLODetector and SAMSegmentor into a single pipeline object.
The pipeline is the main entry point used by the Flask API and the
evaluation module — nothing outside this file needs to know how YOLO
and SAM are wired together.

Output schema returned by SegmentationPipeline.run():
    {
        "image_id":       str,           # unique ID derived from the filename
        "detections":     list[dict],    # raw YOLO detections
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

from modules.yolo_detector import YOLODetector
from modules.sam_segmentor import SAMSegmentor
from utils.visualization import draw_boxes, draw_masks, draw_combined, save_visualization
import config


class SegmentationPipeline:
    """Chains YOLO detection → SAM segmentation and saves visualizations."""

    def __init__(self):
        self.detector  = YOLODetector()
        self.segmentor = SAMSegmentor()

    def run(self, image_path: str, image_id: str = None) -> dict:
        """
        Run the full YOLO → SAM pipeline on a single image.

        Args:
            image_path: Path to the input image file.
            image_id:   Optional identifier used as the output filename prefix.
                        If omitted, a short UUID is generated.

        Returns:
            Result dict (see module docstring for schema).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        if image_id is None:
            image_id = uuid.uuid4().hex[:8]

        # ── Stage 1: Object detection ─────────────────────────────────────────
        detections = self.detector.detect(image_path)

        # ── Stage 2: Segmentation (skipped if no detections) ──────────────────
        seg_results = []
        if detections:
            seg_results = self.segmentor.segment(image_path, detections)

            # Propagate class_id from detection into seg_result for colour mapping
            for det, seg in zip(detections, seg_results):
                seg["class_id"] = det.get("class_id", 0)

        # ── Stage 3: Visualizations ────────────────────────────────────────────
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
