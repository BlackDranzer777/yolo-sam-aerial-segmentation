"""
modules/sam_segmentor.py — SAM segmentation wrapper.

Wraps Meta's Segment Anything Model (SAM) to accept bounding-box prompts
produced by YOLODetector and return binary segmentation masks.

SAM prompt strategy used here:
    Box prompt — each YOLO bounding box is expanded slightly (see
    BBOX_PADDING_FRACTION in config) and passed to SamPredictor as a
    box prompt.  Box prompts are the most reliable prompt type for
    zero-shot segmentation when object location is known (Kirillov et al.,
    2023, Section 3.2).

Segmentation output format (one dict per detection):
    {
        "mask":        np.ndarray (H, W, bool),  # binary mask at input resolution
        "bbox":        [x1, y1, x2, y2],         # original YOLO bbox
        "class_name":  str,
        "confidence":  float                      # YOLO detection confidence
    }
"""

import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

import config


class SAMSegmentor:
    """Wraps SAM's SamPredictor to segment objects given bounding-box prompts."""

    def __init__(self):
        """Load SAM model and initialise the predictor."""
        sam = sam_model_registry[config.SAM_MODEL_TYPE](
            checkpoint=config.SAM_CHECKPOINT_PATH
        )
        sam.to(device=config.DEVICE)
        self.predictor = SamPredictor(sam)

    # ── Public API ────────────────────────────────────────────────────────────

    def segment(self, image_path: str, detections: list[dict]) -> list[dict]:
        """
        Generate a segmentation mask for each detected object.

        SAM's image encoder runs once per image (set_image), then the prompt
        encoder + mask decoder run once per bounding-box prompt.  This is the
        efficient usage pattern recommended by Kirillov et al.

        Args:
            image_path:  Path to the input image (must match the image used
                         for detection).
            detections:  List of detection dicts from YOLODetector.detect().

        Returns:
            List of segmentation result dicts (one per detection).
            If detections is empty, returns an empty list.
        """
        if not detections:
            return []

        # Load image as RGB numpy array — SAM expects RGB, not BGR
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Encode the image once; the embedding is reused for all prompts
        self.predictor.set_image(image_rgb)

        h, w = image_rgb.shape[:2]
        results = []

        for det in detections:
            padded_box = self._pad_bbox(det["bbox"], w, h)
            box_array = np.array(padded_box, dtype=float)

            # SAM returns three candidate masks ranked by predicted IoU.
            # We take the highest-scoring one (index 0).
            masks, scores, _ = self.predictor.predict(
                box=box_array,
                multimask_output=True,  # get 3 candidates → pick best
            )
            best_idx = int(np.argmax(scores))

            results.append({
                "mask":       masks[best_idx],   # bool array (H, W)
                "bbox":       det["bbox"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
            })

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pad_bbox(self, bbox: list[int], img_w: int, img_h: int) -> list[int]:
        """
        Expand a bounding box by BBOX_PADDING_FRACTION on all four sides,
        clamped to image boundaries.

        A small expansion helps SAM capture object boundaries that sit just
        outside the tight detector box (common in aerial imagery where
        object extents are hard to predict precisely).
        """
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * config.BBOX_PADDING_FRACTION)
        pad_y = int((y2 - y1) * config.BBOX_PADDING_FRACTION)

        x1 = max(0,     x1 - pad_x)
        y1 = max(0,     y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)

        return [x1, y1, x2, y2]
