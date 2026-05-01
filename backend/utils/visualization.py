"""
utils/visualization.py — Visualization utilities for detection and segmentation results.

Produces three output images that directly support the thesis comparison:
  1. boxes-only   — YOLO detection output (Research Question 2 baseline)
  2. masks-only   — SAM segmentation masks without box outlines
  3. combined     — Bounding boxes + segmentation masks overlaid together

All functions take a BGR numpy array (as returned by cv2.imread) and return
a new BGR numpy array with the visualization drawn on top.  The original
image is never modified in-place.
"""

import os
import cv2
import numpy as np

import config

# One distinct colour per class slot (BGR).  Cycles if more than 20 classes.
_PALETTE = [
    (  0, 255, 128), (  0, 128, 255), (255,  64,  64), (255,   0, 255),
    (  0, 255, 255), (128,   0, 255), (255, 128,   0), (  0,  64, 255),
    ( 64, 255,   0), (255, 255,   0), (  0, 192, 255), (255,   0, 128),
    (128, 255,   0), (  0, 255,  64), (255, 192,   0), (192,   0, 255),
    (  0, 128, 128), (128,   0,   0), (  0,   0, 128), (128, 128,   0),
]


def _colour_for(index: int) -> tuple:
    return _PALETTE[index % len(_PALETTE)]


# ── Public drawing functions ──────────────────────────────────────────────────

def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw YOLO bounding boxes and class labels onto the image.

    Args:
        image:      BGR numpy array (H, W, 3).
        detections: List of detection dicts from YOLODetector.detect().

    Returns:
        New BGR image with boxes drawn.
    """
    output = image.copy()

    for i, det in enumerate(detections):
        colour = _colour_for(det.get("class_id", i))
        x1, y1, x2, y2 = det["bbox"]

        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)

        label = f"{det['class_name']}  {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        # Background pill for readability
        cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(output, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return output


def draw_masks(image: np.ndarray, seg_results: list[dict]) -> np.ndarray:
    """
    Draw semi-transparent SAM segmentation masks onto the image.

    Args:
        image:       BGR numpy array (H, W, 3).
        seg_results: List of segmentation dicts from SAMSegmentor.segment().

    Returns:
        New BGR image with coloured mask overlays.
    """
    output = image.copy()

    for i, seg in enumerate(seg_results):
        colour = _colour_for(seg.get("class_id", i))
        mask = seg["mask"]  # bool (H, W)

        # Alpha-blend the mask colour into the masked region
        output[mask] = (
            output[mask] * (1 - config.MASK_ALPHA)
            + np.array(colour, dtype=np.float32) * config.MASK_ALPHA
        ).astype(np.uint8)

        # Thin contour around the mask boundary for clarity
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, colour, 2)

    return output


def draw_combined(image: np.ndarray, detections: list[dict],
                  seg_results: list[dict]) -> np.ndarray:
    """
    Draw both segmentation masks and bounding boxes on the same image.

    Masks are drawn first (underneath), boxes and labels on top, so the
    label text remains readable.

    Args:
        image:       BGR numpy array (H, W, 3).
        detections:  List of detection dicts from YOLODetector.detect().
        seg_results: List of segmentation dicts from SAMSegmentor.segment().

    Returns:
        New BGR image with masks + boxes combined.
    """
    output = draw_masks(image, seg_results)
    output = draw_boxes(output, detections)
    return output


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_visualization(image: np.ndarray, image_id: str, suffix: str) -> str:
    """
    Save a visualization image to the outputs directory.

    Args:
        image:    BGR numpy array to save.
        image_id: Unique identifier for the source image (used as filename prefix).
        suffix:   One of "boxes", "masks", or "combined".

    Returns:
        Absolute path to the saved file.
    """
    filename = f"{image_id}_{suffix}.jpg"
    out_path = os.path.join(config.OUTPUT_DIR, filename)
    cv2.imwrite(out_path, image)
    return out_path
