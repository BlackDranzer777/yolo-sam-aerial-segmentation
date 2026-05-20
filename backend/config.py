"""
config.py — Central configuration for the aerial image segmentation system.

All file paths, model settings, and pipeline hyperparameters are defined here.
Import this module anywhere in the backend instead of hardcoding paths.
"""

import os

# ─── Root paths ──────────────────────────────────────────────────────────────

# Absolute path to the backend/ directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories for uploaded images and processed outputs
UPLOAD_DIR  = os.path.join(BACKEND_DIR, "uploads")
OUTPUT_DIR  = os.path.join(BACKEND_DIR, "outputs")
MODELS_DIR  = os.path.join(BACKEND_DIR, "models")

# ─── YOLO configuration ───────────────────────────────────────────────────────

# YOLOv8n (nano) is fast and sufficient for detecting coarse bounding boxes
# that are then refined by SAM.  Swap for "yolov8s.pt" if you need higher
# detection recall at the cost of speed.
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8s-visdrone.pt")

# COCO-trained YOLOv8n used exclusively for person detection.
# VisDrone YOLO misses close-range pedestrians; COCO covers all scales.
COCO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

# Minimum detection confidence to pass a bounding box to SAM.
# Lower → more boxes (higher recall, more false positives).
# Higher → fewer boxes (higher precision, possible missed objects).
YOLO_CONFIDENCE_THRESHOLD = 0.50

# IoU threshold used by YOLO's non-maximum suppression step.
YOLO_IOU_THRESHOLD = 0.45

# ─── SAM configuration ────────────────────────────────────────────────────────

# SAM ViT-B checkpoint — smallest SAM variant, good balance of speed/quality.
# ViT-L or ViT-H give better masks but require more VRAM / RAM.
SAM_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sam_vit_b_01ec64.pth")

# SAM model type must match the checkpoint above.
# Options: "vit_b" | "vit_l" | "vit_h"
SAM_MODEL_TYPE = "vit_b"

# ─── Pipeline / prompt engineering ───────────────────────────────────────────

# Fractional padding added to each YOLO bounding box before it is passed to
# SAM as a box prompt.  E.g. 0.05 expands each box by 5 % of its width/height
# on all four sides.  A small expansion helps SAM capture object boundaries
# that sit just outside the tight detector box.
BBOX_PADDING_FRACTION = 0.05

# ─── Visualization ────────────────────────────────────────────────────────────

# Alpha blending weight for mask overlays drawn on the original image.
# 0.0 = fully transparent, 1.0 = fully opaque.
MASK_ALPHA = 0.45

# ─── Device ───────────────────────────────────────────────────────────────────

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Allowed upload extensions ────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ─── ICG Semantic Drone Dataset ───────────────────────────────────────────────

DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")

# Original aerial images — kept at download location to avoid duplicating 3.9 GB
ORIGINAL_IMAGES_DIR = r"C:\Users\divya\OneDrive\Divyansh\SAM_Thesis\Aerial semantic segmentation dron dataset\dataset\semantic_drone_dataset\original_images"

# Grayscale semantic label masks — pixel value = class index (0–23)
LABEL_MASKS_DIR = os.path.join(DATASET_DIR, "label_images_semantic")

# RGB coloured masks — for visualization only
RGB_MASKS_DIR = os.path.join(DATASET_DIR, "RGB_color_image_masks")

# Class definition CSV
CLASS_DICT_PATH = os.path.join(DATASET_DIR, "class_dict_seg.csv")

# Mapping from VisDrone YOLO class names to ICG dataset class indices
# (index order matches class_dict_seg.csv, starting at 0)
VISDRONE_TO_ICG_CLASS = {
    "person":          15,   # person
    "pedestrian":      15,   # person
    "people":          15,   # person (VisDrone group class)
    "car":             17,   # car
    "van":             17,   # car (closest match)
    "truck":           17,   # car (closest match)
    "bicycle":         18,   # bicycle
    "motor":           18,   # bicycle (closest match)
    "tricycle":        18,   # bicycle (closest match)
    "awning-tricycle": 18,   # bicycle (closest match)
    "bus":             17,   # car (closest match)
}
