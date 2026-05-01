"""
modules/evaluator.py — Segmentation evaluation metrics.

Computes pixel-level IoU, Precision, and Recall by comparing SAM's predicted
binary masks against ground truth masks from the ICG Semantic Drone Dataset.

Ground truth format (ICG dataset):
    label_images_semantic/<id>.png — grayscale PNG where each pixel value
    is a class index (0–23) matching the order in class_dict_seg.csv.

Evaluation approach:
    For each SAM-predicted mask:
      1. Map the YOLO class name to an ICG class index (via config).
      2. Extract the binary ground truth mask for that class from the GT image.
      3. Compute IoU, Precision, Recall between predicted and GT binary masks.

    Aggregate per-class and overall metrics across all predictions.
"""

import os
import numpy as np
import cv2

import config


# ── Core metric functions ─────────────────────────────────────────────────────

def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.

    IoU = |pred ∩ gt| / |pred ∪ gt|

    Args:
        pred_mask: Boolean numpy array (H, W) — SAM predicted mask.
        gt_mask:   Boolean numpy array (H, W) — ground truth binary mask.

    Returns:
        IoU score in [0, 1]. Returns 0.0 if union is empty.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def calculate_precision_recall(pred_mask: np.ndarray,
                                gt_mask: np.ndarray) -> tuple[float, float]:
    """
    Compute pixel-level Precision and Recall.

    Precision = TP / (TP + FP)  — of all pixels we predicted as object,
                                   how many actually are?
    Recall    = TP / (TP + FN)  — of all actual object pixels,
                                   how many did we find?

    Args:
        pred_mask: Boolean numpy array (H, W).
        gt_mask:   Boolean numpy array (H, W).

    Returns:
        (precision, recall) tuple, each in [0, 1].
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall    = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth_mask(image_id: str, class_index: int) -> np.ndarray | None:
    """
    Load a binary ground truth mask for one class from the ICG dataset.

    Args:
        image_id:    Numeric image identifier, e.g. "038" or "38".
        class_index: ICG class index (0–23) from class_dict_seg.csv.

    Returns:
        Boolean numpy array (H, W) where True = pixels belonging to that class.
        None if the ground truth file is not found.
    """
    # Normalise to 3-digit zero-padded filename e.g. "38" → "038"
    padded_id = str(image_id).zfill(3)
    gt_path   = os.path.join(config.LABEL_MASKS_DIR, f"{padded_id}.png")

    if not os.path.exists(gt_path):
        return None

    # Load as grayscale — pixel values are class indices
    gt_label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_label is None:
        return None

    # Squeeze any extra dimensions (some PNGs load as H×W×1)
    gt_label = np.squeeze(gt_label)

    return gt_label == class_index


# ── Batch evaluator ───────────────────────────────────────────────────────────

def evaluate_results(seg_results: list[dict], image_id: str) -> dict:
    """
    Evaluate all SAM masks for one image against ICG ground truth.

    Args:
        seg_results: List of segmentation dicts from SAMSegmentor.segment().
                     Each dict must have: mask (bool array), class_name, confidence.
        image_id:    Numeric image identifier matching the ICG dataset filename.

    Returns:
        {
            "per_object": [ { class_name, iou, precision, recall, ... }, ... ],
            "per_class":  { class_name: { iou, precision, recall, count }, ... },
            "overall":    { mean_iou, mean_precision, mean_recall, evaluated_count }
        }
    """
    per_object = []

    for seg in seg_results:
        class_name  = seg["class_name"]
        pred_mask   = seg["mask"].astype(bool)
        icg_index   = config.VISDRONE_TO_ICG_CLASS.get(class_name)

        if icg_index is None:
            # Class not mappable to ICG — skip
            per_object.append({
                "class_name": class_name,
                "iou":        None,
                "precision":  None,
                "recall":     None,
                "note":       "no ICG class mapping",
            })
            continue

        gt_mask = load_ground_truth_mask(image_id, icg_index)

        if gt_mask is None:
            per_object.append({
                "class_name": class_name,
                "iou":        None,
                "precision":  None,
                "recall":     None,
                "note":       f"ground truth file not found for image {image_id}",
            })
            continue

        # Resize pred mask to GT size if they differ (can happen with large images)
        if pred_mask.shape != gt_mask.shape:
            pred_resized = cv2.resize(
                pred_mask.astype(np.uint8),
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            pred_resized = pred_mask

        iou               = calculate_iou(pred_resized, gt_mask)
        precision, recall = calculate_precision_recall(pred_resized, gt_mask)

        per_object.append({
            "class_name": class_name,
            "iou":        round(iou, 4),
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "confidence": seg.get("confidence"),
            "note":       "ok",
        })

    # ── Per-class aggregation ─────────────────────────────────────────────────
    per_class = {}
    for obj in per_object:
        if obj["iou"] is None:
            continue
        cls = obj["class_name"]
        if cls not in per_class:
            per_class[cls] = {"iou": [], "precision": [], "recall": [], "count": 0}
        per_class[cls]["iou"].append(obj["iou"])
        per_class[cls]["precision"].append(obj["precision"])
        per_class[cls]["recall"].append(obj["recall"])
        per_class[cls]["count"] += 1

    per_class_summary = {
        cls: {
            "mean_iou":       round(np.mean(v["iou"]), 4),
            "mean_precision": round(np.mean(v["precision"]), 4),
            "mean_recall":    round(np.mean(v["recall"]), 4),
            "count":          v["count"],
        }
        for cls, v in per_class.items()
    }

    # ── Overall metrics ───────────────────────────────────────────────────────
    valid = [o for o in per_object if o["iou"] is not None]
    if valid:
        overall = {
            "mean_iou":        round(float(np.mean([o["iou"]       for o in valid])), 4),
            "mean_precision":  round(float(np.mean([o["precision"]  for o in valid])), 4),
            "mean_recall":     round(float(np.mean([o["recall"]     for o in valid])), 4),
            "evaluated_count": len(valid),
        }
    else:
        overall = {
            "mean_iou": 0.0, "mean_precision": 0.0,
            "mean_recall": 0.0, "evaluated_count": 0,
        }

    return {
        "per_object":  per_object,
        "per_class":   per_class_summary,
        "overall":     overall,
    }