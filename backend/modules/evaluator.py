"""
modules/evaluator.py — Segmentation evaluation metrics.

Computes pixel-level IoU, Precision, and Recall by comparing SAM's predicted
binary masks against ground truth masks from the ICG Semantic Drone Dataset.

Ground truth format (ICG dataset):
    label_images_semantic/<id>.png — grayscale PNG where each pixel value
    is a class index (0–23) matching the order in class_dict_seg.csv.

Evaluation approach (merged / semantic):
    The ICG dataset provides semantic masks — all instances of a class share
    one combined mask. We match this by merging all SAM predictions of the
    same class into one mask before comparing against GT. This gives one
    IoU per class, which is the standard semantic segmentation metric.

    For each class present in predictions:
      1. Map the YOLO class name to an ICG class index (via config).
      2. OR all SAM masks of that class into one merged predicted mask.
      3. Compare merged mask against the full GT class mask (entire image).
      4. Report IoU, Precision, Recall for that class.
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
    Evaluate SAM masks for one image using merged (semantic) IoU.

    All SAM masks of the same class are ORed into one combined predicted mask,
    then compared against the full GT class mask once. This matches the ICG
    dataset's semantic label format and gives one IoU per class.

    Args:
        seg_results: List of segmentation dicts from SAMSegmentor.segment().
                     Each dict must have: mask (bool array), class_name, confidence.
        image_id:    Numeric image identifier matching the ICG dataset filename.

    Returns:
        {
            "per_object": [ { class_name, iou, precision, recall, ... }, ... ],
            "per_class":  { class_name: { mean_iou, mean_precision, mean_recall, count } },
            "overall":    { mean_iou, mean_precision, mean_recall, evaluated_count }
        }
        per_object has one entry per class (the merged result).
    """
    # ── Group predictions by ICG class index (not VisDrone name) ────────────────
    # Multiple VisDrone classes can map to the same ICG class (e.g. car + van →
    # class 17).  We must merge all of them into one mask before comparing against
    # the GT, otherwise we evaluate the same GT mask multiple times.
    icg_groups: dict[int, list[dict]] = {}       # icg_index → list of seg dicts
    unmapped:   list[dict]            = []        # segs with no ICG mapping

    for seg in seg_results:
        icg_index = config.VISDRONE_TO_ICG_CLASS.get(seg["class_name"])
        if icg_index is None:
            unmapped.append(seg)
        else:
            icg_groups.setdefault(icg_index, []).append(seg)

    # ICG class index → human-readable label (first VisDrone name seen wins)
    # Canonical human-readable label for each ICG class index
    ICG_LABEL = {
        15: "person",
        17: "car",
        18: "bicycle",
    }

    per_object = []

    # Unmapped classes — report as skipped
    for seg in unmapped:
        per_object.append({
            "class_name": seg["class_name"],
            "iou":        None,
            "precision":  None,
            "recall":     None,
            "confidence": None,
            "note":       "no ICG class mapping",
        })

    for icg_index, segs in icg_groups.items():
        gt_mask = load_ground_truth_mask(image_id, icg_index)

        label = ICG_LABEL.get(icg_index, f"class_{icg_index}")

        if gt_mask is None:
            per_object.append({
                "class_name": label,
                "iou":        None,
                "precision":  None,
                "recall":     None,
                "confidence": None,
                "note":       f"ground truth file not found for image {image_id}",
            })
            continue

        # Merge ALL SAM masks that share this ICG class into one predicted mask
        merged_pred = np.zeros(gt_mask.shape, dtype=bool)
        for seg in segs:
            pred_mask = seg["mask"].astype(bool)
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            merged_pred |= pred_mask

        iou               = calculate_iou(merged_pred, gt_mask)
        precision, recall = calculate_precision_recall(merged_pred, gt_mask)
        mean_conf         = float(np.mean([s.get("confidence") or 0.0 for s in segs]))

        per_object.append({
            "class_name": label,
            "iou":        round(iou, 4),
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "confidence": round(mean_conf, 2),
            "note":       f"merged {len(segs)} mask(s) → ICG class {icg_index}",
        })

    # ── Per-class summary ─────────────────────────────────────────────────────
    per_class_summary = {}
    for obj in per_object:
        if obj["iou"] is None:
            continue
        cn = obj["class_name"]
        icg_idx = next(
            idx for idx, segs in icg_groups.items()
            if ICG_LABEL.get(idx, f"class_{idx}") == cn
        )
        per_class_summary[cn] = {
            "mean_iou":       obj["iou"],
            "mean_precision": obj["precision"],
            "mean_recall":    obj["recall"],
            "count":          len(icg_groups[icg_idx]),
        }

    # ── Overall metrics ───────────────────────────────────────────────────────
    valid = [o for o in per_object if o["iou"] is not None]
    if valid:
        overall = {
            "mean_iou":        round(float(np.mean([o["iou"]      for o in valid])), 4),
            "mean_precision":  round(float(np.mean([o["precision"] for o in valid])), 4),
            "mean_recall":     round(float(np.mean([o["recall"]    for o in valid])), 4),
            "evaluated_count": len(valid),
        }
    else:
        overall = {
            "mean_iou": 0.0, "mean_precision": 0.0,
            "mean_recall": 0.0, "evaluated_count": 0,
        }

    return {
        "per_object": per_object,
        "per_class":  per_class_summary,
        "overall":    overall,
    }