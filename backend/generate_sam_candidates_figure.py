"""
generate_sam_candidates_figure.py

Produces a 4-panel figure showing the three SAM candidate masks for one
bounding box prompt, with the selected (highest-IoU) mask highlighted.

Run from the backend/ directory with the venv activated:
    python generate_sam_candidates_figure.py

Output:
    ../thesis/figures/fig_sam_three_candidates.png
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Allow importing config and modules from backend/ ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


# ── Settings ──────────────────────────────────────────────────────────────────

# ICG image to use (must exist in ORIGINAL_IMAGES_DIR)
ICG_IMAGE_ID  = "004"
IMAGE_PATH    = os.path.join(config.ORIGINAL_IMAGES_DIR, f"{ICG_IMAGE_ID}.jpg")

# Output path (thesis figures folder)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(SCRIPT_DIR, "..", "thesis", "figures")
OUT_PATH    = os.path.join(OUT_DIR, "fig_sam_three_candidates.png")

# Confidence threshold for detection (use baseline)
CONF_THRESHOLD = 0.25


# ── Load models ───────────────────────────────────────────────────────────────

print("Loading YOLOv8 ...")
yolo = YOLO(config.YOLO_MODEL_PATH)

print("Loading SAM ViT-B ...")
sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT_PATH)
sam.to(device=config.DEVICE)
predictor = SamPredictor(sam)


# ── Load image and run YOLO ───────────────────────────────────────────────────

if not os.path.exists(IMAGE_PATH):
    sys.exit(f"ERROR: Image not found at {IMAGE_PATH}\n"
             f"Make sure ICG image {ICG_IMAGE_ID}.jpg exists in ORIGINAL_IMAGES_DIR.")

print(f"Running YOLO on image {ICG_IMAGE_ID} ...")
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

results = yolo.predict(
    source=IMAGE_PATH,
    conf=CONF_THRESHOLD,
    iou=config.YOLO_IOU_THRESHOLD,
    device=config.DEVICE,
    verbose=False,
)

detections = []
for box in results[0].boxes:
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    detections.append({
        "bbox":       [x1, y1, x2, y2],
        "class_name": yolo.names[int(box.cls[0])],
        "confidence": round(float(box.conf[0]), 4),
    })

# Pick the highest-confidence car detection for the figure
car_detections = [d for d in detections if d["class_name"] == "car"]
if not car_detections:
    # Fall back to whatever has highest confidence
    car_detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

if not car_detections:
    sys.exit("ERROR: No detections found. Check confidence threshold or image path.")

best_det = max(car_detections, key=lambda d: d["confidence"])
print(f"Selected detection: {best_det['class_name']} (conf={best_det['confidence']})")


# ── Pad bounding box ──────────────────────────────────────────────────────────

x1, y1, x2, y2 = best_det["bbox"]
pad_x = int((x2 - x1) * config.BBOX_PADDING_FRACTION)
pad_y = int((y2 - y1) * config.BBOX_PADDING_FRACTION)
h, w  = image_rgb.shape[:2]
x1p   = max(0, x1 - pad_x)
y1p   = max(0, y1 - pad_y)
x2p   = min(w, x2 + pad_x)
y2p   = min(h, y2 + pad_y)
padded_box = np.array([x1p, y1p, x2p, y2p], dtype=float)


# ── Run SAM and get all three candidate masks ─────────────────────────────────

print("Encoding image with SAM (this takes ~90 s on CPU) ...")
predictor.set_image(image_rgb)

print("Predicting three candidate masks ...")
masks, scores, _ = predictor.predict(
    box=padded_box,
    multimask_output=True,   # returns exactly 3 masks
)

# masks  shape: (3, H, W)  dtype: bool
# scores shape: (3,)       higher = better predicted IoU

best_idx = int(np.argmax(scores))
print(f"Scores: {[round(float(s), 4) for s in scores]}  →  selected mask {best_idx + 1}")


# ── Crop the region of interest for display ───────────────────────────────────

# Add generous margin around the padded box for the display crop
margin = 60
cx1 = max(0, x1p - margin)
cy1 = max(0, y1p - margin)
cx2 = min(w, x2p + margin)
cy2 = min(h, y2p + margin)

crop_rgb  = image_rgb[cy1:cy2, cx1:cx2]


def overlay_mask_on_crop(image_crop, full_mask, cy1, cx1, cy2, cx2,
                          colour=(0, 200, 80), alpha=0.50):
    """Alpha-blend a full-image binary mask onto a cropped region."""
    mask_crop = full_mask[cy1:cy2, cx1:cx2]
    out = image_crop.copy().astype(np.float32)
    col = np.array(colour, dtype=np.float32)
    out[mask_crop] = out[mask_crop] * (1 - alpha) + col * alpha

    # Draw contour
    contours, _ = cv2.findContours(
        mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    out_uint8 = out.astype(np.uint8)
    cv2.drawContours(out_uint8, contours, -1, colour, 2)
    return out_uint8


MASK_COLOURS = [
    (100, 180, 255),   # blue  — mask 1
    (80,  200,  80),   # green — mask 2
    (255, 140,  60),   # orange— mask 3
]

panels = []
for i in range(3):
    panel = overlay_mask_on_crop(
        crop_rgb, masks[i], cy1, cx1, cy2, cx2,
        colour=MASK_COLOURS[i], alpha=0.50
    )
    panels.append(panel)


# ── Build the figure ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.patch.set_facecolor("white")

titles = [
    f"Candidate 1\nSAM IoU score: {scores[0]:.3f}",
    f"Candidate 2\nSAM IoU score: {scores[1]:.3f}",
    f"Candidate 3\nSAM IoU score: {scores[2]:.3f}",
    "Original crop\n(padded box shown)",
]

for ax, panel, title in zip(axes[:3], panels, titles[:3]):
    ax.imshow(panel)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

# Panel 4: original crop with padded box drawn
orig_crop = crop_rgb.copy()
box_x1 = x1p - cx1
box_y1 = y1p - cy1
box_x2 = x2p - cx1
box_y2 = y2p - cy1
orig_with_box = cv2.rectangle(
    orig_crop.copy(), (box_x1, box_y1), (box_x2, box_y2), (255, 50, 50), 2
)
axes[3].imshow(orig_with_box)
axes[3].set_title(titles[3], fontsize=11)
axes[3].axis("off")

# Highlight the selected panel with a coloured border
selected_colour = [c / 255 for c in MASK_COLOURS[best_idx]]
for spine in axes[best_idx].spines.values():
    spine.set_visible(True)
    spine.set_edgecolor(selected_colour)
    spine.set_linewidth(4)

# Legend
selected_patch = mpatches.Patch(
    facecolor=selected_colour,
    edgecolor="black",
    linewidth=0.5,
    label=f"Selected mask (Candidate {best_idx + 1}, highest IoU score)",
)
fig.legend(
    handles=[selected_patch],
    loc="lower center",
    ncol=1,
    fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.5, -0.04),
)

fig.suptitle(
    f"SAM three-candidate mask selection — {best_det['class_name']} detection "
    f"(conf={best_det['confidence']}) in ICG image {ICG_IMAGE_ID}",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nFigure saved to: {OUT_PATH}")
