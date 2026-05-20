"""
app.py — Flask API entry point for the aerial image segmentation system.

Endpoints:
    POST /api/upload                — Upload an image, returns image_id
    POST /api/detect/<image_id>     — Run YOLO detection only
    POST /api/segment/<image_id>    — Run full YOLO → SAM pipeline
    POST /api/evaluate/<image_id>   — Evaluate masks against ICG ground truth
    GET  /api/results/<image_id>    — Return saved results for an image
    GET  /api/image/<filename>      — Serve an output image file
"""

import os
import uuid
import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import config
from modules.yolo_detector import YOLODetector
from modules.pipeline import SegmentationPipeline
from modules.evaluator import evaluate_results

app = Flask(__name__)
CORS(app)

# Initialise models once at startup — avoids reloading weights on every request
detector = YOLODetector()
pipeline = SegmentationPipeline()

# In-memory store for results keyed by image_id.
# Good enough for a thesis demo; no database needed.
_results_store: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def _serialisable(result: dict) -> dict:
    """
    Convert a pipeline result dict to a JSON-serialisable form.
    Numpy bool arrays (masks) are replaced with pixel-count summaries
    because raw mask arrays cannot be JSON-encoded and are too large to
    send over HTTP.  The frontend only needs paths to the saved images.
    """
    seg_summary = []
    for seg in result.get("seg_results", []):
        mask = seg["mask"]
        seg_summary.append({
            "class_name":    seg["class_name"],
            "confidence":    seg["confidence"],
            "bbox":          seg["bbox"],
            "mask_pixels":   int(mask.sum()),
            "total_pixels":  int(mask.size),
            "coverage_pct":  round(100 * mask.sum() / mask.size, 2),
        })

    return {
        "image_id":        result["image_id"],
        "detections":      result["detections"],
        "seg_results":     seg_summary,
        "confidence_used": result.get("confidence_used", config.YOLO_CONFIDENCE_THRESHOLD),
        "output_paths": {
            k: os.path.basename(v)
            for k, v in result["output_paths"].items()
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accept a multipart image upload.

    Request:  multipart/form-data  field name: "image"
    Response: { "image_id": str, "filename": str }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {config.ALLOWED_EXTENSIONS}"}), 400

    ext      = os.path.splitext(file.filename)[1].lower()
    image_id = uuid.uuid4().hex[:8]
    filename = f"{image_id}{ext}"
    save_path = os.path.join(config.UPLOAD_DIR, filename)
    file.save(save_path)

    return jsonify({"image_id": image_id, "filename": filename}), 200


@app.route("/api/detect/<image_id>", methods=["POST"])
def detect(image_id: str):
    """
    Run YOLO detection only (no SAM).

    Optional JSON body: { "confidence": float }
    Response: { "image_id", "detections", "output_paths": { "boxes" }, "confidence_used" }
    """
    image_path = _find_upload(image_id)
    if image_path is None:
        return jsonify({"error": f"Image not found for id: {image_id}"}), 404

    import cv2
    from utils.visualization import draw_boxes, save_visualization

    body = request.get_json(silent=True) or {}
    conf = body.get("confidence")
    if conf is not None:
        conf = float(conf)

    detections = detector.detect(image_path, conf=conf)
    image_bgr  = cv2.imread(image_path)
    boxes_img  = draw_boxes(image_bgr, detections)
    boxes_path = save_visualization(boxes_img, image_id, "boxes")

    result = {
        "image_id":        image_id,
        "detections":      detections,
        "confidence_used": conf if conf is not None else config.YOLO_CONFIDENCE_THRESHOLD,
        "output_paths":    {"boxes": os.path.basename(boxes_path)},
    }
    _results_store[image_id] = result
    return jsonify(result), 200


@app.route("/api/segment/<image_id>", methods=["POST"])
def segment(image_id: str):
    """
    Run the full YOLO → SAM pipeline.

    Optional JSON body: { "confidence": float }
    Response: serialisable pipeline result (see _serialisable helper above)
    """
    image_path = _find_upload(image_id)
    if image_path is None:
        return jsonify({"error": f"Image not found for id: {image_id}"}), 404

    body = request.get_json(silent=True) or {}
    conf = body.get("confidence")
    if conf is not None:
        conf = float(conf)

    models = body.get("models")  # e.g. ["visdrone", "coco"] or subset

    result = pipeline.run(image_path, image_id=image_id, confidence=conf, models=models)
    result["confidence_used"] = conf if conf is not None else config.YOLO_CONFIDENCE_THRESHOLD
    _results_store[image_id] = result

    return jsonify(_serialisable(result)), 200


@app.route("/api/results/<image_id>", methods=["GET"])
def get_results(image_id: str):
    """Return previously computed results for an image_id."""
    result = _results_store.get(image_id)
    if result is None:
        return jsonify({"error": f"No results found for id: {image_id}"}), 404

    return jsonify(_serialisable(result)), 200


@app.route("/api/evaluate/<image_id>", methods=["POST"])
def evaluate(image_id: str):
    """
    Evaluate SAM masks against ICG ground truth.

    image_id  — identifies the seg_results in _results_store (the upload UUID)
    dataset_id — ICG image number for ground truth lookup (e.g. "001").
                 Sent in the JSON body. Falls back to image_id if omitted.

    Response: { per_object, per_class, overall }
    """
    result = _results_store.get(image_id)
    if result is None:
        return jsonify({"error": "No segmentation results found. Run segmentation first."}), 404

    seg_results = result.get("seg_results", [])
    if not seg_results:
        return jsonify({"error": "No masks to evaluate."}), 400

    body = request.get_json(silent=True) or {}
    dataset_id = str(body.get("dataset_id", image_id)).zfill(3)

    metrics = evaluate_results(seg_results, dataset_id)
    return jsonify(metrics), 200


@app.route("/api/image/<filename>", methods=["GET"])
def serve_image(filename: str):
    """Serve a saved output image by filename."""
    return send_from_directory(config.OUTPUT_DIR, filename)


# ── Internal helper ───────────────────────────────────────────────────────────

def _find_upload(image_id: str) -> str | None:
    """
    Find an uploaded file by its image_id prefix.
    Returns the full path, or None if not found.
    """
    for ext in config.ALLOWED_EXTENSIONS:
        candidate = os.path.join(config.UPLOAD_DIR, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
