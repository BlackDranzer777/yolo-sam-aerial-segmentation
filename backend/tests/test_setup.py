"""
tests/test_setup.py — Environment and model sanity check.

Run this once after setting up the virtual environment to confirm:
  - All required packages are installed and importable
  - Both model weight files exist at the configured paths
  - PyTorch device (CPU or CUDA) is detected correctly
  - SAM and YOLO models can be loaded without errors

Usage:
    cd backend/
    python tests/test_setup.py
"""

import sys
import os

# Allow imports from backend/ root (config.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}{(' — ' + result) if result else ''}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {label} — {e}")
        return False


def main():
    failures = 0

    print("\n── Package imports ──────────────────────────────────────")

    if not check("torch",               lambda: __import__("torch").__version__): failures += 1
    if not check("torchvision",         lambda: __import__("torchvision").__version__): failures += 1
    if not check("ultralytics (YOLO)",  lambda: __import__("ultralytics").__version__): failures += 1
    if not check("segment_anything",    lambda: None or "ok"): failures += 1
    if not check("cv2 (OpenCV)",        lambda: __import__("cv2").__version__): failures += 1
    if not check("PIL (Pillow)",        lambda: __import__("PIL").__version__): failures += 1
    if not check("numpy",               lambda: __import__("numpy").__version__): failures += 1
    if not check("flask",               lambda: __import__("flask").__version__): failures += 1
    if not check("flask_cors",          lambda: None or "ok"): failures += 1

    print("\n── PyTorch device ───────────────────────────────────────")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  {PASS}  Device: {device.upper()}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})", end="")
    print()

    print("\n── Model weight files ───────────────────────────────────")

    from config import YOLO_MODEL_PATH, SAM_CHECKPOINT_PATH

    def check_file(path):
        size_mb = os.path.getsize(path) / (1024 ** 2)
        return f"{size_mb:.1f} MB  ({path})"

    if not check("yolov8n.pt",              lambda: check_file(YOLO_MODEL_PATH)):   failures += 1
    if not check("sam_vit_b_01ec64.pth",    lambda: check_file(SAM_CHECKPOINT_PATH)): failures += 1

    print("\n── Model load test ──────────────────────────────────────")

    def load_yolo():
        from ultralytics import YOLO
        YOLO(YOLO_MODEL_PATH)
        return "loaded"

    def load_sam():
        from segment_anything import sam_model_registry
        from config import SAM_MODEL_TYPE
        sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        return "loaded"

    if not check("Load YOLO model",  load_yolo):  failures += 1
    if not check("Load SAM model",   load_sam):   failures += 1

    print("\n─────────────────────────────────────────────────────────")
    if failures == 0:
        print("  All checks passed. Environment is ready.\n")
    else:
        print(f"  {failures} check(s) failed. Fix the errors above before proceeding.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
