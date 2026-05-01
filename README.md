# YOLO + SAM Aerial Image Segmentation

Interactive aerial image segmentation pipeline combining **YOLOv8 object detection** with the **Segment Anything Model (SAM)**. Developed as an MSc thesis project at the Warsaw University of Technology, Faculty of Geodesy and Cartography.

---

## Overview

The system works in two stages:

1. **YOLOv8** (fine-tuned on VisDrone) detects objects in a drone image and returns bounding boxes
2. **SAM ViT-B** uses those bounding boxes as spatial prompts to generate pixel-level segmentation masks

Segmentation quality is evaluated quantitatively against the **ICG Semantic Drone Dataset** ground truth using IoU, Precision, and Recall.

A web application (Flask + React) provides an interactive interface for uploading images, running the pipeline, switching between output views, and inspecting evaluation metrics in real time.

---

## Demo

| YOLO Detection | SAM Segmentation Masks | Combined |
|---|---|---|
| Bounding boxes with class labels | Pixel-level colour overlays | Masks + boxes overlaid |

---

## Project Structure

```
├── backend/                  # Flask REST API + pipeline modules
│   ├── app.py                # API server (6 endpoints)
│   ├── config.py             # Paths, thresholds, class mapping
│   ├── requirements.txt
│   ├── modules/
│   │   ├── yolo_detector.py  # YOLOv8 wrapper
│   │   ├── sam_segmentor.py  # SAM SamPredictor wrapper
│   │   ├── pipeline.py       # Detection → segmentation → visualisation
│   │   └── evaluator.py      # IoU / Precision / Recall computation
│   ├── utils/
│   │   └── visualization.py  # OpenCV drawing utilities
│   ├── tests/                # Unit tests
│   ├── models/               # Place model weights here (see below)
│   ├── uploads/              # Runtime image uploads (auto-created)
│   └── outputs/              # Runtime output images (auto-created)
│
└── frontend/                 # React + Vite SPA
    ├── src/
    │   ├── components/
    │   │   ├── ImageUploader.jsx
    │   │   ├── ResultViewer.jsx
    │   │   ├── ComparisonView.jsx
    │   │   ├── MetricsPanel.jsx
    │   │   └── LayerToggle.jsx
    │   └── App.jsx
    ├── index.html
    └── package.json
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with 4+ GB VRAM (optional but strongly recommended — CPU inference is ~90s/image)

### 1. Download Model Weights

Place the following files in `backend/models/`:

| File | Size | Source |
|------|------|--------|
| `yolov8s-visdrone.pt` | ~22 MB | [Ultralytics HuggingFace Hub](https://huggingface.co/mshamrai/yolov8s-visdrone) |
| `sam_vit_b_01ec64.pth` | 375 MB | [Meta AI Research — SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) |

### 2. Download the ICG Dataset (for evaluation)

Download the **ICG Semantic Drone Dataset** from:
[http://dronedataset.icg.tugraz.at](http://dronedataset.icg.tugraz.at)

Place the `label_images_semantic/` folder at:
```
backend/dataset/ICG/label_images_semantic/
```

> The original aerial RGB images (3.9 GB) are optional — they are only needed for qualitative visualisation. Metric computation uses only the label masks.

### 3. Backend

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

The Flask server starts on `http://localhost:5000`. Both models are loaded at startup (~5–10 s).

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Usage

1. **Upload** a drone aerial image (JPEG or PNG)
2. Click **Run Detection** to see YOLO bounding boxes
3. Click **Run Segmentation** to generate SAM pixel masks
4. Use the **Layer Toggle** to switch between: YOLO boxes / SAM masks / Combined
5. In the **Comparison View**, see detection vs. segmentation side by side
6. In the **Evaluation Panel**, enter the ICG image number (e.g. `004`) to compute IoU, Precision, and Recall against ground truth

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload an image; returns `image_id` |
| POST | `/api/detect/<image_id>` | Run YOLO detection only |
| POST | `/api/segment/<image_id>` | Run full YOLO + SAM pipeline |
| POST | `/api/evaluate/<image_id>` | Evaluate against ICG ground truth |
| GET  | `/api/results/<image_id>` | Retrieve cached pipeline results |
| GET  | `/api/image/<filename>` | Serve an output image |

---

## Performance

| Hardware | SAM encoding | Total per image |
|----------|-------------|-----------------|
| CPU only (i7 12th Gen) | ~85 s | ~90 s |
| GPU (RTX 3060, 6 GB VRAM) | ~0.8 s | ~3–5 s |

---

## Results

Evaluated on the ICG Semantic Drone Dataset (images 001 and 004) in zero-shot mode (no ICG training data used):

| Class | Mean IoU | Mean Precision | Mean Recall |
|-------|----------|----------------|-------------|
| Car | 0.2994 | 0.7884 | 0.3257 |
| Pedestrian | 0.1216 | 0.2171 | 0.1958 |
| Motor | 0.1042 | 0.1368 | 0.2331 |
| Van | 0.3264 | 0.7834 | 0.3588 |
| **Overall** | **0.1919** | **0.4290** | **0.2585** |

---

## Tech Stack

**Backend:** Python 3.12, Flask 3, PyTorch 2, Ultralytics YOLOv8, Meta SAM, OpenCV, NumPy

**Frontend:** React 18, Vite 5, Axios

---

## Citation

If you use this code in your research, please cite:

```
@mastersthesis{aerial-yolo-sam-2025,
  title  = {Analysis and Visualization of Aerial Image Segmentation Using the Segment Anything Model},
  school = {Warsaw University of Technology, Faculty of Geodesy and Cartography},
  year   = {2025}
}
```

---

## Acknowledgements

- [Meta AI Research](https://github.com/facebookresearch/segment-anything) — Segment Anything Model
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8
- [ICG Semantic Drone Dataset](http://dronedataset.icg.tugraz.at) — Graz University of Technology
- [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) — Tianjin University

---

## License

MIT License — see [LICENSE](LICENSE) for details.