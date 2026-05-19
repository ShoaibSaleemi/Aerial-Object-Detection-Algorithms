# Aerial Object Detection — Bird vs. Drone Classification

A thesis project for detecting and classifying aerial objects (birds, drones, and unknowns) using an ensemble of object detection models with advanced post-processing and multi-object tracking.

---

## Overview

This repository implements a full pipeline for aerial object detection:

1. **Dataset Preprocessing** — label remapping, OOD filtering, and format conversion
2. **Model Training** — multiple YOLO variants and Faster R-CNN (ResNet-50 FPN)
3. **Model Evaluation** — per-class and macro-averaged metrics with confusion matrices
4. **Ensemble Fusion** — Weighted Boxes Fusion (WBF) and WC-NMS across all models
5. **Hyperparameter Tuning** — Bayesian optimisation (Optuna) for fusion and confidence thresholds
6. **Video Inference** — real-time detection with Kalman filter + LSTM multi-object tracking

### Classes

| ID | Label   | Description               |
|----|---------|---------------------------|
| 0  | bird    | Bird (known class)        |
| 1  | drone   | Drone (known class)       |
| 2  | unknown | Ambiguous / out-of-set    |

---

## Repository Structure

```
thesis/
├── data.yaml                          # Dataset configuration (paths + class names)
├── requirements.txt                   # Python dependencies
│
├── dataset/
│   ├── train/       images/ labels/
│   ├── validation/  images/ labels/
│   └── test/        images/ labels/ videos/
│
├── scripts/
│   ├── dataset_preprocess/
│   │   ├── prepare_training_yolo.py          # Full preprocessing pipeline for YOLO
│   │   ├── prepare_training_fasterrcnn.py    # Full preprocessing pipeline for Faster R-CNN
│   │   └── prepare_validating_yolo_fasterrcnn.py
│   ├── model_training/
│   │   ├── training_yolo.py                  # Train any YOLO model
│   │   ├── training_fasterrcnn.py            # Train Faster R-CNN
│   │   ├── training_yolo_kaggle.ipynb        # Kaggle notebook (YOLO)
│   │   └── training_fasterrcnn_kaggle.ipynb  # Kaggle notebook (Faster R-CNN)
│   ├── model_evaluation/
│   │   ├── eval_yolo.py                      # Evaluate individual YOLO models
│   │   ├── eval_fasterrcnn.py                # Evaluate Faster R-CNN
│   │   ├── eval_wbf.py                       # Evaluate WBF ensemble
│   │   ├── eval_wc-nms.py                    # Evaluate WC-NMS ensemble
│   │   ├── plot_confusion_from_cache.py      # Plot confusion matrices from cache
│   │   └── plot_wcnms_confusion_from_cache.py
│   └── inference_video/
│       ├── inference_video.py                # Basic video inference (YOLO)
│       ├── inference_video_wbf_tracking.py   # WBF ensemble + Kalman/LSTM tracking
│       └── inference_video_wbf_tracking_stl.py  # WBF + tracking + STL/RTM overrides
│
├── tools/
│   ├── labels_train.py / labels_validation.py  # Label ID remapping
│   ├── convert_labels_fasterrcnn.py             # Convert YOLO labels → Faster R-CNN format
│   ├── tune_wbf_3.py / tune_wbf_6.py           # Bayesian tuning for 3/6-model WBF
│   ├── tune_wbf_tracking.py                     # Bayesian tuning for WBF + tracking
│   ├── tune_yolo_f1.py / tune_yolo_f1_sweep.py # Confidence threshold tuning for YOLO
│   ├── tune_fasterrcnn_f1.py                    # Confidence threshold tuning for Faster R-CNN
│   ├── plot_pr_curves.py                        # PR curve plotting
│   ├── plot_yolo_f1_curve.py                    # F1 curve plotting
│   ├── check_dataset_overlap.py                 # Detect train/test overlap
│   ├── split_by_class.py                        # Split dataset by class
│   ├── eval_yolo_map50.py                       # mAP@50 evaluation (YOLO)
│   ├── eval_fasterrcnn_map50.py                 # mAP@50 evaluation (Faster R-CNN)
│   └── gui_remap.py / gui_delete_empty.py       # GUI utilities
│
└── runs/
    ├── detect/          # YOLO training runs and ensemble outputs
    ├── fasterrcnn/      # Faster R-CNN training runs
    └── eval_wcnms/      # WC-NMS evaluation results per model
```

---

## Models

| Model       | Backbone / Variant           | Framework       |
|-------------|------------------------------|-----------------|
| YOLOv8n     | YOLOv8 nano                  | Ultralytics     |
| YOLOv8m     | YOLOv8 medium                | Ultralytics     |
| YOLOv8s     | YOLOv8 small                 | Ultralytics     |
| YOLOv9t     | YOLOv9 tiny                  | Ultralytics     |
| YOLOv10n    | YOLOv10 nano                 | Ultralytics     |
| YOLOv11n    | YOLO11 nano                  | Ultralytics     |
| YOLOv12n    | YOLO12 nano                  | Ultralytics     |
| YOLO26n     | YOLO26 nano                  | Ultralytics     |
| Faster R-CNN| ResNet-50 FPN                | torchvision     |

All YOLO models are fine-tuned from pretrained weights. Faster R-CNN uses `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` as backbone initialisation.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd thesis

# Install dependencies (CUDA 12.6 build of PyTorch)
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` installs PyTorch with CUDA 12.6 support. Adjust the `--index-url` line for your CUDA version or for CPU-only use.

### Minimal Example (GPU required)

```bash
# 1. Preprocess dataset
python scripts/dataset_preprocess/prepare_training_yolo.py

# 2. Train a single YOLO model (edit model choice inside script)
python scripts/model_training/training_yolo.py

# 3. Evaluate the trained model
python scripts/model_evaluation/eval_yolo.py

# 4. Run inference on a video with WBF ensemble
python scripts/inference_video/inference_video_wbf_tracking.py
```

---

## Installation (Detailed)

```bash
# Clone the repository
git clone <repo-url>
cd thesis

# Install dependencies (CUDA 12.6 build of PyTorch)
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` installs PyTorch with CUDA 12.6 support. Adjust the `--index-url` line for your CUDA version or for CPU-only use.

---

## Dataset

The dataset follows the YOLO label format (`class x_center y_center width height`, normalised). Class IDs are remapped during preprocessing:

| Original ID | Original Label | Remapped ID | Remapped Label |
|-------------|----------------|-------------|----------------|
| 1           | bird           | 0           | bird           |
| 2           | drone          | 1           | drone          |
| 0, 3        | airplane / helicopter | 2  | unknown        |

Update `data.yaml` to point to your dataset root before training or evaluation.

---

## Usage

### 1. Preprocess Dataset

```bash
# For YOLO training
python scripts/dataset_preprocess/prepare_training_yolo.py

# For Faster R-CNN training
python scripts/dataset_preprocess/prepare_training_fasterrcnn.py

# For evaluation only (validation/test remapping)
python scripts/dataset_preprocess/prepare_validating_yolo_fasterrcnn.py
```

### 2. Train Models

**YOLO (edit model path and hyperparameters inside the script):**
```bash
python scripts/model_training/training_yolo.py
```

**Faster R-CNN (edit `CONFIG` dict inside the script):**
```bash
python scripts/model_training/training_fasterrcnn.py
```

### 3. Evaluate Models

```bash
# Individual YOLO model evaluation
python scripts/model_evaluation/eval_yolo.py

# Faster R-CNN evaluation
python scripts/model_evaluation/eval_fasterrcnn.py

# WBF ensemble evaluation
python scripts/model_evaluation/eval_wbf.py

# WC-NMS ensemble evaluation
python scripts/model_evaluation/eval_wc-nms.py
```

Results (confusion matrix PNGs, metrics CSVs) are saved under `runs/`.

### 4. Tune Hyperparameters

```bash
# Tune WBF parameters (6-model ensemble, Bayesian optimisation)
python tools/tune_wbf_6.py --trials 100 --seed 42

# Tune WBF parameters (3-model ensemble)
python tools/tune_wbf_3.py --trials 50 --seed 42

# Tune per-model YOLO confidence thresholds
python tools/tune_yolo_f1.py

# Tune Faster R-CNN confidence threshold
python tools/tune_fasterrcnn_f1.py
```

### 5. Video Inference

```bash
# WBF ensemble + Kalman/LSTM tracking
python scripts/inference_video/inference_video_wbf_tracking.py

# WBF + tracking + STL/RTM physical requirement overrides
python scripts/inference_video/inference_video_wbf_tracking_stl.py
```

The scripts prompt for a video file via an interactive selector. Output is saved to `runs/detect/inference_video/`.

---

## Ensemble Methods

### Weighted Boxes Fusion (WBF)

Boxes from all enabled models are clustered by IoU and merged into a single fused detection. Each model has per-class confidence weights tuned via Bayesian optimisation. An *unknown* fallback is applied when:

- Fewer than `MIN_MODEL_SUPPORT` models agree on a detection
- The fused confidence is below `KNOWN_FUSED_CONF_THRESH`
- The score margin between the top two classes is below `SCORE_MARGIN_THRESH`
- The disagreement ratio exceeds `DISAGREEMENT_RATIO_THRESH`

### WC-NMS (Weighted Cluster NMS)

An alternative ensemble approach that operates on raw pre-NMS class scores from multiple YOLO models, clusters overlapping boxes, and applies weighted NMS to produce final detections.

---

## Tracking Pipeline (Video)

Each video frame is processed as follows:

1. All enabled ensemble models run inference on the frame.
2. Detections are clustered and merged with WBF.
3. Fused detections are matched to existing tracks via greedy IoU matching.
4. Each track is smoothed with an 8-D constant-velocity **Kalman filter**.
5. An **LSTM** (window = 8 frames) predicts the next bounding box centre.

Visual output per track includes a coloured bounding box, class label + track ID, a fading trail (last 30 frames), and the LSTM-predicted next position.

### STL/RTM Variant

`inference_video_wbf_tracking_stl.py` additionally applies physical requirement-based overrides:

| Requirement | Rule | Override |
|-------------|------|----------|
| REQ-03 | Fused confidence < 0.65 | → bird |
| REQ-04 | Shape deformation > ε over 10 frames | → bird |

---

## Evaluation Metrics

All evaluation scripts compute and save:

- **Confusion matrix** (normalised by ground-truth column totals or raw counts)
- **Per-class**: Precision, Recall, F1-score
- **Macro-averaged**: Precision, Recall, F1-score
- **Summary**: Overall accuracy, mAP@50 (where applicable)

Results are cached as JSON files under `runs/<model>/eval_cache/` for fast re-plotting without re-running inference.

---

## Requirements

- Python 3.10+
- PyTorch (CUDA 12.6 recommended, CPU supported)
- ultralytics
- torchvision
- numpy, pillow, pyyaml
- questionary (interactive CLI prompts)
- optuna (Bayesian hyperparameter tuning)
- fpdf2, reportlab (report generation)
