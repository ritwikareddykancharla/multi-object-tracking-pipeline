# multi-object-tracking-pipeline

Real-time video object detection and multi-object tracking pipeline using YOLO and tracking-by-detection methods.

---

## Overview

This repository implements an end-to-end **real-time video analytics pipeline** for object detection and persistent multi-object tracking. The system is designed to operate under real-time constraints and produces stable object identities across frames along with structured outputs for downstream analysis.

The pipeline follows a **tracking-by-detection** paradigm, separating detection, association, and tracking logic to enable modular experimentation and evaluation.

---

## System Architecture

### Components

**Detection**
- Object detection using **YOLOv8**
- Batched GPU inference using **PyTorch**
- Mixed-precision execution for improved throughput

**Tracking**
- Multi-object tracking using **ByteTrack** or **DeepSORT**
- Frame-to-frame association for persistent object IDs
- Robustness to occlusion and short-term detector failures

**Outputs**
- Annotated video streams
- Frame-level structured metadata (CSV / JSON)
- Stable object IDs across time

---

### Data Flow

```text
Input Video
     |
     v
+----------------------+
|  Frame Extraction    |
+----------------------+
            |
            v
+----------------------+
|  Object Detection    |
|   YOLOv8 (GPU)       |
+----------------------+
            |
            v
+----------------------+
| Detection Outputs    |
| (bboxes, scores)    |
+----------------------+
            |
            v
+----------------------+
| Multi-Object Tracker |
| ByteTrack / DeepSORT |
+----------------------+
            |
            v
+-----------------------------+
| Tracked Objects per Frame   |
| (ID, bbox, confidence)     |
+-----------------------------+
            |
            v
+-----------------------------+
| Outputs                     |
| - Annotated Video           |
| - CSV / JSON Metadata       |
+-----------------------------+
````

---

## Evaluation

Tracking performance is evaluated using standard multi-object tracking metrics:

* **IDF1** — identity preservation across frames
* **MOTA** — overall tracking accuracy
* Analysis of failure modes including:

  * Occlusion-induced ID switches
  * Detector–tracker coupling errors
  * Missed detections under motion blur

Evaluation is performed on full video sequences to capture temporal behavior.

---

## Repository Structure

```text
video-detection-tracking/
├── detection/          # YOLO-based detection modules
├── tracking/           # ByteTrack / DeepSORT implementations
├── inference/          # Batched GPU inference logic
├── eval/               # Tracking metrics and evaluation scripts
├── visualization/      # Video rendering and overlays
├── scripts/            # Video preprocessing and runners
├── README.md
├── LICENSE
└── .gitignore
```

---

## Design Principles

* Explicit separation of detection and tracking stages
* Real-time performance considerations
* Structured outputs for downstream consumption
* Evaluation-driven iteration and failure analysis
* Production-oriented modular design

---

## Notes

This project is intended as an applied computer vision systems demonstration and is structured to support extension to streaming inputs, alternative detectors, and large-scale deployment scenarios.


## License

MIT License
