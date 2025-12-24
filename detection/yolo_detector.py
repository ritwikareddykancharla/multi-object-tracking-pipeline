"""
YOLOv8 Detector Module

Responsible ONLY for object detection.
- Loads YOLOv8 model
- Runs batched inference on frames
- Returns clean detection outputs per frame

Tracking, video I/O, and visualization are handled elsewhere.
"""

from typing import List
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics YOLO not installed. Run: pip install ultralytics"
    )


class YOLODetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        use_fp16: bool = True,
    ):
        """
        Args:
            model_name: YOLOv8 model checkpoint
            device: 'cuda' or 'cpu'
            conf_threshold: confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            use_fp16: enable mixed precision inference on GPU
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_fp16 = use_fp16 and device == "cuda"

        self.model = YOLO(model_name)
        self.model.to(device)

        if self.use_fp16:
            self.model.model.half()

    @torch.no_grad()
    def detect(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run object detection on a batch of frames.

        Args:
            frames: list of frames (H x W x 3, BGR or RGB)

        Returns:
            detections_per_frame: list of arrays, one per frame
            Each array shape: (N, 6)
            Columns: [x1, y1, x2, y2, confidence, class_id]
        """
        if len(frames) == 0:
            return []

        # Ultralytics handles preprocessing internally
        results = self.model(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections_per_frame = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                detections_per_frame.append(
                    np.zeros((0, 6), dtype=np.float32)
                )
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            detections = np.concatenate(
                [
                    boxes,
                    scores[:, None],
                    classes[:, None],
                ],
                axis=1,
            ).astype(np.float32)

            detections_per_frame.append(detections)

        return detections_per_frame
