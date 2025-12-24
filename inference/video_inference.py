"""
Video Inference Pipeline

Responsibilities:
- Read video frames
- Run object detection
- Update multi-object tracker
- Collect structured per-frame tracking outputs

This module orchestrates the full detection + tracking system.
"""

from typing import List, Dict
import cv2
import numpy as np
from tqdm import tqdm

from detection.yolo_detector import YOLODetector
from tracking.bytetrack import ByteTracker


class VideoInferenceEngine:
    def __init__(
        self,
        detector: YOLODetector,
        tracker: ByteTracker,
        batch_size: int = 1,
    ):
        """
        Args:
            detector: object detector instance
            tracker: multi-object tracker instance
            batch_size: number of frames per detection batch
        """
        self.detector = detector
        self.tracker = tracker
        self.batch_size = batch_size

    def run(self, video_path: str) -> List[Dict]:
        """
        Run detection + tracking on a video file.

        Args:
            video_path: path to input video

        Returns:
            results: list of dicts per frame
                {
                  "frame_idx": int,
                  "objects": [
                      {"id": int, "bbox": [x1,y1,x2,y2], "confidence": float}
                  ]
                }
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Failed to open video: {video_path}"

        results = []
        frame_buffer = []
        frame_indices = []

        frame_idx = 0

        pbar = tqdm(desc="Running video inference")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(frame)
            frame_indices.append(frame_idx)

            # Run detection when batch is full
            if len(frame_buffer) == self.batch_size:
                batch_results = self._process_batch(
                    frame_buffer, frame_indices
                )
                results.extend(batch_results)

                frame_buffer.clear()
                frame_indices.clear()

            frame_idx += 1
            pbar.update(1)

        # Process remaining frames
        if len(frame_buffer) > 0:
            batch_results = self._process_batch(
                frame_buffer, frame_indices
            )
            results.extend(batch_results)

        cap.release()
        pbar.close()
        return results

    def _process_batch(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
    ) -> List[Dict]:
        """
        Run detection on a batch and update tracker frame-by-frame.
        """
        detections_batch = self.detector.detect(frames)

        batch_outputs = []

        for frame_idx, detections in zip(frame_indices, detections_batch):
            tracked_objects = self.tracker.update(detections)

            batch_outputs.append(
                {
                    "frame_idx": frame_idx,
                    "objects": tracked_objects,
                }
            )

        return batch_outputs
