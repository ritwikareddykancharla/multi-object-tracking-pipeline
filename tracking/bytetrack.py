"""
ByteTrack-style Multi-Object Tracker

Responsibilities:
- Maintain persistent object IDs across frames
- Associate detections frame-to-frame using IoU
- Handle short-term occlusions via track buffers

This implementation is intentionally simple and readable,
optimized for applied systems clarity over micro-optimizations.
"""

from typing import List, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box_a, box_b) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


class Track:
    """Internal track state."""
    def __init__(self, track_id: int, bbox, confidence: float):
        self.id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0          # total frames alive
        self.time_since_update = 0  # frames since last match

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.time_since_update = 0
        self.age += 1

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1


class ByteTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 30,
        min_confidence: float = 0.3,
    ):
        """
        Args:
            iou_threshold: minimum IoU for association
            max_lost: max frames a track can be unmatched before deletion
            min_confidence: minimum detection confidence to initialize tracks
        """
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_confidence = min_confidence

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def update(self, detections: np.ndarray) -> List[Dict]:
        """
        Update tracker with detections from a single frame.

        Args:
            detections: np.ndarray (N, 6)
                columns = [x1,y1,x2,y2,confidence,class_id]

        Returns:
            tracked_objects: list of dicts
                {
                  "id": int,
                  "bbox": [x1,y1,x2,y2],
                  "confidence": float
                }
        """
        active_tracks = list(self.tracks.values())

        # Filter low-confidence detections
        detections = detections[
            detections[:, 4] >= self.min_confidence
        ] if len(detections) > 0 else detections

        if len(active_tracks) == 0:
            self._init_tracks(detections)
            return self._collect_outputs()

        if len(detections) == 0:
            for track in active_tracks:
                track.mark_missed()
            self._prune_tracks()
            return self._collect_outputs()

        # Compute IoU cost matrix
        cost_matrix = np.zeros((len(active_tracks), len(detections)), dtype=np.float32)

        for i, track in enumerate(active_tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - iou(track.bbox, det[:4])

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_detections = set()

        # Apply matches
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < (1.0 - self.iou_threshold):
                track = active_tracks[r]
                det = detections[c]
                track.update(det[:4], float(det[4]))
                matched_tracks.add(track.id)
                matched_detections.add(c)

        # Mark unmatched tracks
        for track in active_tracks:
            if track.id not in matched_tracks:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                self._create_track(det)

        self._prune_tracks()
        return self._collect_outputs()

    def _init_tracks(self, detections: np.ndarray):
        for det in detections:
            self._create_track(det)

    def _create_track(self, det):
        track = Track(
            track_id=self.next_track_id,
            bbox=det[:4],
            confidence=float(det[4]),
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def _prune_tracks(self):
        to_delete = [
            tid for tid, t in self.tracks.items()
            if t.time_since_update > self.max_lost
        ]
        for tid in to_delete:
            del self.tracks[tid]

    def _collect_outputs(self) -> List[Dict]:
        outputs = []
        for track in self.tracks.values():
            if track.time_since_update == 0:
                outputs.append(
                    {
                        "id": track.id,
                        "bbox": track.bbox.tolist(),
                        "confidence": track.confidence,
                    }
                )
        return outputs
