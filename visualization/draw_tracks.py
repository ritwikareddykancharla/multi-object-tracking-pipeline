"""
Visualization Utilities for Multi-Object Tracking

Responsibilities:
- Draw bounding boxes
- Draw persistent object IDs
- Assign consistent colors per track ID

This module is intentionally stateless and pure.
"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
import random


def _get_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generate a consistent color for a given track ID.
    """
    random.seed(track_id)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )


def draw_tracks(
    frame: np.ndarray,
    tracked_objects: List[Dict],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw tracked objects on a frame.

    Args:
        frame: input image (H, W, 3)
        tracked_objects: list of dicts
            {
              "id": int,
              "bbox": [x1,y1,x2,y2],
              "confidence": float
            }
        thickness: bounding box line thickness

    Returns:
        annotated_frame: frame with boxes and IDs drawn
    """
    annotated_frame = frame.copy()

    for obj in tracked_objects:
        track_id = obj["id"]
        x1, y1, x2, y2 = map(int, obj["bbox"])
        conf = obj["confidence"]

        color = _get_color(track_id)

        # Draw bounding box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        label = f"ID {track_id} | {conf:.2f}"

        # Compute text size
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        # Draw label background
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_h - 6),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated_frame
