"""
Tracking Evaluation Metrics

Implements lightweight tracking metrics inspired by MOTChallenge:
- IDF1 (identity preservation)
- MOTA (overall tracking accuracy)

This module assumes frame-level tracked outputs with persistent IDs.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np


def _bbox_iou(a, b) -> float:
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_idf1(
    gt: List[Dict],
    preds: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute IDF1 score.

    Args:
        gt: ground-truth per-frame annotations
        preds: predicted per-frame tracked outputs
        iou_threshold: IoU threshold for matching

    Returns:
        IDF1 score
    """
    assert len(gt) == len(preds)

    idtp = 0  # true positives (ID correct)
    idfp = 0
    idfn = 0

    for gt_frame, pr_frame in zip(gt, preds):
        gt_objs = gt_frame["objects"]
        pr_objs = pr_frame["objects"]

        matched_gt = set()
        matched_pr = set()

        for i, g in enumerate(gt_objs):
            for j, p in enumerate(pr_objs):
                if j in matched_pr:
                    continue
                if _bbox_iou(g["bbox"], p["bbox"]) >= iou_threshold:
                    if g["id"] == p["id"]:
                        idtp += 1
                    else:
                        idfp += 1
                        idfn += 1
                    matched_gt.add(i)
                    matched_pr.add(j)
                    break

        idfn += len(gt_objs) - len(matched_gt)
        idfp += len(pr_objs) - len(matched_pr)

    if idtp == 0:
        return 0.0

    precision = idtp / (idtp + idfp)
    recall = idtp / (idtp + idfn)
    return 2 * precision * recall / (precision + recall)


def compute_mota(
    gt: List[Dict],
    preds: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute MOTA score.

    Args:
        gt: ground-truth per-frame annotations
        preds: predicted per-frame tracked outputs
        iou_threshold: IoU threshold for matching

    Returns:
        MOTA score
    """
    assert len(gt) == len(preds)

    misses = 0
    false_positives = 0
    id_switches = 0
    total_gt = 0

    prev_matches = {}

    for gt_frame, pr_frame in zip(gt, preds):
        gt_objs = gt_frame["objects"]
        pr_objs = pr_frame["objects"]

        total_gt += len(gt_objs)

        matched_gt = set()
        matched_pr = set()

        for i, g in enumerate(gt_objs):
            best_j = -1
            best_iou = 0.0
            for j, p in enumerate(pr_objs):
                iou = _bbox_iou(g["bbox"], p["bbox"])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_j = j

            if best_j >= 0:
                matched_gt.add(i)
                matched_pr.add(best_j)

                prev_id = prev_matches.get(i)
                if prev_id is not None and prev_id != pr_objs[best_j]["id"]:
                    id_switches += 1
                prev_matches[i] = pr_objs[best_j]["id"]

        misses += len(gt_objs) - len(matched_gt)
        false_positives += len(pr_objs) - len(matched_pr)

    if total_gt == 0:
        return 0.0

    return 1.0 - (misses + false_positives + id_switches) / total_gt
