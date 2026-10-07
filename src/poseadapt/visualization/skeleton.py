"""OpenCV rendering for COCO-format pose predictions."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from poseadapt.models.base import PosePrediction

# COCO keypoint indices: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)
UPPER_LIMB_INDICES = frozenset({5, 6, 7, 8, 9, 10})


def draw_pose_predictions(
    image: NDArray[np.uint8],
    predictions: list[PosePrediction],
    *,
    keypoint_threshold: float = 0.3,
    draw_bbox: bool = True,
    draw_labels: bool = True,
) -> NDArray[np.uint8]:
    """Render skeletons on a copy of a BGR image."""
    if not 0.0 <= keypoint_threshold <= 1.0:
        raise ValueError("keypoint_threshold must be between 0 and 1")
    canvas = np.ascontiguousarray(image.copy())

    for person_index, prediction in enumerate(predictions):
        points = prediction.keypoints
        for start, end in COCO_SKELETON:
            if max(start, end) >= len(points):
                continue
            if points[start, 2] >= keypoint_threshold and points[end, 2] >= keypoint_threshold:
                color = (0, 165, 255) if {start, end} & UPPER_LIMB_INDICES else (90, 220, 90)
                cv2.line(canvas, _pixel(points[start]), _pixel(points[end]), color, 2, cv2.LINE_AA)

        for index, point in enumerate(points):
            if point[2] < keypoint_threshold:
                continue
            color = (0, 80, 255) if index in UPPER_LIMB_INDICES else (255, 180, 40)
            cv2.circle(canvas, _pixel(point), 4, color, -1, cv2.LINE_AA)

        if draw_bbox and prediction.bbox is not None:
            x1, y1, x2, y2 = prediction.bbox.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (235, 210, 60), 2)
        if draw_labels:
            label = f"person {person_index + 1}"
            if prediction.score is not None:
                label += f"  {prediction.score:.2f}"
            anchor = _label_anchor(prediction, person_index)
            cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return canvas


def _pixel(point: NDArray[np.float32]) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _label_anchor(prediction: PosePrediction, person_index: int) -> tuple[int, int]:
    if prediction.bbox is not None:
        return int(prediction.bbox[0]), max(18, int(prediction.bbox[1]) - 8)
    return 12, 24 + 22 * person_index
