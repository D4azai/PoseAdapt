"""Backend-neutral pose estimator contract and prediction types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PosePrediction:
    """One person's 2D keypoints in image coordinates.

    `keypoints` has shape ``(K, 3)`` with ``x, y, confidence`` columns. A bounding
    box, when available, uses ``(x_min, y_min, x_max, y_max)``.
    """

    keypoints: NDArray[np.float32]
    bbox: NDArray[np.float32] | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.keypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("keypoints must have shape (K, 3): x, y, confidence")
        object.__setattr__(self, "keypoints", points)

        if self.bbox is not None:
            box = np.asarray(self.bbox, dtype=np.float32).reshape(-1)
            if box.shape != (4,):
                raise ValueError("bbox must contain four xyxy coordinates")
            object.__setattr__(self, "bbox", box)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "keypoints": self.keypoints.tolist(),
            "bbox": None if self.bbox is None else self.bbox.tolist(),
            "score": self.score,
        }


class PoseEstimator(ABC):
    """Interface implemented by pose-estimation backends."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model configuration and weights."""

    @abstractmethod
    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Validate and prepare a BGR image."""

    @abstractmethod
    def predict(
        self,
        image: NDArray[np.uint8],
        person_bboxes: NDArray[np.float32] | None = None,
    ) -> list[PosePrediction]:
        """Estimate poses for an image and optional person boxes."""

    def get_keypoints(
        self,
        image: NDArray[np.uint8],
        person_bboxes: NDArray[np.float32] | None = None,
    ) -> list[NDArray[np.float32]]:
        """Convenience method returning only keypoint arrays."""
        return [prediction.keypoints for prediction in self.predict(image, person_bboxes)]

    def train(self, *_: Any, **__: Any) -> None:
        """Training hook to be implemented in the training phase."""
        raise NotImplementedError("Training is planned for Phase G and is not implemented yet")

    @abstractmethod
    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights to a local checkpoint."""
