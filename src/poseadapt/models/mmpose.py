"""MMPose adapter shared by ViTPose and RTMPose configurations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from poseadapt.models.base import PoseEstimator, PosePrediction
from poseadapt.utils.device import resolve_device

LOGGER = logging.getLogger(__name__)


class MMPoseEstimator(PoseEstimator):
    """Top-down MMPose estimator with lazy optional imports.

    When no person boxes are supplied, MMPose treats the full image as a single
    person region. Person detection will be added to the video pipeline phase.
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.config_path = self._require_file(config_path, "model config")
        self.checkpoint_path = self._require_file(checkpoint_path, "checkpoint")
        self.requested_device = device
        self.device: str | None = None
        self.model: Any | None = None

    @staticmethod
    def _require_file(path: str | Path, description: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{description.capitalize()} not found: {resolved}")
        return resolved

    @staticmethod
    def _optional_apis() -> tuple[Any, Any]:
        try:
            from mmpose.apis import inference_topdown, init_model
        except ImportError as error:
            raise RuntimeError(
                "MMPose is not installed. Use Python 3.10/3.11, install PyTorch for "
                "your platform, then run `pip install -e .[openmmlab]`."
            ) from error
        return init_model, inference_topdown

    def load_model(self) -> None:
        """Initialize MMPose from local configuration and checkpoint files."""
        init_model, _ = self._optional_apis()
        self.device = resolve_device(self.requested_device, require_torch=True)
        LOGGER.info("Loading MMPose model on %s", self.device)
        self.model = init_model(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )
        self.model.eval()

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Validate a non-empty uint8 BGR image and make it contiguous."""
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise TypeError(f"Expected a uint8 image, received {array.dtype}")
        if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
            raise ValueError("Expected a non-empty BGR image with shape (H, W, 3)")
        return np.ascontiguousarray(array)

    def predict(
        self,
        image: NDArray[np.uint8],
        person_bboxes: NDArray[np.float32] | None = None,
    ) -> list[PosePrediction]:
        """Run top-down pose estimation."""
        if self.model is None:
            raise RuntimeError("Model is not loaded; call load_model() first")
        image = self.preprocess(image)
        boxes = self._validate_bboxes(person_bboxes)
        _, inference_topdown = self._optional_apis()
        samples = inference_topdown(
            self.model,
            image,
            bboxes=boxes,
            bbox_format="xyxy",
        )
        return self._convert_samples(samples)

    @staticmethod
    def _validate_bboxes(
        bboxes: NDArray[np.float32] | None,
    ) -> NDArray[np.float32] | None:
        if bboxes is None:
            return None
        array = np.asarray(bboxes, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("person_bboxes must have shape (N, 4) in xyxy format")
        return array

    @classmethod
    def _convert_samples(cls, samples: list[Any]) -> list[PosePrediction]:
        predictions: list[PosePrediction] = []
        for sample in samples:
            instances = sample.pred_instances
            coordinates = cls._as_numpy(instances.keypoints)
            if coordinates.ndim == 2:
                coordinates = coordinates[None, ...]

            raw_scores = getattr(instances, "keypoint_scores", None)
            if raw_scores is None:
                scores = np.ones(coordinates.shape[:2], dtype=np.float32)
            else:
                scores = cls._as_numpy(raw_scores)
                if scores.ndim == 1:
                    scores = scores[None, ...]

            raw_boxes = getattr(instances, "bboxes", None)
            boxes = None if raw_boxes is None else cls._as_numpy(raw_boxes)
            raw_instance_scores = getattr(instances, "bbox_scores", None)
            instance_scores = (
                None
                if raw_instance_scores is None
                else cls._as_numpy(raw_instance_scores).reshape(-1)
            )

            for index, (points, confidences) in enumerate(zip(coordinates, scores, strict=True)):
                keypoints = np.column_stack((points[:, :2], confidences)).astype(np.float32)
                bbox = None if boxes is None else np.asarray(boxes[index, :4], dtype=np.float32)
                score = None if instance_scores is None else float(instance_scores[index])
                predictions.append(PosePrediction(keypoints=keypoints, bbox=bbox, score=score))
        return predictions

    @staticmethod
    def _as_numpy(value: Any) -> NDArray[np.float32]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the current state dictionary without serializing executable code."""
        if self.model is None:
            raise RuntimeError("Model is not loaded; nothing can be saved")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to save a checkpoint") from error
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, destination)
