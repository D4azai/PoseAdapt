"""Estimator construction without coupling callers to a backend class."""

from __future__ import annotations

from pathlib import Path

from poseadapt.models.base import PoseEstimator
from poseadapt.models.rtmpose import RTMPoseEstimator
from poseadapt.models.vitpose import ViTPoseEstimator


def create_estimator(
    name: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "auto",
) -> PoseEstimator:
    """Create a supported estimator from a normalized model name."""
    estimators = {
        "vitpose": ViTPoseEstimator,
        "rtmpose": RTMPoseEstimator,
    }
    normalized = name.strip().lower()
    try:
        estimator_type = estimators[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(estimators))
        raise ValueError(f"Unknown model {name!r}; expected one of: {choices}") from error
    return estimator_type(config_path, checkpoint_path, device)
