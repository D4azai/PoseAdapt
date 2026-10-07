from types import SimpleNamespace

import numpy as np
import pytest

from poseadapt.models.base import PosePrediction
from poseadapt.models.mmpose import MMPoseEstimator


def test_pose_prediction_requires_k_by_three_keypoints() -> None:
    with pytest.raises(ValueError, match="shape"):
        PosePrediction(np.zeros((17, 2), dtype=np.float32))


def test_mmpose_sample_conversion_preserves_scores_and_box() -> None:
    instances = SimpleNamespace(
        keypoints=np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32),
        keypoint_scores=np.array([[0.9, 0.7]], dtype=np.float32),
        bboxes=np.array([[1.0, 2.0, 50.0, 60.0]], dtype=np.float32),
        bbox_scores=np.array([0.95], dtype=np.float32),
    )

    predictions = MMPoseEstimator._convert_samples([SimpleNamespace(pred_instances=instances)])

    assert len(predictions) == 1
    np.testing.assert_allclose(
        predictions[0].keypoints,
        np.array([[10.0, 20.0, 0.9], [30.0, 40.0, 0.7]], dtype=np.float32),
    )
    np.testing.assert_allclose(predictions[0].bbox, [1.0, 2.0, 50.0, 60.0])
    assert predictions[0].score == pytest.approx(0.95)


def test_preprocess_rejects_non_uint8_images() -> None:
    estimator = object.__new__(MMPoseEstimator)
    with pytest.raises(TypeError, match="uint8"):
        estimator.preprocess(np.zeros((8, 8, 3), dtype=np.float32))


def test_bbox_validation_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        MMPoseEstimator._validate_bboxes(np.zeros((3, 5), dtype=np.float32))
