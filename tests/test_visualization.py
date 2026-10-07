import numpy as np

from poseadapt.models.base import PosePrediction
from poseadapt.visualization.skeleton import draw_pose_predictions


def test_draw_pose_predictions_does_not_mutate_input() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original = image.copy()
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = [20, 30, 0.9]
    keypoints[7] = [40, 50, 0.9]
    prediction = PosePrediction(keypoints, bbox=np.array([10, 10, 80, 90], dtype=np.float32))

    rendered = draw_pose_predictions(image, [prediction])

    np.testing.assert_array_equal(image, original)
    assert np.count_nonzero(rendered) > 0


def test_draw_pose_predictions_skips_low_confidence_points() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = [10, 10, 0.1]
    keypoints[7] = [30, 30, 0.1]

    rendered = draw_pose_predictions(
        image,
        [PosePrediction(keypoints)],
        keypoint_threshold=0.5,
        draw_labels=False,
    )

    assert np.count_nonzero(rendered) == 0
