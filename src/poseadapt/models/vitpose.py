"""ViTPose adapter."""

from poseadapt.models.mmpose import MMPoseEstimator


class ViTPoseEstimator(MMPoseEstimator):
    """MMPose adapter configured with a ViTPose model definition."""
