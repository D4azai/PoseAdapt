"""RTMPose fallback adapter."""

from poseadapt.models.mmpose import MMPoseEstimator


class RTMPoseEstimator(MMPoseEstimator):
    """MMPose adapter configured with an RTMPose model definition."""
