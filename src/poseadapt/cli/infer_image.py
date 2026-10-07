"""Image inference command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2

from poseadapt.models.factory import create_estimator
from poseadapt.utils.config import load_config
from poseadapt.utils.logging import configure_logging
from poseadapt.visualization.skeleton import draw_pose_predictions

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""
    parser = argparse.ArgumentParser(description="Run MMPose image inference with PoseAdapt")
    parser.add_argument("--input", type=Path, required=True, help="Input image")
    parser.add_argument("--output", type=Path, required=True, help="Rendered output image")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--model", choices=("vitpose", "rtmpose"), help="Override model name")
    parser.add_argument("--model-config", type=Path, help="Local MMPose .py model config")
    parser.add_argument("--checkpoint", type=Path, help="Local official .pth checkpoint")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--json-output", type=Path, help="Optional keypoint JSON output")
    parser.add_argument("--threshold", type=float, help="Keypoint drawing threshold")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _value(cli_value: Any, config_value: Any) -> Any:
    return config_value if cli_value is None else cli_value


def run(args: argparse.Namespace) -> int:
    """Run inference from parsed arguments."""
    config = load_config(args.config)
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode the input image: {input_path}")

    mmpose_config = _value(args.model_config, model_config.get("config"))
    checkpoint = _value(args.checkpoint, model_config.get("checkpoint"))
    if not mmpose_config or not checkpoint:
        raise ValueError(
            "A local MMPose model config and checkpoint are required. Set model.config and "
            "model.checkpoint in YAML or pass --model-config and --checkpoint."
        )

    estimator = create_estimator(
        name=_value(args.model, model_config.get("name", "vitpose")),
        config_path=mmpose_config,
        checkpoint_path=checkpoint,
        device=_value(args.device, config.get("device", "auto")),
    )
    estimator.load_model()
    predictions = estimator.predict(image)
    threshold = float(_value(args.threshold, inference_config.get("keypoint_threshold", 0.3)))
    rendered = draw_pose_predictions(
        image,
        predictions,
        keypoint_threshold=threshold,
        draw_bbox=bool(inference_config.get("draw_bbox", True)),
        draw_labels=bool(inference_config.get("draw_labels", True)),
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rendered):
        raise OSError(f"OpenCV failed to write output image: {output_path}")

    if args.json_output is not None:
        json_path = args.json_output.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {"input": str(input_path), "predictions": [p.to_dict() for p in predictions]},
                indent=2,
            ),
            encoding="utf-8",
        )
    LOGGER.info("Wrote %d pose prediction(s) to %s", len(predictions), output_path)
    return 0


def main() -> int:
    """CLI entry point with concise error reporting."""
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    try:
        return run(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
