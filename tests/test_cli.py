from pathlib import Path

from poseadapt.cli.infer_image import build_parser, main


def test_parser_requires_input_and_output() -> None:
    parser = build_parser()
    args = parser.parse_args(["--input", "in.jpg", "--output", "out.jpg"])
    assert args.input == Path("in.jpg")
    assert args.output == Path("out.jpg")


def test_main_reports_missing_input(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["poseadapt-infer-image", "--input", "missing.jpg", "--output", "out.jpg"],
    )
    assert main() == 2
