from pathlib import Path

import pytest

from poseadapt.utils.config import deep_merge, load_config


def test_deep_merge_keeps_unmodified_nested_values() -> None:
    result = deep_merge(
        {"model": {"name": "vitpose", "checkpoint": None}, "seed": 42},
        {"model": {"checkpoint": "model.pth"}},
    )
    assert result == {
        "model": {"name": "vitpose", "checkpoint": "model.pth"},
        "seed": 42,
    }


def test_load_config_resolves_relative_base(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("seed: 42\nmodel:\n  name: vitpose\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "base: base.yaml\nmodel:\n  name: rtmpose\n", encoding="utf-8"
    )
    assert load_config(tmp_path / "child.yaml") == {"seed": 42, "model": {"name": "rtmpose"}}


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
