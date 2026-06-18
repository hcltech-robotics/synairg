from __future__ import annotations

import importlib

import numpy as np
import pytest
from synairg_core.cli import app
from typer.testing import CliRunner
from video_gen.cli import (
    _compose_labeled_grid_frame,
    _grid_shape,
    _parse_material_map_specs,
    _parse_material_variants,
)

runner = CliRunner()


@pytest.mark.parametrize(
    "module_name",
    [
        "synairg_core",
        "volume_gen",
        "mesh_gen",
        "pose_gen",
        "video_gen",
        "trainer",
    ],
)
def test_package_modules_are_importable(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["volume", "--help"],
        ["volume", "import-local", "--help"],
        ["volume", "import-dicom", "--help"],
        ["volume", "list-tcia-series", "--help"],
        ["volume", "pull-tcia", "--help"],
        ["volume", "generate-maisi", "--help"],
        ["mesh", "--help"],
        ["mesh", "status", "--help"],
        ["mesh", "run", "--help"],
        ["pose", "--help"],
        ["video", "--help"],
        ["video", "render-scope", "--help"],
        ["video", "render-material-variants", "--help"],
        ["video", "render-material-maps", "--help"],
        ["video", "export-material-atlas", "--help"],
        ["video", "export-material-mdl", "--help"],
        ["train", "--help"],
        ["train", "build-bronchogen-manifest", "--help"],
        ["train", "download-bm-broncholc", "--help"],
        ["train", "audit-bm-broncholc", "--help"],
        ["train", "split-bm-broncholc", "--help"],
        ["train", "evaluate-temporal-consistency", "--help"],
        ["validate", "--help"],
        ["publish", "--help"],
    ],
)
def test_cli_help_commands_succeed(args: list[str]) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_cli_rejects_unknown_commands() -> None:
    result = runner.invoke(app, ["not-a-command"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_material_variant_parser_validates_unique_known_variants() -> None:
    assert _parse_material_variants("healthy, inflamed, smoker") == ("healthy", "inflamed", "smoker")
    with pytest.raises(ValueError, match="unsupported"):
        _parse_material_variants("healthy,unknown")
    with pytest.raises(ValueError, match="unique"):
        _parse_material_variants("healthy,healthy")


def test_material_map_specs_parse_labels_and_baseline(tmp_path) -> None:
    map_path = tmp_path / "secretions.json"
    map_path.write_text('{"schema_version":"1.0","name":"distal secretions"}\n', encoding="utf-8")

    specs = _parse_material_map_specs(f"baseline=none, secretions={map_path}, {map_path}")

    assert specs[0] == ("baseline", None)
    assert specs[1] == ("secretions", map_path)
    assert specs[2] == ("distal secretions", map_path)
    with pytest.raises(ValueError, match="unique"):
        _parse_material_map_specs("baseline=none,baseline=none")
    with pytest.raises(ValueError, match="does not exist"):
        _parse_material_map_specs("missing=/not/a/map.json")


def test_labeled_grid_frame_builds_stable_variant_layout() -> None:
    frames = [
        np.full((4, 6, 3), value, dtype=np.uint8)
        for value in (32, 64, 96, 128, 160)
    ]

    grid = _compose_labeled_grid_frame(frames, ("healthy", "inflamed", "smoker", "pale", "edematous"))

    assert _grid_shape(5) == (3, 2)
    assert grid.shape == (8, 18, 3)
    assert grid[3, 3].mean() >= 20
    assert grid[3, 9].mean() >= 55
    assert grid[3, 15].mean() >= 80
    assert grid[7, 3].mean() >= 120
    assert grid[7, 9].mean() >= 150
    assert grid[7, 15].tolist() == [0, 0, 0]
